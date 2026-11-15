"""Full-image and overlapping tile proposals for wheat inference."""

import math

import numpy as np

from fusion import fuse
from metric import validate_predictions


def tile_predictions(model, image, spec, batch_size=4):
    height, width = image.shape[-2:]
    window = spec["window_fraction"]
    margin = spec["seam_margin"]
    if not 0.5 < window < 1 or not 0 <= margin < window - 0.5:
        raise ValueError("Expected overlapping tiles with a smaller seam margin.")
    tw, th = math.ceil(window * width), math.ceil(window * height)
    origins = [(x, y) for y in (0, height - th) for x in (0, width - tw)]
    rows = []
    for offset in range(0, len(origins), batch_size):
        positions = origins[offset:offset + batch_size]
        detections = model([image[:, y:y + th, x:x + tw] for x, y in positions])
        for (x, y), result in zip(positions, detections, strict=True):
            for score, (left, top, right, bottom) in zip(
                result["scores"].tolist(), result["boxes"].tolist(), strict=True
            ):
                if ((x > 0 and left < margin * width) or
                    (y > 0 and top < margin * height) or
                    (x + tw < width and right > tw - margin * width) or
                    (y + th < height and bottom > th - margin * height)):
                    continue
                if (right - left) * (bottom - top) <= spec["max_area"] * width * height:
                    rows.append([score, x + left, y + top, x + right, y + bottom])
    return fuse([rows], width, height, [1], spec["iou"])


def overlaps(boxes, other):
    lo = np.maximum(boxes[:, None, :2], other[None, :, :2])
    hi = np.minimum(boxes[:, None, 2:], other[None, :, 2:])
    joint = (hi - lo).clip(min=0).prod(axis=2)
    area = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    other_area = (other[:, 2:] - other[:, :2]).prod(axis=1)
    return joint / (area[:, None] + other_area[None, :] - joint)


def as_array(rows):
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


def tile_rescue(baseline, tiles, spec):
    if not tiles:
        return baseline
    guesses = as_array(tiles)
    present = (overlaps(guesses[:, 1:], as_array(baseline)[:, 1:]).max(axis=1)
               if baseline else np.zeros(len(tiles)))
    return [*baseline, *guesses[(guesses[:, 0] >= spec["rescue_confidence"]) &
                               (present < spec["rescue_iou"])].tolist()]


def quality_features(candidates, sources, width, height):
    boxes = candidates[:, 1:]
    coords = boxes / [width, height, width, height]
    span = coords[:, 2:] - coords[:, :2]
    border = np.minimum(coords[:, :2], 1 - coords[:, 2:]).min(axis=1)
    columns = [candidates[:, 0], np.log(span[:, 0]), np.log(span[:, 1]), border]
    for rows in sources:
        raw = as_array(rows)
        support = np.zeros(len(candidates))
        agreement = np.zeros(len(candidates))
        if len(raw):
            similarity = overlaps(boxes, raw[:, 1:])
            best = similarity.argmax(axis=1)
            agreement = similarity[np.arange(len(best)), best]
            support = np.where(agreement >= 0.25, raw[best, 0], 0)
        columns.extend((agreement, support, agreement * support,
                        np.full(len(candidates), np.log1p(len(raw)))))
    return np.column_stack(columns)


def quality_candidates(global_rows, tiles, width, height, weights, threshold, with_tiles):
    sources = [*global_rows, tiles] if with_tiles else global_rows
    global_fused = fuse(global_rows, width, height, weights, threshold)
    # Keep alternative coordinates until the predicted quality determines NMS order.
    candidates = as_array([*global_fused, *(box for rows in sources for box in rows)])
    return candidates, quality_features(candidates, sources, width, height)


def rescore(candidates, features, width, height, spec):
    if not len(candidates):
        return []
    scores = np.clip(spec["intercept"] + features @ spec["coefficients"], 0, 1)
    remaining = np.argsort(-scores, kind="stable")
    remaining = remaining[scores[remaining] >= spec["confidence"]]
    selected = []
    while len(remaining):
        first, remaining = remaining[0], remaining[1:]
        selected.append(first)
        similarity = overlaps(candidates[[first], 1:], candidates[remaining, 1:])[0]
        remaining = remaining[similarity < spec["nms_iou"]]
    rows = np.column_stack((scores[selected], candidates[selected, 1:])).tolist()
    validate_predictions(rows, width, height)
    return rows


def refine_with_tiles(baseline, candidates, features, tiles, spec):
    """Use learned box quality to refine supported detections and rescue novel ones."""
    if not tiles:
        return baseline
    tile_count = len(tiles)
    tile_boxes = as_array(tiles)[:, 1:]
    quality = np.clip(spec["intercept"] + features @ spec["coefficients"], 0, 1)
    tile_quality = quality[-tile_count:]
    reference = as_array(baseline)
    similarities = overlaps(reference[:, 1:], tile_boxes)
    updated = [row.copy() for row in baseline]
    if len(reference):
        owner = similarities.argmax(axis=0)
        nearest = similarities.max(axis=0)
        current = quality[:len(reference)]
        # The baseline is the confidence-filtered prefix of the sorted WBF candidates.
        if not np.allclose(reference, candidates[:len(reference)]):
            raise ValueError("Expected baseline detections to prefix the quality candidates.")
        for index, row in enumerate(reference):
            supported = (owner == index) & (similarities[index] >= 0.50)
            eligible = np.flatnonzero(supported & (tile_quality >= current[index] + 0.05))
            if len(eligible):
                best = eligible[tile_quality[eligible].argmax()]
                updated[index][1:] = ((row[1:] + tile_boxes[best]) / 2).tolist()
    else:
        nearest = np.zeros(tile_count)
    for index in np.flatnonzero((nearest < 0.30) & (tile_quality >= spec["confidence"])):
        updated.append([float(tile_quality[index]), *tile_boxes[index].tolist()])
    return updated
