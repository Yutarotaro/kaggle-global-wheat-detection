"""Image-averaged wheat precision over IoU thresholds 0.50–0.75."""

import math

THRESHOLDS = tuple(n / 100 for n in range(50, 76, 5))


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return overlap / (area(a) + area(b) - overlap)


def validate_predictions(rows, width, height):
    if not isinstance(rows, list):
        raise ValueError("Predictions must be a list.")
    for row in rows:
        if not isinstance(row, list) or len(row) != 5 or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in row
        ):
            raise ValueError(f"Expected finite [confidence,x1,y1,x2,y2]: {row!r}")
        confidence, x1, y1, x2, y2 = row
        if not (0 <= confidence <= 1 and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Invalid prediction: {row!r}")


def match(actual, predicted, threshold):
    remaining = set(range(len(actual)))
    hits, extras, duplicates = [], [], []
    for j in sorted(range(len(predicted)), key=lambda n: -predicted[n][0]):
        box = predicted[j][1:]
        overlaps = [iou(box, truth) for truth in actual]
        i = max(remaining, key=lambda k: (overlaps[k], -k), default=None)
        if i is not None and overlaps[i] > threshold:
            remaining.remove(i)
            hits.append(j)
        elif max(overlaps, default=0) > threshold:
            duplicates.append(j)
        else:
            extras.append(j)
    return hits, extras, duplicates, sorted(remaining)


def image_score(actual, predicted):
    if not actual and not predicted:
        return None
    scores = []
    for threshold in THRESHOLDS:
        hits, extra, duplicates, misses = match(actual, predicted, threshold)
        scores.append(len(hits) / (len(hits) + len(extra) + len(duplicates) + len(misses)))
    return sum(scores) / len(scores)
