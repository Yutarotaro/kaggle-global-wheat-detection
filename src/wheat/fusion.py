"""Weighted Boxes Fusion for validated wheat detections in image coordinates."""

from ensemble_boxes.ensemble_boxes_wbf import weighted_boxes_fusion

from metric import validate_predictions


def fuse(rows, width, height, weights, threshold):
    scale = (width, height, width, height)
    for guesses in rows:
        validate_predictions(guesses, width, height)
    # Zero-confidence boxes carry no weight when averaging positions.
    rows = [[box for box in guesses if box[0] > 0] for guesses in rows]
    boxes, scores, _ = weighted_boxes_fusion(
        [[[value / size for value, size in zip(box[1:], scale)] for box in guesses] for guesses in rows],
        [[box[0] for box in guesses] for guesses in rows],
        [[1] * len(guesses) for guesses in rows],
        weights=weights, iou_thr=threshold, conf_type="box_and_model_avg",
    )
    # The library uses float32 cluster averages, which can round above 1.
    return [[min(float(conf), 1.0), *(min(float(value), 1.0) * size for value, size in zip(box, scale))]
            for box, conf in zip(boxes, scores)]
