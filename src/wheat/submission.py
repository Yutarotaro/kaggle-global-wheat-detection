"""Write Kaggle predictions from frozen detector checkpoints."""

import csv
import json
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from data import sha
from fusion import fuse
from metric import validate_predictions
from model import create_model
from proposals import quality_candidates, rescore, tile_predictions, tile_rescue


def kaggle_inputs(mounted, spec):
    roots = sorted(path.parent for path in mounted.rglob("sample_submission.csv")
                   if path.parent.name == "global-wheat-detection")
    if len(roots) != 1:
        raise ValueError("Expected one mounted competition dataset.")
    by_hash = {sha(path): path for path in mounted.rglob("weights.pt")}
    return roots[0], [by_hash[model["sha256"]] for model in spec["models"]]


def prediction_string(predictions, width, height):
    validate_predictions(predictions, width, height)
    tokens = []
    for score, x1, y1, x2, y2 in predictions:
        left, top = math.floor(x1), math.floor(y1)
        right, bottom = math.ceil(x2), math.ceil(y2)
        tokens.append(f"{score:.8f} {left} {top} {right-left} {bottom-top}")
    return " ".join(tokens)


def predict_views(model, tensor, flips, iou):
    height, width = tensor.shape[-2:]
    predictions = []
    for view in ("original", *flips):
        if view == "original":
            result = model([tensor])[0]
            boxes = result["boxes"]
        else:
            dimension, extent = {"horizontal": (-1, width), "vertical": (-2, height)}[view]
            result = model([tensor.flip(dims=(dimension,))])[0]
            boxes = result["boxes"].clone()
            columns = (0, 2) if view == "horizontal" else (1, 3)
            boxes[:, columns] = extent - boxes[:, columns[::-1]]
        predictions.append([[score, *box] for score, box in
                            zip(result["scores"].tolist(), boxes.tolist(), strict=True)])
    return fuse(predictions, width, height, [1] * len(predictions), iou) if flips else predictions[0]


@torch.inference_mode()
def predict(data_root, checkpoints, spec, output, device):
    if len(checkpoints) != len(spec["models"]):
        raise ValueError("Expected one checkpoint per model.")
    models = []
    for path, conditions in zip(checkpoints, spec["models"], strict=True):
        if sha(path) != conditions["sha256"]:
            raise ValueError(f"Checkpoint mismatch for {conditions['trial_id']}.")
        model = create_model(conditions["config"], pretrained=False).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        models.append(model.eval())
    paths = {path.stem: path for path in (data_root / "test").glob("*.jpg")}
    views = [conditions.get("tta_flips", spec.get("tta_flips", [])) for conditions in spec["models"]]
    for flips in views:
        if len(set(flips)) != len(flips) or set(flips) - {"horizontal", "vertical"}:
            raise ValueError("Expected a unique list of horizontal or vertical TTA flips.")
    with (data_root / "sample_submission.csv").open(newline="") as handle:
        ids = [row["image_id"] for row in csv.DictReader(handle)]
    if not ids or len(ids) != len(paths) or set(ids) != paths.keys():
        raise ValueError("Sample submission and test images must contain the same unique IDs.")
    temporary = output.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_id", "PredictionString"))
        for key in ids:
            with Image.open(paths[key]) as image:
                width, height = image.size
                tensor = pil_to_tensor(image.convert("RGB")).float().div_(255).to(device)
            rows = [predict_views(model, tensor, flips, spec.get("tta_iou", 0.55))
                    for model, flips in zip(models, views, strict=True)]
            tiles = spec.get("tiles")
            tile_rows = tile_predictions(models[tiles["model_index"]], tensor, tiles) if tiles else []
            quality = spec.get("quality")
            if quality:
                candidates, features = quality_candidates(rows, tile_rows, width, height,
                                                         spec["weights"], spec["iou"], bool(tiles))
                detections = rescore(candidates, features, width, height, quality)
                writer.writerow((key, prediction_string(detections, width, height)))
                continue
            if len(rows) == 1:
                detections = rows[0]
            else:
                detections = fuse(rows, width, height, spec["weights"], spec["iou"])
            detections = [box for box in detections if box[0] >= spec["confidence"]]
            if tiles:
                detections = tile_rescue(detections, tile_rows, tiles)
            writer.writerow((key, prediction_string(detections, width, height)))
    temporary.replace(output)
    print(f"Validated and wrote {len(ids)} predictions")


def run_kaggle(spec_path):
    spec = json.loads(spec_path.read_text())
    data, checkpoints = kaggle_inputs(Path("/kaggle/input"), spec)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict(data, checkpoints, spec, Path("/kaggle/working/submission.csv"), device)
