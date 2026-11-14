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
            results = [model([tensor])[0] for model in models]
            rows = [[[score, *box] for score, box in zip(result["scores"].tolist(), result["boxes"].tolist(),
                                                          strict=True)] for result in results]
            if len(rows) == 1:
                detections = rows[0]
            else:
                detections = fuse(rows, width, height, spec["weights"], spec["iou"])
            detections = [box for box in detections if box[0] >= spec["confidence"]]
            writer.writerow((key, prediction_string(detections, width, height)))
    temporary.replace(output)
    print(f"Validated and wrote {len(ids)} predictions")


def run_kaggle(spec_path):
    spec = json.loads(spec_path.read_text())
    data, checkpoints = kaggle_inputs(Path("/kaggle/input"), spec)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict(data, checkpoints, spec, Path("/kaggle/working/submission.csv"), device)
