"""Train only on frozen training IDs, and write predictions without scores."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
from collections import defaultdict
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from data import sha
from model import create_model
from preprocess import read_image


class Images(Dataset):
    def __init__(self, root, ids, boxes, flip=0):
        self.root, self.ids, self.boxes, self.flip = root, ids, boxes, flip

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        key = self.ids[i]
        return read_image(self.root / "train" / f"{key}.jpg", self.boxes.get(key, []), self.flip)


def collate(samples):
    return tuple(zip(*samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--weights", type=Path, help="Predict from a saved checkpoint instead of training.")
    parser.add_argument("--partition", choices=("validation", "holdout"), default="validation")
    args = parser.parse_args()
    if args.partition == "holdout" and not args.weights:
        parser.error("Holdout inference requires an existing checkpoint.")
    config = json.loads(Path("configs/baseline.json").read_text())
    assert 0 <= config["horizontal_flip"] <= 1
    assert 0 < config["confidence"] < 1 and 0 < config["nms"] < 1
    assert 256 <= config["image_size"] <= 1280 and 1 <= config["epochs"] <= 15
    assert 1 <= config["batch_size"] <= 16 and 0 < config["learning_rate"] < 1
    seed = int(os.environ.get("ML_LOOP_SEED", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if args.smoke:
        torch.set_num_threads(2)
        model = create_model(config, pretrained=False)
        sample = [torch.zeros(3, 256, 256)]
        labels = [{"boxes": torch.tensor([[30., 30., 160., 150.]]), "labels": torch.tensor([1])}]
        loss = sum(model(sample, labels).values())
        if not torch.isfinite(loss):
            raise ValueError("Smoke loss is not finite.")
        loss.backward()
        model.eval()
        with torch.inference_mode():
            model(sample)
        return

    project = Path(os.environ["ML_LOOP_PROJECT"])
    out = Path(os.environ["ML_LOOP_ARTIFACTS"])
    if os.environ.get("ML_LOOP_REMOTE") == "kaggle":
        candidates = sorted(Path("/kaggle/input").rglob("train.csv"))
        candidates = [p.parent for p in candidates if p.parent.name == "global-wheat-detection"]
        if len(candidates) != 1:
            raise ValueError(f"Expected one mounted competition data root; found {candidates}")
        data = candidates[0]
        if not torch.cuda.is_available():
            raise RuntimeError("Kaggle GPU was requested but CUDA is unavailable.")
        device = torch.device("cuda:0")
    else:
        data = project / "data/raw"
        device = torch.device("cpu")
    split = json.loads((project / "data/prepared/split.json").read_text())
    if hashlib.sha256((data / "train.csv").read_bytes()).hexdigest() != split["csv_sha256"]:
        raise ValueError("Training annotation file does not match the frozen split.")
    train_ids = set(split["partitions"]["train"])
    inference_ids = split["partitions"][args.partition]
    for key in sorted(train_ids | set(inference_ids)):
        if sha(data / "train" / f"{key}.jpg") != split["images"][key]:
            raise ValueError(f"Image does not match frozen data: {key}")
    boxes = defaultdict(list)
    with (data / "train.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["image_id"] in train_ids:
                x, y, w, h = json.loads(row["bbox"])
                boxes[row["image_id"]].append([x, y, x + w, y + h])

    model = create_model(config, pretrained=not args.weights).to(device)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    else:
        loader = DataLoader(Images(data, sorted(train_ids), boxes, config["horizontal_flip"]),
                            batch_size=config["batch_size"], shuffle=True, collate_fn=collate,
                            generator=torch.Generator().manual_seed(seed))
        opt = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        for epoch in range(config["epochs"]):
            model.train()
            loss_total = 0
            for images, labels in loader:
                loss = sum(model([i.to(device) for i in images],
                                 [{k: v.to(device) for k, v in t.items()} for t in labels]).values())
                if not math.isfinite(loss.item()):
                    raise ValueError("Non-finite training loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_total += loss.item()
            print(f"Epoch {epoch + 1}: loss {loss_total / len(loader):.5f}", flush=True)
            torch.save(model.state_dict(), out / "weights.pt")

    ids = inference_ids
    model.eval()
    predictions = {}
    with torch.inference_mode():
        for image_id in ids:
            image, _ = read_image(data / "train" / f"{image_id}.jpg", [], 0)
            pred = model([image.to(device)])[0]
            predictions[image_id] = [[score, *box] for score, box in
                                     zip(pred["scores"].tolist(), pred["boxes"].tolist(), strict=True)]
    (out / "predictions.json").write_text(json.dumps({"images": predictions, "environment": {
        "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
        "device": str(device), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }}, allow_nan=False))


if __name__ == "__main__":
    main()
