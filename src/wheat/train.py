"""Train only on frozen training IDs, and write predictions without scores."""

import argparse
import csv
import hashlib
import json
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
from torchvision.transforms import ColorJitter


class Images(Dataset):
    def __init__(self, root, ids, boxes, flip=0, vertical_flip=0, color_jitter=0):
        self.root, self.ids, self.boxes, self.flip = root, ids, boxes, flip
        self.vertical_flip = vertical_flip
        self.color_jitter = (ColorJitter(brightness=color_jitter, contrast=color_jitter,
                                        saturation=color_jitter) if color_jitter else None)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        key = self.ids[i]
        return read_image(self.root / "train" / f"{key}.jpg", self.boxes.get(key, []),
                          self.flip, self.vertical_flip, self.color_jitter)


def collate(samples):
    return tuple(zip(*samples))


def detection_loss(model, images, labels, device):
    loss = sum(model([image.to(device) for image in images],
                     [{k: v.to(device) for k, v in target.items()} for target in labels]).values())
    if not torch.isfinite(loss):
        raise ValueError("Non-finite detection loss")
    return loss


@torch.no_grad()
def validation_loss(model, loader, device, seed):
    # TorchVision produces detection losses through these training branches.
    # Keep the backbone (including BatchNorm) and the transforms in eval mode.
    modes = {module: module.training for module in model.modules()}
    cuda_devices = [device.index] if device.type == "cuda" else []
    try:
        model.eval()
        for module in (model, model.rpn, model.roi_heads):
            module.training = True
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            total = sum(detection_loss(model, images, labels, device).item() * len(images)
                        for images, labels in loader)
        return total / len(loader.dataset)
    finally:
        for module, training in modes.items():
            module.training = training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-source", choices=("competition", "gwhd2020-v4"), default="competition")
    parser.add_argument("--weights", type=Path, help="Predict from a saved checkpoint instead of training.")
    parser.add_argument("--partition", choices=("validation", "holdout"), default="validation")
    args = parser.parse_args()
    if args.partition == "holdout" and not args.weights:
        parser.error("Holdout inference requires an existing checkpoint.")
    config = json.loads(Path("configs/baseline.json").read_text())
    config.setdefault("vertical_flip", 0)
    config.setdefault("color_jitter", 0)
    config.setdefault("validation_loss", False)
    assert 0 <= config["horizontal_flip"] <= 1
    assert 0 <= config["vertical_flip"] <= 1 and 0 <= config["color_jitter"] <= 1
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
        if config["validation_loss"]:
            validation_loss(model, DataLoader([(sample[0], labels[0])], collate_fn=collate),
                            torch.device("cpu"), seed)
        model.eval()
        with torch.inference_mode():
            model(sample)
        return

    project = Path(os.environ["ML_LOOP_PROJECT"])
    out = Path(os.environ["ML_LOOP_ARTIFACTS"])
    remote = os.environ.get("ML_LOOP_REMOTE") == "kaggle"
    if remote:
        if args.data_source == "gwhd2020-v4":
            from zipfile import ZipFile

            dataset = Path("/kaggle/input")
            candidates = sorted(p.parent for p in dataset.rglob("train.csv")
                                if (p.parent / "GWHD2020-V4.txt").is_file())
            if not candidates:
                archives = sorted(dataset.rglob("gwhd2020-v4.zip"))
                if len(archives) != 1:
                    raise ValueError("Expected the mounted public GWHD archive")
                data = Path("/kaggle/temp/gwhd2020-v4")
                with ZipFile(archives[0]) as archive:
                    archive.extractall(data)
                candidates = [data]
        else:
            candidates = sorted(p.parent for p in Path("/kaggle/input").rglob("train.csv")
                                if p.parent.name == "global-wheat-detection")
        if len(candidates) != 1:
            raise ValueError(f"Expected one data root for {args.data_source}; found {candidates}")
        data = candidates[0]
        if not torch.cuda.is_available():
            raise RuntimeError("Kaggle GPU was requested but CUDA is unavailable.")
        device = torch.device("cuda:0")
    else:
        data = project / ("data/gwhd2020-v4/raw" if args.data_source == "gwhd2020-v4" else "data/raw")
        device = torch.device("cpu")
    split_path = ("data/gwhd2020-v4/split.json" if args.data_source == "gwhd2020-v4"
                  else "data/prepared/split.json")
    if remote and args.data_source == "gwhd2020-v4":
        split_path = "data/gwhd2020-v4/split-remote.json"
    split = json.loads((project / split_path).read_text())
    checksum_key = ("remote_train_csv_sha256" if remote and args.data_source == "gwhd2020-v4"
                    else "csv_sha256")
    if hashlib.sha256((data / "train.csv").read_bytes()).hexdigest() != split[checksum_key]:
        raise ValueError("Training annotation file does not match the frozen split.")
    train_ids = set(split["partitions"]["train"])
    inference_ids = split["partitions"][args.partition]
    for key in sorted(train_ids | set(inference_ids)):
        if sha(data / "train" / f"{key}.jpg") != split["images"][key]:
            raise ValueError(f"Image does not match frozen data: {key}")
    valid_loss_ids = set(inference_ids) if not args.weights and config["validation_loss"] else set()
    if valid_loss_ids and (train_ids & valid_loss_ids or remote and args.data_source == "gwhd2020-v4"):
        raise ValueError("Validation loss requires separate validation annotations in the training environment.")
    boxes = defaultdict(list)
    with (data / "train.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["image_id"] in train_ids or row["image_id"] in valid_loss_ids:
                x, y, w, h = json.loads(row["bbox"])
                boxes[row["image_id"]].append([x, y, x + w, y + h])
    if valid_loss_ids - boxes.keys():
        raise ValueError("Missing validation annotations for validation loss.")

    model = create_model(config, pretrained=not args.weights).to(device)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    else:
        loader = DataLoader(Images(data, sorted(train_ids), boxes, config["horizontal_flip"],
                                   config["vertical_flip"], config["color_jitter"]),
                            batch_size=config["batch_size"], shuffle=True, collate_fn=collate,
                            generator=torch.Generator().manual_seed(seed))
        valid_loader = None
        if config["validation_loss"]:
            if not valid_loss_ids:
                raise ValueError("Validation split is empty.")
            valid_loader = DataLoader(Images(data, sorted(valid_loss_ids), boxes),
                                      batch_size=config["batch_size"], collate_fn=collate)
        opt = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        for epoch in range(config["epochs"]):
            model.train()
            loss_total = 0
            for images, labels in loader:
                loss = detection_loss(model, images, labels, device)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_total += loss.item() * len(images)
            progress = f"Epoch {epoch + 1}: loss {loss_total / len(loader.dataset):.5f}"
            if valid_loader is not None:
                progress += f" val_loss {validation_loss(model, valid_loader, device, seed):.5f}"
            print(f"{progress} optimizer_steps {(epoch + 1) * len(loader)}", flush=True)
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
