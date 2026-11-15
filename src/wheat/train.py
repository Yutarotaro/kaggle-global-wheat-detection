"""Train only on frozen training IDs, and write predictions without scores."""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
from collections import defaultdict
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision.transforms import ColorJitter

from data import sha
from metric import area, image_score, match
from model import create_model
from preprocess import mosaic, read_image, zoom_crop
from preview_samples import select_preview_ids
from supplement import unpack_supplement


class Images(Dataset):
    def __init__(self, root, ids, boxes, flip=0, vertical_flip=0, color_jitter=0, mosaic_probability=0,
                 extra_root=None, crop_probability=0, seed=42):
        self.root, self.ids, self.boxes, self.flip = root, ids, boxes, flip
        self.vertical_flip = vertical_flip
        self.mosaic_probability = mosaic_probability
        self.extra_root = extra_root
        self.crop_probability = crop_probability
        self.crop_rng = random.Random(seed)
        self.color_jitter = (ColorJitter(brightness=color_jitter, contrast=color_jitter,
                                        saturation=color_jitter) if color_jitter else None)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self.mosaic_probability and random.random() < self.mosaic_probability:
            partners = random.sample(range(len(self.ids) - 1), 3)
            return mosaic([self.load(i), *(self.load(j + (j >= i)) for j in partners)])
        image, target = self.load(i)
        if self.crop_probability and self.crop_rng.random() < self.crop_probability:
            return zoom_crop(image, target, self.crop_rng)
        return image, target

    def load(self, i):
        key = self.ids[i]
        root = self.extra_root if self.extra_root and key.startswith("gwhd2021-") else self.root
        return read_image(root / "train" / f"{key}.jpg", self.boxes.get(key, []),
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
        if hasattr(model, "rpn"):
            for module in (model, model.rpn, model.roi_heads):
                module.training = True
        else:
            model.loss_bench.training = True
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


@torch.inference_mode()
def predict(model, data, ids, device):
    model.eval()
    predictions = {}
    for image_id in ids:
        image, _ = read_image(data / "train" / f"{image_id}.jpg", [], 0)
        result = model([image.to(device)])[0]
        predictions[image_id] = [[score, *box] for score, box in
                                 zip(result["scores"].tolist(), result["boxes"].tolist(), strict=True)]
    return predictions


def validation_diagnostics(predictions, boxes, metadata):
    scores = defaultdict(list)
    small_hits = small_total = false_positives = duplicates = 0
    for key, guesses in predictions.items():
        source, pixels = metadata[key]
        truth = boxes[key]
        scores[source].append(image_score(truth, guesses))
        _, extras, repeated, misses = match(truth, guesses, 0.50)
        small = {index for index, box in enumerate(truth) if area(box) < .002 * pixels}
        small_hits += len(small.difference(misses))
        small_total += len(small)
        false_positives += len(extras)
        duplicates += len(repeated)
    metrics = {f"valid_score_by_source.{source}": sum(values) / len(values)
               for source, values in scores.items()}
    metrics.update(valid_false_positives_iou_50=false_positives, valid_duplicates_iou_50=duplicates)
    if small_total:
        metrics["valid_small_recall_iou_50"] = small_hits / small_total
    overall = sum(map(sum, scores.values())) / len(predictions)
    return overall, metrics


def main(resources):
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    parser.add_argument("--data-source", choices=("competition", "gwhd2020-v4"), default="competition")
    parser.add_argument("--weights", type=Path, help="Predict from a saved checkpoint instead of training.")
    parser.add_argument("--partition", choices=("validation", "holdout"), default="validation")
    parser.add_argument("--epoch-previews", action="store_true", help="Save predictions for four fixed validation images.")
    parser.add_argument("--mons-supplement", type=Path, help="Manifest for the optional Arvalis_10 training data.")
    parser.add_argument("--training-supplement", type=Path, help="Pinned supplement used when enabled in the config.")
    parser.add_argument("--epoch-scores", action="store_true", help="Score fixed validation after every epoch.")
    args = parser.parse_args()
    if args.partition == "holdout" and not args.weights:
        parser.error("Holdout inference requires an existing checkpoint.")
    config = json.loads(args.config.read_text())
    if args.mons_supplement and args.training_supplement:
        parser.error("Provide one training supplement manifest.")
    if config.get("use_training_supplement") and not (args.training_supplement or args.weights):
        parser.error("The config requires --training-supplement.")
    supplement = args.mons_supplement or (args.training_supplement if config.get("use_training_supplement") else None)
    if (supplement or args.epoch_scores) and (args.data_source != "competition" or args.weights):
        parser.error("Supplements and epoch scores apply only to competition model training.")
    config.setdefault("vertical_flip", 0)
    config.setdefault("color_jitter", 0)
    config.setdefault("validation_loss", False)
    config.setdefault("mosaic", 0)
    config.setdefault("zoom_crop_probability", 0)
    if not 0 <= config["mosaic"] <= 1:
        raise ValueError("Mosaic probability must be in [0, 1].")
    if not 0 <= config["zoom_crop_probability"] <= 1 or config["zoom_crop_probability"] and config["mosaic"]:
        raise ValueError("Zoom-crop probability must be in [0, 1] and requires Mosaic to be disabled.")
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
        sample = [torch.zeros(3, 256, 256) for _ in range(2)]
        labels = [{"boxes": torch.tensor([[30., 30., 160., 150.]]), "labels": torch.tensor([1])}
                  for _ in sample]
        if config["mosaic"]:
            sample[0], labels[0] = mosaic([(sample[0], labels[0])] * 4)
        if config["zoom_crop_probability"]:
            sample[0], labels[0] = zoom_crop(sample[0], labels[0], random.Random(seed))
        loss = sum(model(sample, labels).values())
        if not torch.isfinite(loss):
            raise ValueError("Smoke loss is not finite.")
        loss.backward()
        if config["validation_loss"]:
            validation_loss(model, DataLoader(list(zip(sample, labels, strict=True)), collate_fn=collate),
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
    valid_loss_ids = set(inference_ids) if not args.weights and (config["validation_loss"] or args.epoch_scores) else set()
    if valid_loss_ids and (train_ids & valid_loss_ids or remote and args.data_source == "gwhd2020-v4"):
        raise ValueError("Validation loss requires separate validation annotations in the training environment.")
    boxes = defaultdict(list)
    valid_metadata = {}
    with (data / "train.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["image_id"] in train_ids or row["image_id"] in valid_loss_ids:
                x, y, w, h = json.loads(row["bbox"])
                boxes[row["image_id"]].append([x, y, x + w, y + h])
                if args.epoch_scores and row["image_id"] in valid_loss_ids:
                    valid_metadata[row["image_id"]] = row["source"], int(row["width"]) * int(row["height"])
    if valid_loss_ids - boxes.keys():
        raise ValueError("Missing validation annotations for validation loss.")

    extra = {}
    extra_root = None
    if supplement:
        manifest_path = project / supplement
        archive_name = json.loads(manifest_path.read_text()).get("archive_file", "gwhd2021-mons.zip")
        if not archive_name or Path(archive_name).name != archive_name:
            raise ValueError("Expected a supplement archive filename.")
        archives = (list(Path("/kaggle/input").rglob(archive_name)) if remote else
                    [manifest_path.with_name(archive_name)])
        extra_root, extra = unpack_supplement(manifest_path, archives, project / split_path, resources)
        if set(extra).intersection(split["images"]):
            raise ValueError("Supplement contains a competition image ID.")
        for key in (min(extra), max(extra)):
            with Image.open(extra_root / "train" / f"{key}.jpg") as image:
                image.load()
        boxes.update(extra)
        print(f"Verified and opened training supplement with {len(extra)} images.", flush=True)

    predictions = None
    model = create_model(config, pretrained=not args.weights).to(device)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    else:
        if config["mosaic"] and len(train_ids) < 4:
            raise ValueError("Mosaic requires at least four training images.")
        dataset = Images(data, sorted(train_ids | extra.keys()), boxes, config["horizontal_flip"],
                         config["vertical_flip"], config["color_jitter"], config["mosaic"], extra_root,
                         crop_probability=config["zoom_crop_probability"], seed=seed)
        sampler = (RandomSampler(dataset, num_samples=len(train_ids), generator=torch.Generator().manual_seed(seed))
                   if extra else None)
        loader = DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler, shuffle=sampler is None,
                            collate_fn=collate, generator=torch.Generator().manual_seed(seed))
        valid_loader = None
        if config["validation_loss"]:
            if not valid_loss_ids:
                raise ValueError("Validation split is empty.")
            valid_loader = DataLoader(Images(data, sorted(valid_loss_ids), boxes),
                                      batch_size=config["batch_size"], collate_fn=collate)
        opt = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        preview_ids = select_preview_ids(inference_ids) if args.epoch_previews else []
        if preview_ids:
            preview_dir = out / "epoch-previews"
            preview_dir.mkdir(exist_ok=True)
        best_score = float("-inf")
        best_epoch = None
        for epoch in range(config["epochs"]):
            model.train()
            loss_total = 0
            for images, labels in loader:
                loss = detection_loss(model, images, labels, device)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                loss_total += loss.item() * len(images)
            progress = f"Epoch {epoch + 1}: loss {loss_total / len(train_ids):.5f}"
            if valid_loader is not None:
                progress += f" val_loss {validation_loss(model, valid_loader, device, seed):.5f}"
            progress += f" optimizer_steps {(epoch + 1) * len(loader)}"
            epoch_predictions = None
            if args.epoch_scores:
                epoch_predictions = predict(model, data, inference_ids, device)
                score, diagnostics = validation_diagnostics(epoch_predictions, boxes, valid_metadata)
                progress += f" valid_wheat_image_precision {score:.9f}"
                if score > best_score:
                    best_score, best_epoch = score, epoch + 1
                    predictions = epoch_predictions
                    torch.save(model.state_dict(), out / "weights.pt")
            else:
                torch.save(model.state_dict(), out / "weights.pt")
            if preview_ids:
                preview = ({key: epoch_predictions[key] for key in preview_ids} if epoch_predictions is not None
                           else predict(model, data, preview_ids, device))
                (preview_dir / f"epoch-{epoch + 1:03d}.json").write_text(json.dumps(preview, allow_nan=False))
            print(progress, flush=True)
            if args.epoch_scores:
                print(f"Epoch diagnostics {epoch + 1}: {json.dumps(diagnostics, allow_nan=False)}", flush=True)
        if args.epoch_scores:
            torch.save(model.state_dict(), out / "weights-last.pt")
            (out / "checkpoint-selection.json").write_text(
                json.dumps({"best_epoch": best_epoch, "best_score": best_score, "last_epoch": config["epochs"]},
                           allow_nan=False) + "\n")

    if predictions is None:
        predictions = predict(model, data, inference_ids, device)
    environment = {
        "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
        "device": str(device), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    if config.get("detector") == "efficientdet-d2":
        environment.update(effdet=version("effdet"), timm=version("timm"))
    (out / "predictions.json").write_text(json.dumps({"images": predictions, "environment": environment}, allow_nan=False))


if __name__ == "__main__":
    with ExitStack() as resources:
        main(resources)
