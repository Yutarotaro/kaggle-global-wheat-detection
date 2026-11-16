"""Compare training configs locally on the same limited data and update count."""

import argparse
import json
import random
import subprocess
import time
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

from sweep_confidence import PROJECT
from data import load_dataset, sha
from model import create_model
from train import BackgroundSampler, Images, collate, detection_loss, load_backgrounds, predict, validation_diagnostics

RAW = PROJECT / "data/raw"
SPLIT = PROJECT / "data/prepared/split.json"


def run_pilot(config, args, train_keys, validation, labels, backgrounds):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    images = Images(RAW, train_keys, {key: labels[key]["boxes"] for key in train_keys},
                    config["horizontal_flip"], config["vertical_flip"], config["color_jitter"], config["mosaic"],
                    crop_probability=config["zoom_crop_probability"], seed=args.seed, backgrounds=backgrounds,
                    zoom_out_probability=config.get("zoom_out_probability", 0))
    sampler = RandomSampler(range(len(train_keys)), num_samples=len(train_keys),
                            generator=torch.Generator().manual_seed(args.seed))
    batches = BackgroundSampler(sampler, len(train_keys), len(train_keys), len(backgrounds),
                                config["background_fraction"], args.seed)
    loader = DataLoader(images, batch_size=args.batch_size, sampler=batches, collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed))
    if args.steps > len(loader):
        raise ValueError("The pilot can use up to one fixed training epoch per config.")
    device = torch.device(args.device)
    model = create_model(config, pretrained=True).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config.get("weight_decay", .01))
    losses = []
    started = time.monotonic()
    for step, (batch, targets) in enumerate(islice(loader, args.steps), 1):
        optimizer.zero_grad(set_to_none=True)
        loss = detection_loss(model, batch, targets, device)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 16 == 0 or step == args.steps:
            print(f"{config['run_label']} {step}/{args.steps} loss={losses[-1]:.5f}", flush=True)
    training_seconds = time.monotonic() - started
    predictions = predict(model, RAW, validation, device)
    score, metrics = validation_diagnostics(
        predictions, {key: labels[key]["boxes"] for key in validation},
        {key: (labels[key]["source"], labels[key]["width"] * labels[key]["height"]) for key in validation})
    return {"training_seconds": training_seconds, "train_loss": sum(losses) / len(losses),
            "train_zoom_outs": images.zoom_out_count, "validation_pilot_score": score, **metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--images-per-source", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.images_per_source) <= 0:
        parser.error("Steps, batch size, and images per source must be positive.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this PyTorch environment.")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT, text=True).strip():
        raise ValueError("Run the pilot from a clean committed tree.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    split = json.loads(SPLIT.read_text())
    labels, hashes = load_dataset(RAW)
    train_keys = sorted(split["partitions"]["train"])
    development = sorted(split["partitions"]["validation"])
    if sha(RAW / "train.csv") != split["csv_sha256"] or any(
        hashes[key] != split["images"][key] for key in train_keys + development
    ):
        raise ValueError("Pilot data does not match the fixed train and validation partitions.")
    rng = random.Random(args.seed)
    sources = sorted({labels[key]["source"] for key in development})
    validation = sorted(key for source in sources for key in rng.sample(
        [key for key in development if labels[key]["source"] == source], args.images_per_source))
    backgrounds = load_backgrounds(PROJECT / "data/prepared/background-crops.json", SPLIT,
                                   {key: (labels[key]["width"], labels[key]["height"]) for key in train_keys},
                                   {key: labels[key]["boxes"] for key in train_keys})
    torch.set_num_threads(4)
    report = {"git_commit": commit, "split_sha256": sha(SPLIT), "torch": torch.__version__, "device": args.device,
              "seed": args.seed, "steps": args.steps, "batch_size": args.batch_size,
              "images_per_source": args.images_per_source, "results": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in args.configs:
        config = json.loads(path.read_text())
        if args.image_size is not None:
            config["image_size"] = args.image_size
        result = run_pilot(config, args, train_keys, validation, labels, backgrounds)
        report["results"][config["run_label"]] = dict(result, config_sha256=sha(path), image_size=config["image_size"])
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)
        print(config["run_label"], result, flush=True)
        if args.device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
