"""Validate annotations and freeze a split by acquisition source."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path

from PIL import Image


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_dataset(root):
    images = {}
    with (root / "train.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            image_id, source = row["image_id"], row["source"]
            if not image_id or not source or Path(image_id).name != image_id:
                raise ValueError(f"Invalid image id or source: {image_id!r}, {source!r}")
            width, height = int(row["width"]), int(row["height"])
            if min(width, height) <= 0:
                raise ValueError(f"Invalid image dimensions: {image_id}")
            x, y, w, h = map(float, json.loads(row["bbox"]))
            if not all(map(math.isfinite, [x, y, w, h])) or min(x, y) < 0 or min(w, h) <= 0:
                raise ValueError(f"Invalid bounding box: {image_id}")
            if x + w > width + 1e-6 or y + h > height + 1e-6:
                raise ValueError(f"Out-of-bounds annotation: {image_id}")
            record = images.setdefault(image_id, dict(source=source, width=width, height=height, boxes=[]))
            if (record["source"], record["width"], record["height"]) != (source, width, height):
                raise ValueError(f"Inconsistent metadata: {image_id}")
            record["boxes"].append([x, y, min(x + w, width), min(y + h, height)])

    paths = {p.stem: p for p in (root / "train").glob("*.jpg")}
    missing = images.keys() - paths.keys()
    if missing:
        raise ValueError(f"Missing annotated images: {sorted(missing)[:5]}")
    unlabelled = sorted(paths.keys() - images.keys())
    if unlabelled:
        print(f"Excluding {len(unlabelled)} images with no source/annotation in train.csv.")
    checksums = {}
    for image_id, record in sorted(images.items()):
        path = paths[image_id]
        with Image.open(path) as im:
            if im.size != (record["width"], record["height"]):
                raise ValueError(f"Unexpected image dimensions: {image_id}")
        checksums[image_id] = sha(path)
    return images, checksums


def split_dataset(images, checksums, seed=42):
    parent = {r["source"]: r["source"] for r in images.values()}

    def group(s):
        while s != parent[s]:
            s = parent[s]
        return s

    seen = {}
    for image_id in sorted(images):
        source = images[image_id]["source"]
        previous = seen.setdefault(checksums[image_id], source)
        parent[group(source)] = group(previous)

    sizes = Counter(group(row["source"]) for row in images.values())
    groups = sorted(sizes)
    if len(groups) < 3:
        raise ValueError("Need at least three independent source groups.")
    rng = random.Random(seed)
    rng.shuffle(groups)
    target = (0.6, 0.2, 0.2)
    if len(groups) <= 10:
        assignments = itertools.product(range(3), repeat=len(groups))
    else:
        assignments = (rng.choices(range(3), target, k=len(groups)) for _ in range(10_000))
    winner, error = None, math.inf
    for assignment in assignments:
        if set(assignment) != {0, 1, 2}:
            continue
        counts = [sum(sizes[g] for g, part in zip(groups, assignment) if part == i) for i in range(3)]
        if counts[0] < max(counts[1:]):
            continue
        loss = sum((n / len(images) - goal) ** 2 for n, goal in zip(counts, target))
        if loss < error:
            winner, error = assignment, loss
    if winner is None:
        raise ValueError("No valid source-group split.")
    labels = {
        g: ("train", "validation", "holdout")[part] for g, part in zip(groups, winner)}
    partitions = {part: [] for part in ("train", "validation", "holdout")}
    for image_id, row in sorted(images.items()):
        partitions[labels[group(row["source"])]].append(image_id)
    return partitions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared/split.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("The split is already frozen; use a different output path to create a new evaluation cohort.")
    images, hashes = load_dataset(args.data)
    split = split_dataset(images, hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": 42, "csv_sha256": sha(args.data / "train.csv"), "images": hashes, "partitions": split}
    with args.output.open("x") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
    for part, ids in split.items():
        print(part, len(ids), dict(Counter(images[i]["source"] for i in ids)))


if __name__ == "__main__":
    main()
