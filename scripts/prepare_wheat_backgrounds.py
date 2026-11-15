"""Freeze crops without annotated heads from the competition training partition."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from prepare_gwhd2021_mons import PROJECT, sha


def background_crop(boxes, width, height):
    padding = np.array([math.ceil(.015 * width), math.ceil(.015 * height)])
    lower, upper = boxes[:, :2] - padding, boxes[:, 2:] + padding
    for scale in (.60, .50, .40, .32, .25):
        crop = np.rint(np.array([width, height]) * scale).astype(int)
        xs = np.rint(np.linspace(0, width - crop[0], 9)).astype(int)
        ys = np.rint(np.linspace(0, height - crop[1], 9)).astype(int)
        starts = np.array([(x, y) for x in xs for y in ys])
        ends = starts + crop
        collision = (starts[:, None, :] < upper).all(axis=2) & (ends[:, None, :] > lower).all(axis=2)
        free = np.flatnonzero(~collision.any(axis=1))
        if len(free):
            gaps = np.maximum(starts[free, None, :] - upper, lower - ends[free, None, :])
            distances = (gaps.clip(min=0) ** 2).sum(axis=2).min(axis=1)
            index = free[distances.argmin()]
            return [*starts[index].tolist(), *ends[index].tolist()], scale
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split_path = PROJECT / "data/prepared/split.json"
    csv_path = PROJECT / "data/raw/train.csv"
    split = json.loads(split_path.read_text())
    train_ids = set(split["partitions"]["train"])
    if sha(csv_path) != split["csv_sha256"] or not train_ids:
        raise ValueError("Expected the competition training annotations and frozen split.")
    truths, metadata = defaultdict(list), {}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["image_id"]
            if key in train_ids:
                x, y, width, height = json.loads(row["bbox"])
                truths[key].append([x, y, x + width, y + height])
                metadata[key] = int(row["width"]), int(row["height"]), row["source"]
    if truths.keys() != train_ids:
        raise ValueError("Missing training annotations.")
    crops, sources, scales = {}, Counter(), Counter()
    for key, truth in sorted(truths.items()):
        width, height, source = metadata[key]
        candidate = background_crop(np.asarray(truth), width, height)
        if candidate is not None:
            region, scale = candidate
            crops[key] = region
            sources[source] += 1
            scales[str(scale)] += 1
    if not crops:
        raise ValueError("No background crops found.")
    manifest = {"split_sha256": sha(split_path), "images": crops,
                "summary": {"num_background_candidates": len(crops), "training_images": len(train_ids),
                            "annotation_margin_fraction": .015, "grid_positions_per_axis": 9,
                            "sources": dict(sources), "scales": dict(scales)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(manifest["summary"]))


if __name__ == "__main__":
    main()
