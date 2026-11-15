"""Profile annotated wheat images in the frozen train and validation splits."""

import argparse
import json
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, quantiles

from PIL import Image, ImageStat

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src/wheat"))

from data import load_dataset, sha
from metric import area

COUNT_EDGES = (10, 25, 50, 75, 100)
AREA_EDGES = (0.002, 0.01)


def summarize(records):
    counts = [len(record["boxes"]) for record in records]
    brightness = [record["brightness"] for record in records]
    deciles = quantiles(brightness, n=10, method="inclusive") if len(brightness) > 1 else brightness
    count_bins = [0] * (len(COUNT_EDGES) + 1)
    area_bins = [0] * (len(AREA_EDGES) + 1)
    for record, count in zip(records, counts):
        count_bins[bisect_right(COUNT_EDGES, count)] += 1
        pixels = record["width"] * record["height"]
        for box in record["boxes"]:
            area_bins[bisect_right(AREA_EDGES, area(box) / pixels)] += 1
    return {
        "images": len(records),
        "heads": sum(counts),
        "mean": mean(counts),
        "median": median(counts),
        "p90": quantiles(counts, n=10, method="inclusive")[-1] if len(counts) > 1 else counts[0],
        "max": max(counts),
        "count_bins": count_bins,
        "area_bins": area_bins,
        "brightness": {
            "mean": mean(brightness), "median": median(brightness), "p10": deciles[0], "p90": deciles[-1],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = PROJECT / "data/raw"
    manifest_path = PROJECT / "data/prepared/split.json"
    manifest = json.loads(manifest_path.read_text())
    if sha(root / "train.csv") != manifest["csv_sha256"]:
        raise ValueError("Annotations do not match the frozen split.")
    records, hashes = load_dataset(root)
    result = {
        "split_sha256": sha(manifest_path),
        "count_edges": COUNT_EDGES,
        "area_edges": AREA_EDGES,
        "partitions": {},
    }
    seen_ids, seen_sources, seen_hashes = set(), set(), set()
    for partition in ("train", "validation"):
        ids = manifest["partitions"][partition]
        if not ids or len(ids) != len(set(ids)) or seen_ids.intersection(ids):
            raise ValueError(f"Invalid image IDs in {partition}.")
        if any(hashes[image_id] != manifest["images"][image_id] for image_id in ids):
            raise ValueError(f"Images in {partition} do not match the frozen split.")
        groups = defaultdict(list)
        for image_id in ids:
            record = records[image_id]
            with Image.open(root / "train" / f"{image_id}.jpg") as image:
                record["brightness"] = ImageStat.Stat(image.convert("L")).mean[0]
            groups[record["source"]].append(record)
        partition_hashes = {hashes[image_id] for image_id in ids}
        if seen_sources.intersection(groups) or seen_hashes.intersection(partition_hashes):
            raise ValueError("Train and validation share a source or identical image.")
        seen_ids.update(ids)
        seen_sources.update(groups)
        seen_hashes.update(partition_hashes)
        result["partitions"][partition] = {
            "all": summarize([records[image_id] for image_id in ids]),
            "sources": {name: summarize(rows) for name, rows in sorted(groups.items())},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
