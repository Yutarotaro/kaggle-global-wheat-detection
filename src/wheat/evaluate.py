"""Trusted evaluation of validation predictions, with visual error examples."""

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw

from data import sha
from metric import area, image_score, match, validate_predictions


def evaluate(data, manifest, out, partition):
    if hashlib.sha256((data / "train.csv").read_bytes()).hexdigest() != manifest["csv_sha256"]:
        raise ValueError("Annotation file no longer matches the frozen split.")
    ids = set(manifest["partitions"][partition])
    payload = json.loads((out / "predictions.json").read_text())
    predictions = payload["images"]
    if set(predictions) != ids or not ids:
        raise ValueError("Prediction IDs do not exactly match the selected partition.")
    truth = defaultdict(list)
    meta = {}
    with (data / "train.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = row["image_id"]
            if image_id in ids:
                x, y, w, h = json.loads(row["bbox"])
                truth[image_id].append([x, y, x + w, y + h])
                meta[image_id] = row
    if set(meta) != ids:
        raise ValueError("Missing validation labels.")

    scores, errors = {}, []
    sources = defaultdict(list)
    sizes = defaultdict(lambda: [0, 0])
    for image_id in sorted(ids):
        row = meta[image_id]
        if sha(data / "train" / f"{image_id}.jpg") != manifest["images"][image_id]:
            raise ValueError(f"Evaluation image does not match frozen data: {image_id}")
        boxes = predictions[image_id]
        validate_predictions(boxes, int(row["width"]), int(row["height"]))
        actual = truth[image_id]
        score = image_score(actual, boxes)
        scores[image_id] = score
        sources[row["source"]].append(score)
        _, extra, duplicates, misses = match(actual, boxes, 0.5)
        for i, box in enumerate(actual):
            ratio = area(box) / (int(row["width"]) * int(row["height"]))
            bucket = "small" if ratio < 0.002 else "medium" if ratio < 0.01 else "large"
            sizes[bucket][0] += int(i not in misses)
            sizes[bucket][1] += 1
        errors.append((image_id, extra, duplicates, misses))

    error_dir = out / "errors"
    error_dir.mkdir(exist_ok=True)
    for label, position in (("false-positive", 1), ("duplicate", 2), ("miss", 3)):
        ranked = sorted(errors, key=lambda e: (-len(e[position]), e[0]))
        for image_id, extra, duplicates, misses in [e for e in ranked if e[position]][:3]:
            with Image.open(data / "train" / f"{image_id}.jpg") as image:
                image = image.convert("RGB")
                draw = ImageDraw.Draw(image)
                for indices, boxes, color in ((misses, truth[image_id], "red"),
                                              (extra, [x[1:] for x in predictions[image_id]], "orange"),
                                              (duplicates, [x[1:] for x in predictions[image_id]], "violet")):
                    for i in indices:
                        draw.rectangle(boxes[i], outline=color, width=3)
                image.thumbnail((720, 720))
                image.save(error_dir / f"{label}-{image_id}.jpg")
    return {"score": mean(scores.values()), "details": {
        "partition": partition, "image_count": len(ids), "by_source": {k: mean(v) for k, v in sources.items()},
        "recall_by_box_size_iou_50": {k: {"recall": found / count, "boxes": count}
                                     for k, (found, count) in sizes.items()},
        "error_images": sorted(p.name for p in error_dir.glob("*.jpg")),
        "training_environment": payload["environment"],
    }}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("validation", "holdout"), default="validation")
    args = parser.parse_args()
    project = Path(os.environ["ML_LOOP_PROJECT"])
    out = Path(os.environ["ML_LOOP_ARTIFACTS"])
    split = json.loads((project / "data/prepared/split.json").read_text())
    result = evaluate(project / "data/raw", split, out, args.partition)
    (out / "evaluation.json").write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
