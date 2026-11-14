"""Render fixed validation samples on the Mac for local MLflow artifacts."""

import csv
import json
import re
from collections import defaultdict

from PIL import Image, ImageDraw

from data import sha
from metric import match, validate_predictions
from preview_samples import select_preview_ids


def render_previews(project, artifacts, expected_epochs=0):
    manifest = json.loads((project / "data/prepared/split.json").read_text())
    ids = select_preview_ids(manifest["partitions"]["validation"])
    data = project / "data/raw"
    annotation_path = data / "train.csv"
    if sha(annotation_path) != manifest["csv_sha256"]:
        raise ValueError("Preview annotations do not match the frozen dataset.")
    truth = defaultdict(list)
    with annotation_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["image_id"] in ids:
                x, y, w, h = json.loads(row["bbox"])
                truth[row["image_id"]].append([x, y, x + w, y + h])

    batches = []
    for path in sorted((artifacts / "epoch-previews").glob("epoch-*.json")):
        if re.fullmatch(r"epoch-\d{3}", path.stem):
            predictions = json.loads(path.read_text())
            if predictions.keys() != set(ids):
                raise ValueError(f"Preview IDs do not match the fixed samples: {path.name}")
            batches.append((path.stem, predictions))
    if expected_epochs and [epoch for epoch, _ in batches] != [
        f"epoch-{epoch:03d}" for epoch in range(1, expected_epochs + 1)
    ]:
        raise ValueError(f"Expected {expected_epochs} validation preview files in {artifacts.name}.")
    if not batches:
        predictions = json.loads((artifacts / "predictions.json").read_text())["images"]
        batches.append(("final", {image_id: predictions[image_id] for image_id in ids}))

    for image_id in ids:
        path = data / "train" / f"{image_id}.jpg"
        if image_id not in truth or sha(path) != manifest["images"][image_id]:
            raise ValueError("Preview image or annotation does not match the frozen dataset.")
        with Image.open(path) as source:
            original = source.convert("RGB")
        for epoch, predictions in batches:
            guesses = predictions[image_id]
            validate_predictions(guesses, *original.size)
            hits, extras, duplicates, misses = match(truth[image_id], guesses, 0.5)
            image = original.copy()
            draw = ImageDraw.Draw(image)
            for i, box in enumerate(truth[image_id]):
                draw.rectangle(box, outline="red" if i in misses else "white", width=3)
            for indices, color in ((hits, "lime"), (extras, "orange"), (duplicates, "violet")):
                for index in indices:
                    draw.rectangle(guesses[index][1:], outline=color, width=2)
            image.thumbnail((800, 800))
            frame = Image.new("RGB", (image.width, image.height + 45), "#181818")
            frame.paste(image, (0, 45))
            legend = ImageDraw.Draw(frame)
            recall = len(hits) / len(truth[image_id])
            legend.text((8, 5), f"{epoch} | IoU 0.50 | recall {len(hits)}/{len(truth[image_id])} ({recall:.0%})",
                        fill="white")
            legend.text((8, 22), "Green: match  White: truth  Red: miss  Orange: false positive  Violet: duplicate",
                        fill="white")
            output = artifacts / "validation-previews" / epoch
            output.mkdir(parents=True, exist_ok=True)
            frame.save(output / f"{image_id}.jpg", quality=88)
