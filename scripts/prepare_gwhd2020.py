"""Normalize the published GWHD 2020 v4 training annotations for this project."""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ARCHIVE_MD5 = "be1e6969728afcf6768b5b50a42221eb"
SOURCES = ("arvalis_1", "arvalis_2", "arvalis_3", "ethz_1", "inrae_1", "rres_1", "usask_1")


def convert(archive, output):
    counts = Counter()
    with (output / "train.csv").open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image_id", "width", "height", "source", "bbox"))
        for source in SOURCES:
            labels = json.loads(archive.read(f"{source}.json"))
            wheat = {c["id"] for c in labels["categories"] if c["name"] == "wheat_head"}
            if not wheat:
                raise ValueError(f"No wheat category for {source}")
            boxes = defaultdict(list)
            for row in labels["annotations"]:
                if row["category_id"] in wheat and not row.get("iscrowd"):
                    boxes[row["image_id"]].append(row["bbox"])
            for row in sorted(labels["images"], key=lambda im: im["file_name"]):
                filename = row["file_name"]
                if not re.fullmatch(r"[\w.-]+\.png", filename):
                    raise ValueError(f"Invalid filename for {source}")
                width, height = row["width"], row["height"]
                coords = []
                for x, y, w, h in boxes[row["id"]]:
                    if not all(map(math.isfinite, (x, y, w, h))) or min(w, h) <= 0:
                        raise ValueError(f"Invalid source annotation for {filename}")
                    left, top = max(x, 0), max(y, 0)
                    right, bottom = min(x + w, width), min(y + h, height)
                    if left >= right or top >= bottom:
                        raise ValueError(f"Box outside image for {filename}")
                    counts["clipped_boxes"] += int((left, top, right, bottom) != (x, y, x + w, y + h))
                    coords.append((left, top, right - left, bottom - top))
                if not coords:
                    counts["unannotated_images"] += 1
                    continue
                image_id = f"{source}--{Path(filename).stem}"
                with Image.open(io.BytesIO(archive.read(f"{source}/{filename}"))) as image:
                    if image.size != (width, height):
                        raise ValueError(f"Image size mismatch for {filename}")
                    with (output / "train" / f"{image_id}.jpg").open("xb") as destination:
                        image.convert("RGB").save(destination, format="JPEG", quality=95)
                for box in coords:
                    writer.writerow((image_id, width, height, source, json.dumps(box)))
                counts[source] += 1
                counts["boxes"] += len(coords)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.archive.open("rb") as file:
        if hashlib.file_digest(file, "md5").hexdigest() != ARCHIVE_MD5:
            raise ValueError("Archive checksum does not match Zenodo record 4298502")
    args.output.mkdir(parents=True)
    (args.output / "train").mkdir()
    with zipfile.ZipFile(args.archive) as archive:
        counts = convert(archive, args.output)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
