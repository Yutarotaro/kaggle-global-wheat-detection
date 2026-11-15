"""Audit and package CC BY 4.0 South Dakota wheat heads for research training."""

import argparse
import io
import json
import math
import re
from collections import Counter
from hashlib import file_digest
from itertools import combinations
from pathlib import Path, PurePosixPath
from statistics import median
from zipfile import ZipFile, ZipInfo

from PIL import Image

from prepare_gwhd2021_mons import PROJECT, sha, signature

SOURCE = "https://zenodo.org/records/20215489"
ARCHIVE_MD5 = "e91153c30b4eb641a1b78923908b0b6b"
PREFIX = "sd-rwhd-"
ANNOTATION = "sd-rwhd-boxes.json"


def parse_obb(text):
    boxes = []
    clipped_count = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        category, *coords = line.split()
        points = [float(value) for value in coords]
        if category != "0" or len(points) != 8 or not all(map(math.isfinite, points)):
            raise ValueError(f"Expected one wheat class and four normalized vertices: {line}")
        xs, ys = points[::2], points[1::2]
        bounds = min(xs), min(ys), max(xs), max(ys)
        box = tuple(min(max(value, 0.0), 1.0) for value in bounds)
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Empty annotation after converting from an oriented box.")
        boxes.append(box)
        clipped_count += box != bounds
    return boxes, clipped_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.archive.open("rb") as handle:
        if file_digest(handle, "md5").hexdigest() != ARCHIVE_MD5:
            raise ValueError("SD-RWHD archive does not match Zenodo.")
    split = PROJECT / "data/prepared/split.json"
    raw = PROJECT / "data/raw"
    split_meta = json.loads(split.read_text())
    if sha(raw / "train.csv") != split_meta["csv_sha256"] or any(
        sha(raw / "train" / f"{key}.jpg") != value for key, value in split_meta["images"].items()
    ):
        raise ValueError("Kaggle comparison data differs from the fixed manifest.")
    reference = [signature(path) for part in ("train", "test") for path in (raw / part).glob("*.jpg")]
    records = {}
    by_resolution = Counter()
    exclusions = Counter()
    with ZipFile(args.archive) as source:
        members = [PurePosixPath(name) for name in source.namelist()]
        photos = {path.stem: path for path in members if path.parent == PurePosixPath("images")
                  and path.suffix.lower() == ".jpg"}
        labels = {path.stem: path for path in members if path.parent == PurePosixPath("labels")
                  and path.suffix.lower() == ".txt"}
        if photos.keys() != labels.keys() or len(photos) != 159:
            raise ValueError("Expected 159 one-to-one SD-RWHD image and label pairs.")
        archive_counts = []
        clipped_annotations = 0
        for name, path in sorted(photos.items()):
            match = re.fullmatch(r"(\d+)_.+_(2023-\d{2}-\d{2})-\d{2}-\d{2}-\d{2}", name)
            if match is None:
                raise ValueError(f"Missing acquisition date or plot ID: {name}")
            plot, day = match.groups()
            boxes, clipped = parse_obb(source.read(str(labels[name])).decode())
            archive_counts.append(len(boxes))
            clipped_annotations += clipped
            payload = source.read(str(path))
            code = signature(io.BytesIO(payload))
            if any((code ^ known).bit_count() <= 4 for known in reference):
                exclusions["near_kaggle"] += 1
                continue
            with Image.open(io.BytesIO(payload)) as image:
                if image.getexif().get(274, 1) != 1:
                    raise ValueError(f"Review EXIF orientation against annotations: {name}")
                by_resolution[f"{image.width}x{image.height}"] += 1
                image = image.convert("RGB")
                image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                raster = io.BytesIO()
                image.save(raster, format="JPEG", quality=91)
            key = f"{PREFIX}{plot}-{day}-{name.rsplit('_', 1)[1][-8:]}"
            if key in records:
                raise ValueError("Duplicate packaged image ID.")
            records[key] = (day, boxes, image.size, raster.getvalue(), code)
    near = {key for a, b in combinations(records, 2) if (records[a][-1] ^ records[b][-1]).bit_count() <= 4
            for key in (a, b)}
    exclusions["near_sd_rwhd"] = len(near)
    args.output.mkdir(parents=True, exist_ok=False)
    archive = args.output / "sd-rwhd-train.bin"
    images, dates, areas, counts = {}, Counter(), [], []
    with ZipFile(archive, "w") as bundle:
        for key in sorted(records.keys() - near):
            day, boxes, (width, height), jpeg, _ = records[key]
            dates[day] += 1
            counts.append(len(boxes))
            areas.extend((r - l) * (b - t) for l, t, r, b in boxes)
            images[key] = [[l * width, t * height, r * width, b * height] for l, t, r, b in boxes]
            bundle.writestr(ZipInfo(f"train/{key}.jpg"), jpeg)
        bundle.writestr(ZipInfo(ANNOTATION), json.dumps(images, sort_keys=True, allow_nan=False))
    manifest = {"source": SOURCE, "license": "CC BY 4.0", "release_archive_md5": ARCHIVE_MD5,
                "split_sha256": sha(split), "archive_sha256": sha(archive), "archive_file": archive.name,
                "annotation_file": ANNOTATION, "image_prefix": PREFIX, "images": len(images),
                "source_images": len(photos), "source_boxes": sum(archive_counts), "heads": sum(counts),
                "clipped_source_boxes": clipped_annotations,
                "sessions": dict(dates), "original_resolutions": dict(by_resolution),
                "exclusions": dict(exclusions), "head_count_min_median_max": [min(counts), median(counts), max(counts)],
                "small_heads": sum(area < .002 for area in areas), "median_box_area_fraction": median(areas)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
