"""Package the labelled Mallala SPIKE training photographs after a duplicate audit."""

import argparse
import io
import json
from collections import Counter
from itertools import combinations
from pathlib import Path, PurePosixPath
from statistics import median
from zipfile import ZipFile, ZipInfo

from PIL import Image

from prepare_gwhd2021_mons import PROJECT, sha, signature

SOURCE = "https://sourceforge.net/projects/spike-dataset/files/"
ARCHIVE_SHA256 = "baf2fe4874ee13f2935c4053f41c0f615962c8114c6f048ad9f41f06827bfbbf"
PREFIX = "spike-mallala-"
ANNOTATION = "spike-mallala-boxes.json"


def boxes_for(source, original, width, height, stats):
    coordinates = source.read(f"{original}.bboxes.tsv").decode().splitlines()
    classes = source.read(f"{original}.bboxes.labels.tsv").decode().splitlines()
    output = []
    for text, name in zip(coordinates, classes, strict=True):
        if name.strip() != "S":
            raise ValueError(f"Unexpected SPIKE class in {original}.")
        left, top, right, bottom = map(int, text.split())
        box = max(0, left), max(0, top), min(width, right), min(height, bottom)
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Empty SPIKE box in {original}.")
        stats["clipped_boxes"] += box != (left, top, right, bottom)
        output.append(box)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.archive) != ARCHIVE_SHA256:
        raise ValueError("SPIKE archive differs from the recorded publisher release.")
    split = PROJECT / "data/prepared/split.json"
    raw = PROJECT / "data/raw"
    meta = json.loads(split.read_text())
    if sha(raw / "train.csv") != meta["csv_sha256"] or any(
        sha(raw / "train" / f"{key}.jpg") != digest for key, digest in meta["images"].items()
    ):
        raise ValueError("Kaggle data differs from the frozen split.")
    kaggle = [signature(path) for part in ("train", "test") for path in (raw / part).glob("*.jpg")]
    records = {}
    stats = Counter()
    with ZipFile(args.archive) as source:
        paths = [PurePosixPath(name) for name in source.namelist() if name.endswith(".jpg")]
        training = sorted(path for path in paths if path.parent.name == "positive" and path.stem.startswith("Spike_"))
        reserved = [path for path in paths if path.parent.name != "positive"]
        if len(training) != 302 or len(reserved) != 40:
            raise ValueError("Expected the original 302 Spike_* positives and 40 evaluation images.")
        other = [signature(io.BytesIO(source.read(str(path)))) for path in reserved]
        for path in training:
            payload = source.read(str(path))
            code = signature(io.BytesIO(payload))
            if any((code ^ reference).bit_count() <= 4 for reference in kaggle):
                stats["near_kaggle"] += 1
                continue
            if any((code ^ reference).bit_count() <= 4 for reference in other):
                stats["near_reserved"] += 1
                continue
            with Image.open(io.BytesIO(payload)) as image:
                width, height = image.size
                if (width, height) != (2001, 1501):
                    raise ValueError(f"Unexpected SPIKE photo dimensions: {path}")
                boxes = boxes_for(source, path.with_suffix(""), width, height, stats)
                image = image.convert("RGB")
                image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                raster = io.BytesIO()
                image.save(raster, format="JPEG", quality=91)
            scale = (image.width / width, image.height / height) * 2
            areas = [(b[2] - b[0]) * (b[3] - b[1]) / (width * height) for b in boxes]
            name = PREFIX + path.stem.lower().removeprefix("spike_")
            records[name] = ([[value * factor for value, factor in zip(box, scale, strict=True)] for box in boxes],
                             areas, raster.getvalue(), code)
    near = {key for a, b in combinations(records, 2) if (records[a][-1] ^ records[b][-1]).bit_count() <= 4
            for key in (a, b)}
    stats["near_training"] = len(near)
    args.output.mkdir(parents=True, exist_ok=False)
    archive = args.output / "spike-mallala-train.bin"
    annotations, areas = {}, []
    with ZipFile(archive, "w") as bundle:
        for key in sorted(records.keys() - near):
            annotations[key], box_areas, payload, _ = records[key]
            areas.extend(box_areas)
            bundle.writestr(ZipInfo(f"train/{key}.jpg"), payload)
        bundle.writestr(ZipInfo(ANNOTATION), json.dumps(annotations, sort_keys=True, allow_nan=False))
    manifest = {"source": SOURCE, "publication": "https://doi.org/10.1186/s13007-018-0366-8",
                "release_archive_sha256": ARCHIVE_SHA256, "split_sha256": sha(split),
                "archive_sha256": sha(archive), "archive_file": archive.name, "annotation_file": ANNOTATION,
                "image_prefix": PREFIX, "images": len(annotations), "source_images": len(training),
                "heads": len(areas), "small_heads": sum(area < .002 for area in areas),
                "median_box_area_fraction": median(areas), "audit": dict(stats)}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
