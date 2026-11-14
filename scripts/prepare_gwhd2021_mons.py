"""Prepare the Arvalis_10 training-only supplement from GWHD 2021 (JPEG)."""

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile, ZipInfo

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src/wheat"))
from data import sha

DOMAIN = "Arvalis_10"
RECORD = "https://zenodo.org/records/8062267"
EXPECTED_ROWS = 60
ARCHIVE_MD5 = "92b0232d78420d1ddc1894fba89933b0"


def signature(path):
    with Image.open(path) as image:
        cells = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
        return sum((cells[y * 9 + x] > cells[y * 9 + x + 1]) << (y * 8 + x)
                   for y in range(8) for x in range(8))


def parse_boxes(value, width, height):
    if value == "no_box":
        return []
    boxes = []
    for box in value.split(";"):
        x1, y1, x2, y2 = map(float, box.split())
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Out-of-bounds or empty box: {box}")
        boxes.append([x1, y1, x2, y2])
    return boxes


def selected_from_zip(path):
    from hashlib import file_digest

    with path.open("rb") as handle:
        if file_digest(handle, "md5").hexdigest() != ARCHIVE_MD5:
            raise ValueError("GWHD 2021 ZIP checksum does not match Zenodo.")
    with ZipFile(path) as archive:
        table = archive.read("gwhd_2021/competition_train.csv")
        with io.StringIO(table.decode("utf-8-sig")) as annotations:
            for row in csv.DictReader(annotations):
                if row["domain"] == DOMAIN:
                    yield row, archive.read(f"gwhd_2021/images/{row['image_name']}")


def selected_from_directory(path):
    with (path / "competition_train.csv").open(newline="", encoding="utf-8-sig") as annotations:
        for row in csv.DictReader(annotations):
            name = row["image_name"]
            if row["domain"] != DOMAIN or Path(name).name != name:
                raise ValueError("Expected only verified Arvalis_10 images.")
            yield row, (path / "images" / name).read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original Zenodo ZIP or its CRC-verified Arvalis_10 subset.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixed = PROJECT / "data/prepared/split.json"
    raw = PROJECT / "data/raw"
    manifest = json.loads(fixed.read_text())
    if sha(raw / "train.csv") != manifest["csv_sha256"]:
        raise ValueError("Competition annotations do not match the frozen split.")
    train = {path.stem: path for path in (raw / "train").glob("*.jpg")}
    if any(key not in train or sha(train[key]) != digest for key, digest in manifest["images"].items()):
        raise ValueError("Competition reference images do not match the frozen split.")
    with (raw / "sample_submission.csv").open(newline="") as handle:
        test = [raw / "test" / f"{row['image_id']}.jpg" for row in csv.DictReader(handle)]
    if any(not path.is_file() for path in test):
        raise ValueError("A competition sample test image is unavailable.")
    originals = {path.name: signature(path) for path in [*train.values(), *test]}
    selected = list(selected_from_directory(args.source) if args.source.is_dir() else selected_from_zip(args.source))
    names = {row["image_name"] for row, _ in selected}
    if len(selected) != EXPECTED_ROWS or len(names) != EXPECTED_ROWS:
        raise ValueError("Unexpected Arvalis_10 cardinality.")
    records, images, excluded = {}, {}, 0
    for row, image_bytes in sorted(selected, key=lambda item: item[0]["image_name"]):
        name = row["image_name"]
        if Path(name).name != name or Path(name).suffix != ".jpg":
            raise ValueError(f"Unexpected image filename: {name}")
        fingerprint = signature(io.BytesIO(image_bytes))
        if name in originals or any((fingerprint ^ other).bit_count() <= 4 for other in originals.values()):
            excluded += 1
            continue
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format != "JPEG":
                raise ValueError(f"Expected a JPEG: {name}")
            boxes = parse_boxes(row["BoxesString"], *image.size)
            image.verify()
        key = f"gwhd2021-mons-{Path(name).stem}"
        records[key] = boxes
        images[key] = image_bytes
    if not records:
        raise ValueError("The supplement has no images after duplicate checks.")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    artifact = out / "gwhd2021-mons.zip"
    with ZipFile(artifact, "w") as archive:
        archive.writestr(ZipInfo("gwhd2021-mons-boxes.json"), json.dumps(records, sort_keys=True, allow_nan=False))
        for key, payload in images.items():
            archive.writestr(ZipInfo(f"train/{key}.jpg"), payload)
    info = {"source": RECORD, "license": "CC BY 4.0", "domain": DOMAIN, "split_sha256": sha(fixed),
            "archive_sha256": sha(artifact), "images": len(records), "excluded_near_duplicates": excluded,
            "heads": sum(map(len, records.values())), "under_25": sum(len(boxes) < 25 for boxes in records.values()),
            "empty": sum(not boxes for boxes in records.values())}
    (out / "manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
