"""Package the audited 385 GWHD 2021 crops for the fixed wheat validation split."""

import argparse
import csv
import io
import json
from collections import Counter
from hashlib import file_digest
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile, ZipInfo

from PIL import Image

from prepare_gwhd2021_mons import ARCHIVE_MD5, PROJECT, RECORD, parse_boxes, sha, signature

DOMAINS = {"Arvalis_9", "Arvalis_10", "Arvalis_11", "Arvalis_12", "NMBU_1", "NMBU_2", "ULiège-GxABT_1"}
ANNOTATION = "gwhd2021-field-boxes.json"
PREFIX = "gwhd2021-field-"
ARCHIVE = "gwhd2021-field385.bin"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.source.open("rb") as handle:
        if file_digest(handle, "md5").hexdigest() != ARCHIVE_MD5:
            raise ValueError("GWHD 2021 JPEG ZIP does not match Zenodo.")
    split = PROJECT / "data/prepared/split.json"
    fixed = json.loads(split.read_text())
    raw = PROJECT / "data/raw"
    if sha(raw / "train.csv") != fixed["csv_sha256"]:
        raise ValueError("Kaggle annotations do not match the frozen split.")
    reference = [*raw.joinpath("train").glob("*.jpg"), *raw.joinpath("test").glob("*.jpg")]
    by_id = {path.stem: path for path in raw.joinpath("train").glob("*.jpg")}
    if any(sha(by_id[key]) != checksum for key, checksum in fixed["images"].items()):
        raise ValueError("Kaggle images do not match the frozen split.")
    original_names = {path.name for path in reference}
    original_codes = [signature(path) for path in reference]

    with ZipFile(args.source) as source:
        tables = {part: list(csv.DictReader(io.StringIO(
            source.read(f"gwhd_2021/competition_{part}.csv").decode("utf-8-sig"))))
                  for part in ("train", "val", "test")}
        other = {row["image_name"] for part in ("val", "test") for row in tables[part]}
        count = Counter(row["image_name"] for row in tables["train"])
        entries = {}
        for row in tables["train"]:
            name = row["image_name"]
            if row["domain"] not in DOMAINS or count[name] != 1 or name in other:
                continue
            if Path(name).name != name or Path(name).suffix != ".jpg":
                raise ValueError("Unexpected GWHD 2021 image path.")
            payload = source.read(f"gwhd_2021/images/{name}")
            code = signature(io.BytesIO(payload))
            if name in original_names or any((code ^ other_code).bit_count() <= 4 for other_code in original_codes):
                continue
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != "JPEG":
                    raise ValueError("Expected JPEG data.")
                boxes = parse_boxes(row["BoxesString"], *image.size)
                image.verify()
            entries[name] = row["domain"], boxes, payload, code

    near = {name for a, b in combinations(entries, 2)
            if (entries[a][3] ^ entries[b][3]).bit_count() <= 4 for name in (a, b)}
    kept = sorted(entries.keys() - near)
    if len(kept) != 385 or len(near) != 2:
        raise ValueError("Unexpected crop count after duplicate screens.")
    args.output.mkdir(parents=True, exist_ok=False)
    artifact = args.output / ARCHIVE
    boxes = {}
    by_session = Counter()
    with ZipFile(artifact, "w") as archive:
        for name in kept:
            domain, labels, payload, _ = entries[name]
            key = PREFIX + Path(name).stem
            boxes[key] = labels
            by_session[domain] += 1
            archive.writestr(ZipInfo(f"train/{key}.jpg"), payload)
        archive.writestr(ZipInfo(ANNOTATION), json.dumps(boxes, sort_keys=True, allow_nan=False))
    spec = {"source": RECORD, "license": "CC BY 4.0", "split_sha256": sha(split),
            "archive_sha256": sha(artifact), "archive_file": ARCHIVE, "annotation_file": ANNOTATION,
            "image_prefix": PREFIX, "images": len(boxes), "sessions": dict(by_session),
            "heads": sum(map(len, boxes.values())), "under_25": sum(len(labels) < 25 for labels in boxes.values())}
    (args.output / "manifest.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(spec, ensure_ascii=False))


if __name__ == "__main__":
    main()
