"""Replay frozen submission recipes on the published SPIKE and SD-RWHD fields."""

import argparse
import csv
import subprocess
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from statistics import mean
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import torch
from PIL import Image

from compare_checkpoint_tta import PROJECT, SPLIT, read, sha, write
from metric import area, image_score, match, validate_predictions
from prepare_spike_wheat import ARCHIVE_SHA256, boxes_for
from submission import predict
from supplement import unpack_supplement


def stage_fields(root, stack):
    test = root / "test"
    test.mkdir()
    labels = {}
    spike = PROJECT / "data/spike/SPIKE_dataset.zip"
    if sha(spike) != ARCHIVE_SHA256:
        raise ValueError("SPIKE release does not match the audited download.")
    with ZipFile(spike) as bundle:
        photos = sorted(PurePosixPath(name) for name in bundle.namelist()
                        if name.endswith(".jpg") and "testImages_SPIKE" in PurePosixPath(name).parts)
        if len(photos) != 30:
            raise ValueError("Expected the 30 publisher-reserved SPIKE test images.")
        for photo in photos:
            key = f"spike-{photo.stem.lower()}"
            path = test / f"{key}.jpg"
            if key in labels:
                raise ValueError(f"Duplicate SPIKE test ID: {key}")
            path.write_bytes(bundle.read(str(photo)))
            with Image.open(path) as image:
                width, height = image.size
            boxes = boxes_for(bundle, photo.with_suffix(""), width, height, Counter())
            labels[key] = (f"spike/{photo.parent.name.split()[0]}", width, height, boxes)

    sd = PROJECT / "data/sd-rwhd/prepared"
    manifest = read(sd / "manifest.json")
    unpacked, records = unpack_supplement(sd / "manifest.json", [sd / "sd-rwhd-train.bin"], SPLIT, stack)
    for key, boxes in records.items():
        jpeg = unpacked / "train" / f"{key}.jpg"
        (test / jpeg.name).symlink_to(jpeg)
        with Image.open(jpeg) as image:
            width, height = image.size
        day = key.removeprefix("sd-rwhd-").split("-", 1)[1][:10]
        labels[key] = (f"sd-rwhd/{day}", width, height, boxes)
    counts = Counter(group.split("/", 1)[1] for group, *_ in labels.values() if group.startswith("sd-rwhd/"))
    if counts != manifest["sessions"]:
        raise ValueError("SD-RWHD acquisition dates differ from the audited manifest.")
    with (root / "sample_submission.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_id", "PredictionString"))
        writer.writerows((key, "") for key in sorted(labels))
    return labels, {"spike_release_sha256": ARCHIVE_SHA256, "sd_rwhd_archive_sha256": manifest["archive_sha256"]}


def summarize(csv_path, labels):
    counts = defaultdict(Counter)
    scores = defaultdict(list)
    with csv_path.open(newline="") as handle:
        rows = {row["image_id"]: row["PredictionString"] for row in csv.DictReader(handle)}
    if rows.keys() != labels.keys():
        raise ValueError("Prediction IDs do not match the external benchmark.")
    for key, (group, width, height, actual) in labels.items():
        tokens = [float(value) for value in rows[key].split()]
        predicted = []
        for start in range(0, len(tokens), 5):
            score, x, y, w, h = tokens[start:start + 5]
            predicted.append([score, x, y, x + w, y + h])
        validate_predictions(predicted, width, height)
        small = {i for i, box in enumerate(actual) if area(box) / (width * height) < 0.002}
        hits, false_positives, duplicates, misses = match(actual, predicted, 0.5)
        fields = {
            "images": 1, "labels": len(actual), "predictions": len(predicted), "hits": len(hits),
            "false_positives": len(false_positives), "duplicates": len(duplicates), "misses": len(misses),
            "small_labels": len(small), "small_hits": len(small.difference(misses)),
        }
        score = image_score(actual, predicted)
        for level in (group.split("/", 1)[0], group):
            counts[level].update(fields)
            scores[level].append(score)
    return {group: dict(values, score=mean(scores[group])) for group, values in sorted(counts.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipes", nargs="+", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT, text=True)
    if status.strip():
        raise ValueError("Run the diagnostic from a clean committed tree.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    report = {"git_commit": commit, "device": args.device, "torch": torch.__version__, "recipes": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    with ExitStack() as stack:
        root = Path(stack.enter_context(TemporaryDirectory(prefix="wheat-external-fields-")))
        labels, report["data"] = stage_fields(root, stack)
        for recipe in args.recipes:
            spec = read(recipe)
            trial_ids = [model["trial_id"] for model in spec["models"]]
            weights = [PROJECT / ".loop-wheat" / trial.rsplit("-", 1)[0] / "artifacts" / trial / "weights.pt"
                       for trial in trial_ids]
            csv_path = root / f"{recipe.stem}.csv"
            print(f"Running {spec['run_label']} on {len(labels)} independent images.", flush=True)
            predict(root, weights, spec, csv_path, torch.device(args.device))
            report["recipes"][spec["run_label"]] = {"sha256": sha(recipe), "train_trials": trial_ids,
                                                     "by_field": summarize(csv_path, labels)}
            write(args.output, report)


if __name__ == "__main__":
    main()
