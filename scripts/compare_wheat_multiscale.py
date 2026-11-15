"""Test one extra 1024px view of the original D2 in the fixed three-model ensemble."""

import argparse
import copy
import math
import subprocess
from pathlib import Path

import torch

from compare_checkpoint_tta import PROJECT, RAW, SPLIT, load_dataset, read, score, sha, write
from fusion import fuse
from preprocess import read_image
from submission import load_detector, predict_views

REFERENCE = PROJECT / "configs/submission-mbv3-d2x2-hflip.json"
TRIAL = PROJECT / "outputs/checkpoint-tta/competition-only"
RESULTS = PROJECT / "outputs/checkpoint-tta/multiscale"
VIEW = {"image_size": 1024, "weights": [2, 1], "iou": 0.55}


@torch.inference_mode()
def collect(condition, partition, keys, device):
    name = condition["trial_id"]
    folder = PROJECT / ".loop-wheat" / name.rsplit("-", 1)[0] / "artifacts" / name
    checkpoint = folder / "weights.pt"
    if sha(checkpoint) != condition["sha256"]:
        raise ValueError("Unexpected D2 checkpoint.")
    path = RESULTS / f"{partition}-1024.json"
    fingerprint = {"trial": condition, "split": sha(SPLIT), "view": VIEW,
                   "torch": torch.__version__, "device": device.type}
    cache = read(path) if path.is_file() else {"fingerprint": fingerprint, "images": {}}
    rows = cache["images"]
    if cache["fingerprint"] != fingerprint or not rows.keys() <= set(keys):
        raise ValueError("Cached extra views do not correspond to this experiment.")
    missing = sorted(set(keys) - rows.keys())
    if missing:
        weights = torch.load(checkpoint, weights_only=True, map_location="cpu")
        model = load_detector(condition, weights, device, VIEW["image_size"])
        for i, key in enumerate(missing, 1):
            image, _ = read_image(RAW / "train" / f"{key}.jpg", [], 0)
            rows[key] = predict_views(model, image.to(device), [], VIEW["iou"])
            if i % 50 == 0 or i == len(missing):
                write(path, cache)
                print(f"1024px {partition} {len(rows)}/{len(keys)}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partition", choices=("validation", "holdout"))
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT, text=True).strip():
        raise ValueError("Start from a clean committed tree.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    partition = args.partition
    RESULTS.mkdir(parents=True, exist_ok=True)
    reference = read(REFERENCE)
    spec = copy.deepcopy(reference)
    condition = reference["models"][1]
    spec["models"][1]["scale_view"] = VIEW
    spec["variant"] = "d2-1024-view"
    if partition == "holdout":
        frozen = read(RESULTS / "validation.json")
        if not frozen["eligible"] or frozen["spec"] != spec:
            raise ValueError("The exact multiscale recipe must pass fixed validation before holdout.")
    split = read(SPLIT)
    keys = sorted(split["partitions"][partition])
    labels, hashes = load_dataset(RAW)
    if sha(RAW / "train.csv") != split["csv_sha256"] or any(hashes[key] != split["images"][key] for key in keys):
        raise ValueError("Images differ from the fixed data manifest.")
    old = ([views[1] for views in read(PROJECT / "outputs/field385/tta-predictions.json")]
           if partition == "validation" else
           [read(PROJECT / f"outputs/field385/holdout-tta-{index}.json") for index in range(2)])
    old.append(read(TRIAL / f"{partition}-tta.json")["images"])
    if any(rows.keys() != set(keys) for rows in old):
        raise ValueError("Incumbent prediction cache does not match the evaluation partition.")
    torch.set_num_threads(3)
    extra = collect(condition, partition, keys, torch.device(args.device))
    baseline = score(old, labels, keys, reference)
    if not math.isclose(baseline["score"], reference[f"{partition}_score"], abs_tol=1e-9, rel_tol=0):
        raise ValueError(f"The previous model score could not be reproduced: {baseline}")
    revised = {key: fuse([old[1][key], extra[key]], labels[key]["width"], labels[key]["height"],
                         VIEW["weights"], VIEW["iou"]) for key in keys}
    result = score([old[0], revised, old[2]], labels, keys, spec)
    eligible = (result["score"] > baseline["score"] and all(
        result["by_source"][group] >= original - 0.01 for group, original in baseline["by_source"].items()))
    report = {"partition": partition, "eligible": eligible, "spec": spec, "baseline": baseline,
              "reference_submission": 58377134,
              "result": result, "candidate_trial": condition["trial_id"],
              "split_sha256": sha(SPLIT), "git_commit": commit}
    write(RESULTS / f"{partition}.json", report)
    print(report["result"], f"eligible={eligible}", flush=True)


if __name__ == "__main__":
    main()
