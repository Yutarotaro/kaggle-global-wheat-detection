"""Compare confidence cutoffs on saved predictions from the fixed validation split."""

import argparse
import hashlib
import json
import math
import sys
from bisect import bisect_right
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src/wheat"))

from data import load_dataset, sha
from metric import THRESHOLDS, match, validate_predictions


def score_cutoffs(predictions, labels, cutoffs):
    totals = dict.fromkeys(cutoffs, 0.0)
    for image_id, guesses in predictions.items():
        record = labels[image_id]
        actual = record["boxes"]
        validate_predictions(guesses, record["width"], record["height"])
        guesses = sorted(guesses, key=lambda box: -box[0])
        negative_confidence = [-box[0] for box in guesses]
        limits = {cutoff: bisect_right(negative_confidence, -cutoff) for cutoff in cutoffs}
        for threshold in THRESHOLDS:
            # Matching is greedy in descending confidence order. Dropping a suffix
            # leaves the assignments for the retained predictions unchanged.
            hits, _, _, _ = match(actual, guesses, threshold)
            hits.sort()
            for cutoff, limit in limits.items():
                count = bisect_right(hits, limit - 1)
                totals[cutoff] += count / (len(actual) + limit - count)
    return {cutoff: total / (len(predictions) * len(THRESHOLDS)) for cutoff, total in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_artifacts", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"),
                        help="Training config relative to the trial's frozen source.")
    parser.add_argument("--cutoffs", type=float, nargs="+", default=[0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70])
    args = parser.parse_args()
    if any(not math.isfinite(cutoff) or not 0 <= cutoff <= 1 for cutoff in args.cutoffs):
        parser.error("Confidence cutoffs must be in [0, 1].")
    artifacts = args.trial_artifacts.resolve()
    config = artifacts.parent.parent / "sources" / artifacts.name / args.config
    floor = json.loads(config.read_text())["confidence"]
    if min(args.cutoffs) < floor:
        parser.error(f"Predictions were already filtered at {floor}; all cutoffs must be at least {floor}.")
    manifest = json.loads((PROJECT / "data/prepared/split.json").read_text())
    dataset = PROJECT / "data/raw"
    if sha(dataset / "train.csv") != manifest["csv_sha256"]:
        raise ValueError("Annotations do not match the frozen split.")
    labels, hashes = load_dataset(dataset)
    prediction_path = artifacts / "predictions.json"
    payload = prediction_path.read_bytes()
    predictions = json.loads(payload)["images"]
    ids = set(manifest["partitions"]["validation"])
    if not ids or ids != predictions.keys() or any(hashes[key] != manifest["images"][key] for key in ids):
        raise ValueError("Predictions and images must match the fixed validation split.")
    scores = score_cutoffs(predictions, labels, sorted(set(args.cutoffs) | {floor}))
    reference = json.loads((artifacts / "evaluation.json").read_text())
    if not math.isclose(scores[floor], reference["score"], rel_tol=0, abs_tol=1e-10):
        raise ValueError("Confidence sweep disagrees with the independent evaluation score.")
    print(f"Confidence sweep for {artifacts.name} on the fixed validation split:", file=sys.stderr)
    print(json.dumps({"trial_id": artifacts.name, "partition": "validation",
                      "predictions_sha256": hashlib.sha256(payload).hexdigest(),
                      "score_by_cutoff": scores}, indent=2))


if __name__ == "__main__":
    main()
