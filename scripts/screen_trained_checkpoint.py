"""Screen a finished training run on validation before one independent comparison."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from compare_checkpoint_tta import PROJECT, read, trial_condition, write

INCUMBENT = PROJECT / "configs/submission-mbv3-d2x2-hflip.json"
COMPARATOR = PROJECT / "scripts/compare_checkpoint_tta.py"
SOURCE_GUARD = 0.01


def passes(report, baseline):
    partition = report["partition"]
    reference = baseline[f"{partition}_by_source"]
    scores = report["result"]["by_source"]
    return (report["eligible"] and report["result"]["score"] > baseline[f"{partition}_score"]
            and scores.keys() == reference.keys()
            and all(scores[group] >= before - SOURCE_GUARD for group, before in reference.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    model = trial_condition(args.trial)
    if model["config"].get("use_training_supplement"):
        raise ValueError("This screen is for models trained on the original competition inputs.")
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = read(INCUMBENT)
    environment = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")

    def compare(partition, *extra):
        subprocess.run([sys.executable, str(COMPARATOR), str(args.trial), partition, "--device", args.device,
                        "--output", str(args.output), *extra], cwd=PROJECT, env=environment, check=True)

    paths = [args.output / f"validation-{variant}.json" for variant in (
        "replace-mbv3" if model["config"]["detector"].startswith("fasterrcnn") else "replace-d2",
        "add-detector")]
    compare("validation")
    results = [(path, read(path)) for path in paths]
    eligible = [(path, report) for path, report in results if passes(report, baseline)]
    outcome = {"trial_id": model["trial_id"], "reference_submission": 58377134,
               "validation_reports": [str(path) for path in paths], "ready": False}
    if eligible:
        selected, report = max(eligible, key=lambda row: row[1]["result"]["score"])
        compare("holdout", "--candidate", str(selected))
        holdout_path = args.output / f"holdout-{report['spec']['variant']}.json"
        outcome.update(candidate=str(selected), holdout=str(holdout_path),
                       ready=passes(read(holdout_path), baseline))
    write(args.output / "screen-outcome.json", outcome)
    print(outcome, flush=True)


if __name__ == "__main__":
    main()
