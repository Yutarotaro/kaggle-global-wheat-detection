"""Reconcile local Kaggle submission manifests with the scoring service."""

import argparse
import json
import math
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=Path(".loop-wheat"))
    parser.add_argument("--kaggle", default="kaggle")
    args = parser.parse_args()
    pending = {}
    for path in sorted((args.state / "inference").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("status") == "pending":
            pending[int(record["submission_id"])] = path, record
    if not pending:
        return
    remote = json.loads(subprocess.check_output([
        args.kaggle, "competitions", "submissions", "-c", "global-wheat-detection",
        "--page-size", "200", "--format", "json",
    ], text=True, timeout=30))
    for row in remote:
        number = int(row["ref"])
        if number not in pending:
            continue
        path, record = pending[number]
        status = row["status"].rsplit(".", 1)[-1].lower()
        if status not in {"complete", "error"}:
            continue
        if status == "complete":
            public, private = (float(row[key]) for key in ("publicScore", "privateScore"))
            if not all(math.isfinite(score) and 0 <= score <= 1 for score in (public, private)):
                raise ValueError(f"Invalid Kaggle scores for {number}.")
            record["scores"].update(kaggle_public_score=public, kaggle_private_score=private)
        record["status"] = status
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(path)
        print(json.dumps({"submission_id": number, "status": status, "scores": record["scores"]}))


if __name__ == "__main__":
    main()
