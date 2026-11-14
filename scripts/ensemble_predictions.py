# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["ensemble-boxes==1.0.9", "mlflow==3.18.0", "pillow>=10.4,<13"]
# ///
"""Fuse saved wheat validation predictions and compare against the individual models."""

import argparse
import json
import math
import sqlite3
import subprocess
import sys
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

from mlflow import MlflowClient

from sweep_confidence import PROJECT, load_dataset, score_cutoffs, sha

sys.path.insert(0, str(PROJECT / "src/wheat"))
from evaluate import evaluate
from fusion import fuse
from render_previews import render_previews

SPLIT = "data/prepared/split.json"
CUTOFFS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
METRIC = "wheat_image_precision"


def read_trial(folder, split_hash, ids):
    root = folder.parent.parent
    path = root / "state.sqlite"
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        metadata = dict(db.execute("SELECT key, value FROM metadata WHERE key IN ('identity', 'session')"))
        trials = [json.loads(value) for (value,) in db.execute("SELECT value FROM trials")]
    trial = next(item for item in trials if item["id"] == folder.name)
    session = json.loads(metadata["session"])
    task = json.loads(metadata["identity"])["task"]
    if trial["status"] != "succeeded" or task.get(f"input:{SPLIT}") != split_hash:
        raise ValueError(f"{trial['id']} did not complete on the frozen competition split.")
    for name in ("data.py", "evaluate.py", "metric.py"):
        relative = Path("src/wheat") / name
        if sha(root / "trusted" / relative) != sha(PROJECT / relative):
            raise ValueError(f"{trial['id']} used a different evaluation implementation: {name}")
    config_name = session["spec"]["executor"].get("notebook_label_file", "configs/baseline.json")
    config = root / "sources" / folder.name / config_name
    conditions = json.loads(config.read_text())
    path = folder / "predictions.json"
    payload = json.loads(path.read_text())
    if payload["images"].keys() != ids:
        raise ValueError(f"{trial['id']} prediction IDs differ from the validation split.")
    if not trial.get("git_commit") and session.get("working_tree", False):
        raise ValueError(f"{trial['id']} has no committed source.")
    return payload["images"], {
        "trial_id": trial["id"],
        "label": conditions.get("run_label", trial["id"]),
        "trial_git_commit": trial.get("git_commit") or (session["head"] if trial["number"] == 1 else None),
        "session_base_commit": session["head"],
        "source_digest": trial["source_digest"],
        "kaggle_handle": trial["handle"],
        "confidence_floor": conditions["confidence"],
        "score": trial["score"],
        "predictions_sha256": sha(path),
    }


def log_run(out, experiment, report, details):
    client = MlflowClient(tracking_uri="http://localhost:5050")
    experiment_id = client.get_experiment_by_name(experiment).experiment_id
    parents = []
    for model in report["models"]:
        (parent,) = client.search_runs([experiment_id], f"tags.trial_id = '{model['trial_id']}'")
        parents.append(parent)
    name = f"infer/wbf-iou{report['iou']:g}-conf{report['confidence']:g}/{out.name}"
    description = (
        "固定validationで、複数モデルが検出した同じ穂の枠をconfidenceで加重平均し、"
        "座標ずれと誤検出を減らせるかを比較。\n\n"
        f"WBFのIoU {report['iou']:g}、出力confidence {report['confidence']:g}、"
        f"モデル重み {report['weights']}。モデル単体と比較。\n\n"
        + "\n".join(f"- [{parent.info.run_name}](#/experiments/{experiment_id}/runs/{parent.info.run_id})"
                    f"（元の検証スコア {model['score']:.6f}）"
                    for model, parent in zip(report["models"], parents))
    )
    run = client.create_run(experiment_id, run_name=name, tags={
        "run_kind": "infer", "inference_kind": "validation_ensemble",
        "mlflow.parentRunId": parents[0].info.run_id,
        "ensemble_parent_run_ids": ",".join(parent.info.run_id for parent in parents),
        "mlflow.note.content": description, "git_commit": report["git_commit"],
        "validation_split_sha256": report["split_sha256"],
    })
    run_id = run.info.run_id
    for key in ("iou", "confidence", "weights"):
        client.log_param(run_id, key, report[key])
    client.log_metric(run_id, METRIC, report["score"])
    for group in ("by_source", "agent_metrics"):
        for name, value in details[group].items():
            client.log_metric(run_id, f"validation.{group}.{name}", value)
    for name in ("comparison.json", "evaluation.json"):
        client.log_artifact(run_id, str(out / name))
    for name in ("validation-previews", "errors"):
        client.log_artifacts(run_id, str(out / name), name)
    client.set_terminated(run_id, "FINISHED")
    return f"http://localhost:5050/#/experiments/{experiment_id}/runs/{run_id}"


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path, nargs="+", help="Completed Loop trial artifact directories")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--weights", type=float, nargs="+", help="Relative weights in trial order; default equal")
    parser.add_argument("--track", action="store_true", help="Log a run to local MLflow at port 5050")
    args = parser.parse_args()
    folders = [path.resolve() for path in args.trials]
    weights = args.weights or [1.0] * len(folders)
    if len(folders) < 2 or len({p.name for p in folders}) != len(folders):
        parser.error("Provide at least two different trial IDs.")
    if len(weights) != len(folders) or any(not math.isfinite(w) or w <= 0 for w in weights):
        parser.error("Provide one positive finite weight per trial.")
    total = sum(weights)
    if not math.isfinite(total):
        parser.error("Sum of weights is too large.")
    weights = [w / total for w in weights]
    if any(weight == 0 for weight in weights):
        parser.error("The relative weights must remain positive after normalization.")
    if not all(0 < value <= 1 for value in (args.iou, args.confidence)):
        parser.error("IoU and confidence must be in (0, 1].")
    if args.output.exists():
        parser.error("Output directory already exists.")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT).strip():
        parser.error("Run from a clean committed worktree.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    manifest_path = PROJECT / SPLIT
    manifest = json.loads(manifest_path.read_text())
    split_hash = sha(manifest_path)
    ids = set(manifest["partitions"]["validation"])
    if not ids:
        raise ValueError("The frozen split has no validation images.")
    sources = [read_trial(folder, split_hash, ids) for folder in folders]
    data = PROJECT / "data/raw"
    if sha(data / "train.csv") != manifest["csv_sha256"]:
        raise ValueError("Annotations do not match the frozen split.")
    labels, hashes = load_dataset(data)
    if any(hashes[key] != manifest["images"][key] for key in ids):
        raise ValueError("Images do not match the frozen validation split.")
    cutoffs = sorted(set(CUTOFFS) | {args.confidence})
    for predictions, model in sources:
        floor = model["confidence_floor"]
        scores = score_cutoffs(predictions, labels, sorted({c for c in cutoffs if c >= floor} | {floor}))
        if not math.isclose(scores[floor], model["score"], rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"Individual scores disagree with the saved evaluation: {model['trial_id']}.")
        model["score_by_cutoff"] = scores
    fused = {}
    for image_id in sorted(ids):
        record = labels[image_id]
        fused[image_id] = fuse([predictions[image_id] for predictions, _ in sources],
                               record["width"], record["height"], weights, args.iou)
    sweep = score_cutoffs(fused, labels, cutoffs)

    out = args.output.resolve()
    out.mkdir(parents=True)
    environment = {name: version(name) for name in ("ensemble-boxes", "numpy", "pillow")}
    environment["python"] = sys.version.split()[0]
    selected = {key: [box for box in boxes if box[0] >= args.confidence] for key, boxes in fused.items()}
    write_json(out / "predictions.json", {"images": selected, "environment": environment})
    evaluation = evaluate(data, manifest, out, "validation")
    score = evaluation["score"]
    if not math.isclose(score, sweep[args.confidence], rel_tol=0, abs_tol=1e-10):
        raise ValueError("Ensemble confidence sweep disagrees with the independent evaluation.")
    write_json(out / "evaluation.json", evaluation)
    render_previews(PROJECT, out)
    report = {
        "partition": "validation", "git_commit": commit, "split_sha256": split_hash,
        "iou": args.iou, "confidence": args.confidence, "weights": weights, "confidence_type": "box_and_model_avg",
        "score": score, "score_by_cutoff": sweep, "models": [model for _, model in sources],
    }
    write_json(out / "comparison.json", report)
    if args.track:
        report["mlflow_run_url"] = log_run(out, "global-wheat-detection", report, evaluation["details"])
        write_json(out / "comparison.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
