# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mlflow==3.18.0", "pillow>=10.4,<13"]
# ///
"""Verify proposal experiments with the frozen evaluator and record them in local MLflow."""

import argparse
import json
import math
import sys
from pathlib import Path

from mlflow import MlflowClient

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src/wheat"))
from data import sha
from evaluate import evaluate

DESCRIPTIONS = {
    "screen-rescue": "D2の4つの重なりタイルから、小さい穂の候補を既存のWBFの結果に追加する。",
    "screen-global": "既存モデルの枠を、独立したGWHD 2021の7撮影元で学習した枠品質で選び直す。",
    "screen-tiles": "全体画像とD2の高解像度タイルを併用し、GWHD 2021で学習した枠品質で枠を選ぶ。",
    "confirm": "固定検証で採用した手順を凍結し、独立撮影元で再現性を確認する。",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    client = MlflowClient(tracking_uri="http://localhost:5050")
    experiment_id = client.get_experiment_by_name("global-wheat-detection").experiment_id
    parents = [next(run for run in client.search_runs([experiment_id], f"tags.trial_id = '{trial}'")
                    if run.data.tags["run_kind"] == "train")
               for trial in ("d31623fdd5114e49-0001", "8fc98f3e68e8441a-0001")]
    split_path = PROJECT / "data/prepared/split.json"
    split = json.loads(split_path.read_text())
    for path in args.reports:
        report = json.loads(path.read_text())
        if sha(split_path) != report["split_sha256"]:
            raise ValueError("Proposal report uses another split.")
        predictions_file = path.with_name(path.stem + "-predictions.json")
        digest = sha(predictions_file)
        if client.search_runs([experiment_id],
                              f"tags.proposal_predictions_sha256 = '{digest}' and attributes.status = 'FINISHED'"):
            continue
        artifact = path.parent / "mlflow" / path.stem
        artifact.mkdir(parents=True, exist_ok=True)
        payload = json.loads(predictions_file.read_text())
        payload.setdefault("environment", {"git_commit": report["git_commit"],
                                            "split_sha256": report["split_sha256"]})
        (artifact / "predictions.json").write_text(json.dumps(payload, allow_nan=False))
        partition = report["partition"]
        confirmed = evaluate(PROJECT / "data/raw", split, artifact, partition)
        if not math.isclose(confirmed["score"], report["result"]["score"], abs_tol=1e-9, rel_tol=0):
            raise ValueError("The independent evaluator disagrees with the screening score.")
        (artifact / "evaluation.json").write_text(json.dumps(confirmed, indent=2, allow_nan=False))
        phase = path.stem.rsplit("-", 1)[0]
        quality = report["candidate"].get("quality", {})
        label = phase if not quality else f"{phase}-q{quality['confidence']:g}"
        if quality.get("refine_with_tiles"):
            label += "-refine"
        description = DESCRIPTIONS[phase] + "\n\n比較対象は提出 58370638 のWBF。"
        description += "\n\n" + "\n".join(
            f"- [{run.info.run_name}](#/experiments/{experiment_id}/runs/{run.info.run_id})"
            for run in parents)
        run = client.create_run(experiment_id, run_name=f"infer/{label}/{digest[:8]}", tags={
            "run_kind": "infer", "inference_kind": "validation_proposals",
            "mlflow.parentRunId": parents[0].info.run_id,
            "model_input_run_ids": ",".join(parent.info.run_id for parent in parents),
            "proposal_predictions_sha256": digest, "git_commit": report["git_commit"],
            "validation_split_sha256": report["split_sha256"], "mlflow.note.content": description,
        })
        run_id = run.info.run_id
        client.log_metric(run_id, f"{partition}_score", confirmed["score"])
        client.log_metric(run_id, f"{partition}_baseline", report["baseline"]["score"])
        client.log_param(run_id, "reference_submission", "58370638")
        client.log_param(run_id, "passed_source_guard", report["eligible"])
        for group in ("by_source", "agent_metrics"):
            for key, value in confirmed["details"][group].items():
                client.log_metric(run_id, f"{partition}.{group}.{key}", value)
        for file in (path, artifact / "evaluation.json", artifact / "predictions.json"):
            client.log_artifact(run_id, str(file))
        client.log_artifacts(run_id, str(artifact / "errors"), "errors")
        client.set_terminated(run_id, "FINISHED")
        print(f"{path.stem} http://localhost:5050/#/experiments/{experiment_id}/runs/{run_id}", flush=True)


if __name__ == "__main__":
    main()
