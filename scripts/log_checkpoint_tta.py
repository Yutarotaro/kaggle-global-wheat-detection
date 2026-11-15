# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mlflow==3.18.0"]
# ///
"""Save fixed-partition comparisons of trained detectors in local MLflow."""

import argparse
import hashlib
import json
from pathlib import Path

from mlflow import MlflowClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    client = MlflowClient(tracking_uri="http://localhost:5050")
    experiment = client.get_experiment_by_name("global-wheat-detection").experiment_id
    for path in args.reports:
        report = json.loads(path.read_text())
        spec = report["spec"]
        variant = spec["variant"]
        models = spec["models"]
        trials = [model["trial_id"] for model in models]
        candidate = report.get("candidate_trial", trials[-1])
        conditions = next(model["config"] for model in models if model["trial_id"] == candidate)
        name = f"infer/tta-{variant}-{conditions['run_label']}/{candidate}"
        partition = report["partition"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        matches = client.search_runs([experiment], f"tags.comparison_spec_sha256 = '{digest}'")
        if matches and matches[0].info.status == "FINISHED":
            continue
        parents = {}
        for trial in trials:
            parents[trial] = next(run for run in client.search_runs([experiment], f"tags.trial_id = '{trial}'")
                                  if run.data.tags.get("run_kind") == "train")
        links = "\n".join(f"- [{run.info.run_name}](#/experiments/{experiment}/runs/{run.info.run_id})"
                          for run in parents.values())
        description = (
            f"提出58374318の反転TTAで、学習した新しい重みを比較した。"
            f"方法は {variant}、評価分割は {partition}、WBF重みは {spec['weights']}。"
            f"\n\n{links}"
        )
        run = matches[0] if matches else client.create_run(experiment, run_name=f"{name}/{partition}", tags={
            "run_kind": "infer", "inference_kind": "validation_tta",
            "mlflow.parentRunId": parents[candidate].info.run_id,
            "model_input_run_ids": ",".join(run.info.run_id for run in parents.values()),
            "comparison_spec_sha256": digest, "git_commit": report["git_commit"],
            "validation_split_sha256": report["split_sha256"], "mlflow.note.content": description,
        })
        run_id = run.info.run_id
        client.log_metric(run_id, f"{partition}_score", report["result"]["score"])
        client.log_metric(run_id, f"{partition}_baseline", report["baseline"]["score"])
        for source, value in report["result"]["by_source"].items():
            client.log_metric(run_id, f"{partition}.by_source.{source}", value)
        client.log_param(run_id, "passed_source_guard", report["eligible"])
        client.log_artifact(run_id, str(path))
        client.set_terminated(run_id, "FINISHED")
        print(f"{partition} {variant} http://localhost:5050/#/experiments/{experiment}/runs/{run_id}", flush=True)


if __name__ == "__main__":
    main()
