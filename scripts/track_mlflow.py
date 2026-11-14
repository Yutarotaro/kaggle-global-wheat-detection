# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["mlflow==3.18.0"]
# ///
"""Mirror wheat Loop trials and epoch losses into a local MLflow experiment."""

import argparse
import fcntl
import hashlib
import json
import math
import re
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from urllib.parse import urlsplit

from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag, ViewType
from mlflow.exceptions import MlflowException

PROJECT = Path(__file__).resolve().parent.parent
COMPETITION = "global-wheat-detection"
TERMINAL = {"succeeded": "FINISHED", "failed": "FAILED", "interrupted": "KILLED"}
NUMBER = r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
LOSS = re.compile(rf"^Epoch ([1-9]\d*): loss ({NUMBER})(?: val_loss ({NUMBER}))?"
                  r"(?: optimizer_steps (\d+))?[ \t]*$", re.MULTILINE)
KERNEL = re.compile(r"[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/\d+")


def snapshots(state):
    """The imported baseline has the same trial ID as its original record."""
    records = {}
    for database in sorted(state.glob("*/state.sqlite")):
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
            db.execute("BEGIN")
            metadata = dict(db.execute("SELECT key, value FROM metadata WHERE key IN ('session', 'identity')"))
            if "session" not in metadata:
                continue
            session = json.loads(metadata["session"])
            spec = session["spec"]
            if spec["executor"].get("competition") != COMPETITION:
                continue
            identity = json.loads(metadata.get("identity", "null"))
            comparison = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() if identity else ""
            for (value,) in db.execute("SELECT value FROM trials"):
                trial = json.loads(value)
                trial_id = trial["id"]
                if trial_id not in records or not trial.get("imported_from"):
                    records[trial_id] = database.parent, session, trial, comparison
    return sorted(records.values(), key=lambda record: record[2]["started_at"])


def epoch_losses(path):
    """Final Kaggle logs are JSON; the live CLI stream prints plain text."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    if text.lstrip().startswith("[{"):
        try:
            text = "\n".join(event["data"] for event in json.loads(text) if event["stream_name"] == "stdout")
        except json.JSONDecodeError:
            return {}  # The controller may still be writing the final log.
    rows = {}
    for epoch, train, valid, steps in LOSS.findall(text):
        values = {name: float(value) for name, value in
                  (("train_loss", train), ("valid_loss", valid), ("optimizer_steps", steps)) if value}
        rows[int(epoch)] = {name: value for name, value in values.items() if math.isfinite(value)}
    return rows


class LiveLogs:
    def __init__(self, directory, kaggle):
        self.directory, self.kaggle = directory, kaggle
        directory.mkdir(parents=True, exist_ok=True)
        self.processes = {}

    def path(self, trial_id):
        return self.directory / f"{trial_id}.log"

    def reconcile(self, active):
        for trial_id in self.processes.keys() - active.keys():
            process = self.processes.pop(trial_id)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for trial_id, handle in active.items():
            previous = self.processes.get(trial_id)
            if previous is not None and previous.poll() is None:
                continue
            with self.path(trial_id).open("ab") as output:
                self.processes[trial_id] = subprocess.Popen(
                    [self.kaggle, "kernels", "logs", handle, "--follow"],
                    stdout=output, stdin=subprocess.DEVNULL,
                )


def sync(client, experiment_id, records, live):
    runs = {}
    token = None
    while True:
        page = client.search_runs([experiment_id], filter_string="tags.trial_id IS NOT NULL",
                                  run_view_type=ViewType.ALL, page_token=token)
        runs.update((run.data.tags["trial_id"], run) for run in page)
        token = page.token
        if not token:
            break

    for root, session, trial, comparison in records:
        trial_id = trial["id"]
        status = trial["status"]
        run = runs.get(trial_id)
        if run is not None and run.info.lifecycle_stage == "deleted":
            continue
        started = int(trial["started_at"] * 1000)
        run = run or client.create_run(experiment_id, start_time=started, run_name=f"train/{trial_id}",
                                       tags={"trial_id": trial_id})
        run_id = run.info.run_id
        folder = root / "artifacts" / trial_id
        log = folder / f"ml-loop-{trial_id}.log"
        metrics = []
        losses = epoch_losses(live.path(trial_id)) | epoch_losses(log)
        observed = int(time.time() * 1000)
        for name in {key for values in losses.values() for key in values}:
            recorded = {point.step for point in client.get_metric_history(run_id, name)}
            metrics.extend(Metric(name, values[name], observed, epoch) for epoch, values in sorted(losses.items())
                           if name in values and epoch not in recorded)

        score = trial.get("score")
        metric_name = session["spec"]["task"]["metric"]
        if status == "succeeded" and score is not None and metric_name not in run.data.metrics:
            evaluated = int(trial["evaluated_at"] * 1000)
            metrics.append(Metric(metric_name, score, evaluated, 0))
            details = trial.get("evaluation", {})
            for group in ("by_source", "agent_metrics"):
                metrics.extend(Metric(f"validation.{group}.{name}", value, evaluated, 0)
                               for name, value in details.get(group, {}).items()
                               if type(value) in (int, float) and math.isfinite(value))

        config = root / "sources" / trial_id / "configs/baseline.json"
        params = {"seed": session["spec"]["seed"]}
        if trial.get("source_digest") and config.is_file():
            params.update(json.loads(config.read_text(encoding="utf-8")))

        condition = params.get("run_label")
        if not condition and "epochs" in params:
            condition = f"mbv3-{params['epochs']}e-{params['image_size']}px"
            if params.get("horizontal_flip"):
                condition += "-hflip"
        run_name = f"train/{condition}/{trial_id}" if condition else f"train/{trial_id}"
        conditions = "、".join(f"{key}={value}" for key, value in params.items() if key != "run_label")
        description = "\n\n".join(part for part in (
            f"Kaggle GPU学習。{conditions}。",
            trial.get("hypothesis"), trial.get("change"),
            "検証: 撮影元で固定したvalidation。評価基準: Wheat Image Precision。",
        ) if part)

        tags = {
            "run_kind": "train",
            "mlflow.note.content": description,
            "loop_status": status,
            "loop_session": trial_id.rsplit("-", 1)[0],
            "loop_session_status": session["status"],
            "comparison_group": comparison,
            "completed_epochs": str(max(losses, default=0)),
        }
        for field in ("parent_id", "source_digest", "hypothesis", "change", "detail"):
            if trial.get(field):
                tags[field] = trial[field][:4000]
        if trial.get("handle") and KERNEL.fullmatch(trial["handle"]):
            notebook, version = trial["handle"].rsplit("/", 1)
            tags["kaggle_notebook"] = f"https://www.kaggle.com/code/{notebook}"
            tags["kaggle_version"] = version

        new_tags = [RunTag(key, value) for key, value in tags.items() if run.data.tags.get(key) != value]
        new_params = [Param(key, str(value)) for key, value in params.items() if key not in run.data.params]
        if new_tags or new_params or metrics:
            client.log_batch(run_id, metrics=metrics, params=new_params, tags=new_tags, synchronous=True)
        if run.info.run_name != run_name:
            client.update_run(run_id, name=run_name)

        if status in TERMINAL:
            artifact_paths = (log, folder / "evaluate.log", folder / "evaluation.json",
                              folder / "bundle/experiment.ipynb")
            for path in artifact_paths:
                tag = f"artifact.{path.name}"
                if path.is_file() and run.data.tags.get(tag) != "true":
                    client.log_artifact(run_id, str(path))
                    client.set_tag(run_id, tag, "true")
            end = int(trial.get("completed_at", trial["started_at"]) * 1000)
            if run.info.status != TERMINAL[status]:
                client.set_terminated(run_id, TERMINAL[status], end_time=end)


def sync_inference(client, experiment_id, state):
    directory = state / "inference"
    for manifest in sorted(directory.glob("*.json")):
        record = json.loads(manifest.read_text(encoding="utf-8"))
        submission_id = str(int(record["submission_id"]))
        matches = client.search_runs([experiment_id], f"tags.submission_id = '{submission_id}'")
        name = f"infer/{record['run_label']}/{submission_id}"
        tags = {"run_kind": "infer", "submission_id": submission_id,
                "parent_trial_id": record["parent_trial_id"], "kaggle_notebook": record["notebook"],
                "kaggle_version": str(record["version"]), "mlflow.note.content": record["description"]}
        run = matches[0] if matches else client.create_run(experiment_id, run_name=name, tags=tags)
        run_id = run.info.run_id
        metrics = [Metric(key, value, int(time.time() * 1000), 0) for key, value in record["scores"].items()
                   if key not in run.data.metrics]
        if metrics:
            client.log_batch(run_id, metrics=metrics, synchronous=True)
        notebook = PROJECT / "notebooks" / record["source_notebook"]
        if notebook.is_file() and run.data.tags.get("artifact.notebook") != "true":
            client.log_artifact(run_id, str(notebook))
            client.set_tag(run_id, "artifact.notebook", "true")
        if run.info.status != "FINISHED":
            client.set_terminated(run_id, "FINISHED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=PROJECT / ".loop-wheat")
    parser.add_argument("--tracking-uri", default="http://localhost:5050")
    parser.add_argument("--kaggle", default=str(PROJECT / ".venv/bin/kaggle"))
    parser.add_argument("--watch", action="store_true", help="Refresh status and stream live loss every 15 seconds")
    args = parser.parse_args()
    uri = urlsplit(args.tracking_uri)
    if uri.scheme != "http" or uri.hostname not in {"localhost", "127.0.0.1", "::1"}:
        parser.error("--tracking-uri must be a loopback HTTP address")
    state = args.state.resolve()
    if not state.is_dir():
        parser.error(f"Loop state directory does not exist: {state}")
    live = LiveLogs(state / "mlflow/live", args.kaggle)
    with (live.directory / "sync.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("An MLflow sync is already running for this state directory")
        client = MlflowClient(tracking_uri=args.tracking_uri)
        try:
            while True:
                try:
                    records = snapshots(state)
                    if args.watch:
                        live.reconcile({trial["id"]: trial["handle"] for _, session, trial, _ in records
                                        if session["status"] == "running" and trial["status"] == "running"
                                        and KERNEL.fullmatch(trial.get("handle", ""))})
                    experiment = client.get_experiment_by_name(COMPETITION)
                    experiment_id = experiment.experiment_id if experiment else client.create_experiment(COMPETITION)
                    sync(client, experiment_id, records, live)
                    sync_inference(client, experiment_id, state)
                    if not args.watch:
                        print(f"Synced {len(records)} trials: {args.tracking_uri}/#/experiments/{experiment_id}")
                        return
                except (MlflowException, OSError, sqlite3.OperationalError) as exc:
                    if not args.watch:
                        raise
                    print(f"MLflow sync failed: {exc}", file=sys.stderr, flush=True)
                time.sleep(15)
        finally:
            live.reconcile({})


if __name__ == "__main__":
    main()
