"""Compare a trained checkpoint with the fixed two-detector TTA ensemble."""

import argparse
import json
import math
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import torch

from sweep_confidence import PROJECT, load_dataset, score_cutoffs, sha
from fusion import fuse
from model import create_model
from preprocess import read_image
from proposals import tile_rescue
from submission import predict_views

RAW = PROJECT / "data/raw"
SPLIT = PROJECT / "data/prepared/split.json"
BASELINE = PROJECT / "configs/submission-d2-mbv3-hflip-tta.json"


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def trial_condition(path):
    if path.is_file():
        return read(path)
    trial = path.name
    state = PROJECT / ".loop-wheat" / trial.rsplit("-", 1)[0]
    expected = state / "artifacts" / trial
    if path.resolve() != expected.resolve():
        raise ValueError(f"Expected a condition file or a wheat trial directory: {path}.")
    uri = (state / "state.sqlite").as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as database:
        (payload,) = database.execute("SELECT value FROM trials WHERE id=?", (trial,)).fetchone()
        metadata = dict(database.execute("SELECT key, value FROM metadata WHERE key IN ('session', 'identity')"))
    record = json.loads(payload)
    session = json.loads(metadata["session"])
    if record["status"] != "succeeded" or not record.get("git_commit") or not record.get("handle"):
        raise ValueError("Expected a successfully evaluated, committed Kaggle training run.")
    for name in ("data.py", "evaluate.py", "metric.py"):
        relative = Path("src/wheat") / name
        if sha(state / "trusted" / relative) != sha(PROJECT / relative):
            raise ValueError(f"Training run used a different evaluator: {name}.")
    identity = json.loads(metadata["identity"])
    if identity["task"]["input:data/prepared/split.json"] != sha(SPLIT):
        raise ValueError("Training run used a different data split.")
    config = session["spec"]["executor"]["notebook_label_file"]
    return {"trial_id": trial, "notebook": record["handle"], "sha256": sha(path / "weights.pt"),
            "config": read(state / "sources" / trial / config)}


def predict(condition, partition, keys, device, path):
    trial = condition["trial_id"]
    folder = PROJECT / ".loop-wheat" / trial.rsplit("-", 1)[0] / "artifacts" / trial
    checkpoint = folder / "weights.pt"
    if sha(checkpoint) != condition["sha256"]:
        raise ValueError(f"Unexpected checkpoint for {trial}")
    identity = {"condition": condition, "split": sha(SPLIT), "partition": partition,
                "torch": torch.__version__, "device": device.type}
    cache = read(path) if path.is_file() else {"identity": identity, "images": {}}
    rows = cache["images"]
    if cache["identity"] != identity or not rows.keys() <= set(keys):
        raise ValueError(f"Prediction cache does not match the experiment: {path}")
    missing = sorted(set(keys) - rows.keys())
    if not missing:
        return rows
    model = create_model(condition["config"], pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    raw = read(folder / "predictions.json")["images"] if partition == "validation" else {}
    if raw and raw.keys() != set(keys):
        raise ValueError("Saved trial predictions differ from the fixed validation split.")
    with torch.inference_mode():
        for counted, key in enumerate(missing, 1):
            image, _ = read_image(RAW / "train" / f"{key}.jpg", [], 0)
            image = image.to(device)
            height, width = image.shape[-2:]
            if raw:
                reflected, = model([image.flip(-1)])
                boxes = reflected["boxes"].clone()
                boxes[:, (0, 2)] = width - boxes[:, (2, 0)]
                flipped = [[score, *box] for score, box in
                           zip(reflected["scores"].tolist(), boxes.tolist(), strict=True)]
                rows[key] = fuse([raw[key], flipped], width, height, [1, 1], 0.55)
            else:
                rows[key] = predict_views(model, image, ["horizontal"], 0.55)
            if counted % 50 == 0 or counted == len(missing):
                write(path, cache)
                print(f"{trial} {partition} TTA {len(rows)}/{len(keys)}", flush=True)
    return rows


def score(sources, labels, ids, spec, tiles=None):
    partitions = {}
    cutoff = spec["confidence"]
    for key in ids:
        row = labels[key]
        guesses = fuse([source[key] for source in sources], row["width"], row["height"],
                       spec["weights"], spec["iou"])
        guesses = [box for box in guesses if box[0] >= cutoff]
        if tiles is not None:
            guesses = tile_rescue(guesses, tiles[key], spec["tiles"])
        partitions.setdefault(row["source"], {})[key] = guesses
    by_source = {group: score_cutoffs(rows, labels, [0])[0] for group, rows in partitions.items()}
    combined = sum(value * len(partitions[group]) for group, value in by_source.items()) / len(ids)
    return {"score": combined, "by_source": by_source}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("partition", choices=("validation", "holdout"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--candidate", type=Path, help="Eligible frozen validation recipe for holdout.")
    parser.add_argument("--tile-predictions", type=Path, help="Candidate's frozen tile predictions.")
    args = parser.parse_args()
    if (args.partition == "holdout") != (args.candidate is not None):
        parser.error("Holdout requires exactly one frozen validation candidate.")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT, text=True).strip():
        raise ValueError("Start the comparison from a clean committed tree.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    split = read(SPLIT)
    keys = sorted(split["partitions"][args.partition])
    labels, hashes = load_dataset(RAW)
    if sha(RAW / "train.csv") != split["csv_sha256"] or any(hashes[key] != split["images"][key] for key in keys):
        raise ValueError("Dataset fingerprint differs from the fixed split.")
    device = torch.device(args.device)
    torch.set_num_threads(3)
    args.output.mkdir(parents=True, exist_ok=True)
    model = trial_condition(args.model)
    incumbent = read(BASELINE)
    incumbent_rows = ([view[1] for view in read(PROJECT / "outputs/field385/tta-predictions.json")]
                      if args.partition == "validation" else
                      [read(PROJECT / f"outputs/field385/holdout-tta-{i}.json") for i in range(2)])
    if any(rows.keys() != set(keys) for rows in incumbent_rows):
        raise ValueError("Incumbent TTA cache does not match the fixed partition.")
    candidate = predict(model, args.partition, keys, device, args.output / f"{args.partition}-tta.json")
    reference = score(incumbent_rows, labels, keys, incumbent)
    published = incumbent[f"{args.partition}_score"]
    # The earlier screen used weights [1, 2]; the submitted floats were [1/3, 2/3].
    tolerance = 0.00002 if args.partition == "validation" else 1e-9
    if not math.isclose(reference["score"], published, abs_tol=tolerance, rel_tol=0):
        raise ValueError(f"Incumbent score mismatch: {reference}")
    rows = {**dict(zip((m["trial_id"] for m in incumbent["models"]), incumbent_rows, strict=True)),
            model["trial_id"]: candidate}
    replace_mobile = model["config"]["detector"].startswith("fasterrcnn")
    updated = [model, incumbent["models"][1]] if replace_mobile else [incumbent["models"][0], model]
    replacement = "replace-mbv3" if replace_mobile else "replace-d2"
    variants = [(updated, incumbent["weights"], replacement),
                (incumbent["models"] + [model], [0.25, 0.5, 0.25], "add-detector")]
    recipes = [dict(incumbent, models=models, weights=weights, variant=variant)
               for models, weights, variant in variants]
    if args.candidate:
        selected = read(args.candidate)
        if not selected["eligible"] or selected["partition"] != "validation":
            raise ValueError("Holdout candidate must pass the fixed validation comparison.")
        recipes = [selected["spec"]]
    tile_rows = None
    if args.tile_predictions:
        payload = read(args.tile_predictions)
        tile_spec = read(PROJECT / "configs/experiment-tile-quality.json")["tiles"]
        meta = payload["meta"]
        if (meta["weights_sha256"] != model["sha256"] or meta["config"] != model["config"] or
            meta["tiles"] != dict(tile_spec, model_index=0) or meta["split_sha256"] != sha(SPLIT) or
            payload["images"].keys() != set(keys)):
            raise ValueError("Tile predictions have the wrong model, configuration or image IDs.")
        tile_rows = payload["images"]
        if not args.candidate:
            recipes = [dict(recipe, variant=recipe["variant"] + "-tiles",
                            tiles=dict(tile_spec, model_index=[m["trial_id"] for m in recipe["models"]].index(
                                model["trial_id"]))) for recipe in recipes]
    if any(("tiles" in recipe) != bool(args.tile_predictions) for recipe in recipes):
        raise ValueError("The frozen comparison and supplied tiles must agree.")
    for recipe in recipes:
        result = score([rows[m["trial_id"]] for m in recipe["models"]], labels, keys, recipe, tile_rows)
        pass_guard = result["score"] > reference["score"] and all(
            result["by_source"][key] >= value - 0.01 for key, value in reference["by_source"].items())
        report = {"partition": args.partition, "eligible": pass_guard, "git_commit": commit,
                  "candidate_trial": model["trial_id"],
                  "spec": recipe, "baseline": reference, "result": result, "split_sha256": sha(SPLIT)}
        write(args.output / f"{args.partition}-{recipe['variant']}.json", report)
        print(recipe["variant"], result, f"eligible={pass_guard}", flush=True)


if __name__ == "__main__":
    main()
