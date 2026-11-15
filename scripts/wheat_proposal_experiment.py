"""Collect raw proposals, fit on GWHD 2021, and compare on the frozen Kaggle split."""

import argparse
import csv
import io
import json
import math
import subprocess
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from sweep_confidence import PROJECT, load_dataset, score_cutoffs, sha
from fusion import fuse
from metric import THRESHOLDS
from model import create_model
from proposals import (
    as_array, overlaps, quality_candidates, refine_with_tiles, rescore, tile_predictions, tile_rescue,
)
from submission import predict_views
from supplement import unpack_supplement

WORK = PROJECT / "outputs/proposal-quality"
RAW = PROJECT / "data/raw"
SPLIT = PROJECT / "data/prepared/split.json"
FIELD = PROJECT / "data/gwhd2021/field385"
CONFIG = PROJECT / "configs/experiment-tile-quality.json"


def read(path):
    return json.loads(path.read_text())


def write(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def trial_path(model):
    trial = model["trial_id"]
    return PROJECT / ".loop-wheat" / trial.rsplit("-", 1)[0] / "artifacts" / trial


def collect(model_spec, dataset, keys, target, device, tile_spec=None, batch=4):
    weights = trial_path(model_spec) / "weights.pt"
    meta = {"weights_sha256": model_spec["sha256"], "config": model_spec["config"],
            "tiles": tile_spec, "split_sha256": sha(SPLIT), "torch": torch.__version__,
            "device": device.type, "supplement": sha(FIELD / "manifest.json") if dataset != RAW else None}
    saved = read(target) if target.is_file() else {"meta": meta, "images": {}}
    predictions = saved["images"]
    if saved["meta"] != meta or not predictions.keys() <= set(keys):
        raise ValueError(f"Prediction cache fingerprint mismatch: {target}")
    if sha(weights) != model_spec["sha256"]:
        raise ValueError(f"Checkpoint fingerprint mismatch: {weights}")
    missing = sorted(set(keys) - predictions.keys())
    if not missing:
        return
    model = create_model(model_spec["config"], pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        for index, key in enumerate(missing, 1):
            with Image.open(dataset / "train" / f"{key}.jpg") as image:
                tensor = pil_to_tensor(image.convert("RGB")).float().div_(255).to(device)
            predictions[key] = (tile_predictions(model, tensor, tile_spec, batch) if tile_spec else
                                predict_views(model, tensor, [], 0.55))
            if index % 25 == 0 or index == len(missing):
                write(target, saved)
                print(f"{target.stem} {len(predictions)}/{len(keys)}", flush=True)


def supplement_labels(dataset, boxes):
    with ZipFile(FIELD.parent / "gwhd2021jpg.zip") as archive:
        table = archive.read("gwhd_2021/competition_train.csv").decode("utf-8-sig")
    manifest = read(FIELD / "manifest.json")
    prefix = manifest["image_prefix"]
    groups = {prefix + Path(row["image_name"]).stem: row["domain"]
              for row in csv.DictReader(io.StringIO(table)) if row["domain"] in manifest["sessions"]}
    result = {}
    for key, rows in boxes.items():
        with Image.open(dataset / "train" / f"{key}.jpg") as image:
            width, height = image.size
        result[key] = {"source": groups[key], "width": width, "height": height, "boxes": rows}
    if Counter(record["source"] for record in result.values()) != manifest["sessions"]:
        raise ValueError("Supplement image domains differ from the audited manifest.")
    return result


def fit_quality(labels, rows, tiles, spec, nms):
    stats = {}
    for key, record in labels.items():
        candidates, features = quality_candidates([model[key] for model in rows], tiles.get(key, []),
                                                 record["width"], record["height"], spec["weights"],
                                                 spec["iou"], bool(tiles))
        if not len(candidates):
            continue
        actual = np.asarray(record["boxes"]).reshape(-1, 4)
        best = (overlaps(candidates[:, 1:], actual).max(axis=1) if len(actual) else
                np.zeros(len(candidates)))
        target = (best[:, None] > np.asarray(THRESHOLDS)).mean(axis=1)
        stats.setdefault(record["source"], []).append((features, target))

    by_group = {source: (np.concatenate([x for x, _ in data]), np.concatenate([y for _, y in data]))
                for source, data in stats.items()}

    def train(groups):
        features = np.concatenate([by_group[group][0] for group in groups])
        target = np.concatenate([by_group[group][1] for group in groups])
        center = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-9] = 1
        design = (features - center) / scale
        coef = np.linalg.solve(design.T @ design + 100 * np.eye(design.shape[1]),
                               design.T @ (target - target.mean())) / scale
        return float(target.mean() - center @ coef), coef

    diagnostics = {}
    for group, (features, target) in by_group.items():
        intercept, coefficients = train([other for other in by_group if other != group])
        residual = np.clip(intercept + features @ coefficients, 0, 1) - target
        diagnostics[group] = {"candidates": len(target), "mae": float(np.abs(residual).mean())}
    intercept, coefficients = train(list(by_group))
    return {"intercept": intercept, "coefficients": coefficients.tolist(), "nms_iou": nms}, diagnostics


def summarize(pred, labels):
    groups = {}
    for key, rows in pred.items():
        groups.setdefault(labels[key]["source"], {})[key] = rows
    by_source = {group: score_cutoffs(rows, labels, [0])[0] for group, rows in sorted(groups.items())}
    score = sum(result * len(groups[group]) for group, result in by_source.items()) / len(pred)
    return {"score": score, "by_source": by_source}


def complete(path, keys, envelope=True, identity=None):
    raw = read(path)
    if identity and any(raw["meta"].get(key) != value for key, value in identity.items()):
        raise ValueError(f"Proposal fingerprint mismatch: {path}")
    rows = raw["images"] if envelope else raw
    if rows.keys() != set(keys):
        raise ValueError(f"Incomplete proposal set: {path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["field", "validation-tiles", "field-tiles", "holdout-tiles",
                                          "screen-global", "screen-rescue", "screen-tiles", "confirm"])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--candidate", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    WORK.mkdir(parents=True, exist_ok=True)
    config = read(CONFIG)
    spec = read(PROJECT / config["baseline"])
    tiles = config["tiles"]
    models = spec["models"]
    split = read(SPLIT)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT, text=True).strip():
        raise ValueError("The experiment requires a committed working tree.")
    device = torch.device(args.device)
    with ExitStack() as stack:
        if args.phase in {"field", "field-tiles", "screen-global", "screen-tiles"}:
            dataset, boxes = unpack_supplement(FIELD / "manifest.json",
                                               [FIELD / "gwhd2021-field385.bin"], SPLIT, stack)
            fields = supplement_labels(dataset, boxes)
            if args.phase == "field":
                for index, model in enumerate(models):
                    collect(model, dataset, fields, WORK / f"field-{index}.json", device)
                return
            if args.phase == "field-tiles":
                collect(models[tiles["model_index"]], dataset, fields, WORK / "field-tiles.json",
                        device, tiles, args.batch)
                return
        partition = "holdout" if args.phase.startswith("holdout") or args.phase == "confirm" else "validation"
        keys = split["partitions"][partition]
        if sha(RAW / "train.csv") != split["csv_sha256"]:
            raise ValueError("Annotation fingerprint mismatch.")
        labels, hashes = load_dataset(RAW)
        if any(hashes.get(key) != split["images"][key] for key in keys):
            raise ValueError("Image fingerprint mismatch.")
        if args.phase.endswith("-tiles") and not args.phase.startswith("screen"):
            collect(models[tiles["model_index"]], RAW, keys, WORK / f"{partition}-tiles.json",
                    device, tiles, args.batch)
            return
        if partition == "validation":
            rows = [complete(trial_path(model) / "predictions.json", keys) for model in models]
        else:
            rows = [complete(PROJECT / f"outputs/field385/holdout-{index}.json", keys, False)
                    for index in range(len(models))]

        baseline = {}
        for key in keys:
            rec = labels[key]
            result = fuse([model[key] for model in rows], rec["width"], rec["height"],
                          spec["weights"], spec["iou"])
            baseline[key] = [box for box in result if box[0] >= spec["confidence"]]
        reference = summarize(baseline, labels)
        expected = spec[f"{partition}_score"]
        if not math.isclose(reference["score"], expected, abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"Incumbent reproduction failed: {reference['score']} != {expected}")
        tile_model = models[tiles["model_index"]]
        tile_identity = {"weights_sha256": tile_model["sha256"], "split_sha256": sha(SPLIT),
                         "tiles": tiles, "config": tile_model["config"]}
        if args.phase == "screen-rescue":
            candidates, diagnostics = [{"tiles": tiles}], {}
        elif args.phase.startswith("screen"):
            field_rows = [complete(WORK / f"field-{index}.json", fields) for index in range(len(models))]
            field_tiles = (complete(WORK / "field-tiles.json", fields, identity=tile_identity)
                           if args.phase == "screen-tiles" else {})
            fitted, diagnostics = fit_quality(fields, field_rows, field_tiles, spec, config["quality_nms_iou"])
            candidates = [{"quality": dict(fitted, confidence=cutoff)} for cutoff in config["quality_cutoffs"]]
            if field_tiles:
                candidates = [dict(candidate, tiles=tiles) for candidate in candidates]
                candidates.insert(0, {"tiles": tiles, "quality": dict(
                    fitted, confidence=0.58, refine_with_tiles=True)})
        else:
            screened = read(args.candidate)
            if screened["partition"] != "validation" or not screened["eligible"]:
                raise ValueError("Confirm a single eligible validation candidate.")
            candidates = [screened["candidate"]]
            diagnostics = screened["group_diagnostics"]

        for index, candidate in enumerate(candidates):
            tile_rows = (complete(WORK / f"{partition}-tiles.json", keys, identity=tile_identity)
                         if "tiles" in candidate else {})
            predictions = {}
            for key in keys:
                rec = labels[key]
                if "quality" in candidate:
                    proposals, features = quality_candidates([model[key] for model in rows],
                        tile_rows.get(key, []), rec["width"], rec["height"], spec["weights"],
                        spec["iou"], bool(tile_rows))
                    quality = candidate["quality"]
                    if quality.get("refine_with_tiles"):
                        predictions[key] = refine_with_tiles(baseline[key], proposals, features,
                                                             tile_rows[key], quality)
                    else:
                        predictions[key] = rescore(proposals, features, rec["width"], rec["height"], quality)
                else:
                    predictions[key] = tile_rescue(baseline[key], tile_rows[key], candidate["tiles"])
            result = summarize(predictions, labels)
            eligible = (result["score"] > reference["score"] and all(
                result["by_source"][group] >= score - config["source_guard"]
                for group, score in reference["by_source"].items()))
            report = {"candidate": candidate, "result": result, "baseline": reference, "eligible": eligible,
                      "partition": partition, "git_commit": commit, "split_sha256": sha(SPLIT),
                      "group_diagnostics": diagnostics}
            stem = f"{args.phase}-{index}"
            write(WORK / f"{stem}.json", report)
            write(WORK / f"{stem}-predictions.json", {"images": predictions, "environment": {
                "git_commit": commit, "split_sha256": sha(SPLIT), "recipe": candidate}})
            print(stem, json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
