"""Measure false detections on the ten publisher-designated SPIKE negative images."""

import argparse
import json
import subprocess
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from compare_checkpoint_tta import PROJECT, sha, trial_condition, write
from model import create_model

ARCHIVE_SHA256 = "baf2fe4874ee13f2935c4053f41c0f615962c8114c6f048ad9f41f06827bfbbf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT, text=True).strip():
        raise ValueError("Start the comparison from a clean committed tree.")
    archive = PROJECT / "data/spike/SPIKE_dataset.zip"
    if sha(archive) != ARCHIVE_SHA256:
        raise ValueError("Downloaded SPIKE dataset differs from the recorded release.")
    condition = trial_condition(args.trial)
    device = torch.device(args.device)
    torch.set_num_threads(2)
    model = create_model(condition["config"], pretrained=False).to(device).eval()
    checkpoint = args.trial / "weights.pt"
    if sha(checkpoint) != condition["sha256"]:
        raise ValueError("Unexpected training checkpoint.")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    with torch.inference_mode(), ZipFile(archive) as bundle:
        files = [name for name in bundle.namelist() if "/negative/" in name and name.endswith(".jpg")]
        if len(files) != 10 or len(set(files)) != 10:
            raise ValueError("Expected ten distinct publisher-designated negative images.")
        cutoffs = [0.30, 0.40, 0.50]
        if condition["config"]["confidence"] > min(cutoffs):
            raise ValueError("The checkpoint filters predictions above the diagnostic cutoff.")
        counts = {}
        for name in sorted(files):
            with bundle.open(name) as stream, Image.open(stream) as image:
                tensor = pil_to_tensor(image.convert("RGB")).float().div_(255).to(device)
            result, = model([tensor])
            counts[PurePosixPath(name).name] = {cutoff: int((result["scores"] >= cutoff).sum()) for cutoff in cutoffs}
    metrics = {f"false_positives_per_image_at_{cutoff:.2f}":
               sum(row[cutoff] for row in counts.values()) / len(counts) for cutoff in cutoffs}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    report = {"images": len(counts), "metrics": metrics, "detections_by_image": counts,
              "archive_sha256": ARCHIVE_SHA256, "trial_id": condition["trial_id"],
              "weights_sha256": condition["sha256"], "git_commit": commit}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, report)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
