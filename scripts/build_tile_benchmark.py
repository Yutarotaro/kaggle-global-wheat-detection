"""Build a private Kaggle notebook to collect tile proposals on one frozen partition."""

import argparse
import json
import tempfile
from hashlib import sha256
from pathlib import Path

from build_submission_notebook import PROJECT, build


CELL = """from PIL import Image
from contextlib import ExitStack
from zipfile import ZipFile
import torch
from torchvision.transforms.functional import pil_to_tensor

from data import sha
from model import create_model
from proposals import tile_predictions
from submission import kaggle_inputs

torch.set_num_threads(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset, weights = kaggle_inputs(Path("/kaggle/input"), spec)
if spec["partition"] != "field" and sha(dataset / "train.csv") != spec["csv_sha256"]:
    raise ValueError("Competition annotations differ from the frozen split.")
conditions = spec["models"][spec["tiles"]["model_index"]]
checkpoint = weights[spec["tiles"]["model_index"]]
model = create_model(conditions["config"], pretrained=False).to(device).eval()
model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
rows = {}
with torch.inference_mode(), ExitStack() as resources:
    if spec["partition"] == "field":
        (path,) = Path("/kaggle/input").rglob(spec["archive_file"])
        if sha(path) != spec["archive_sha256"]:
            raise ValueError("GWHD 2021 archive differs from its manifest.")
        bundle = resources.enter_context(ZipFile(path))
        keys = sorted(json.loads(bundle.read(spec["annotation_file"])))
        if len(keys) != spec["image_count"]:
            raise ValueError("GWHD 2021 image count differs from its manifest.")
    else:
        keys = sorted(spec["images"])
    for count, key in enumerate(keys, 1):
        relative = f"train/{key}.jpg"
        if spec["partition"] == "field":
            source = bundle.open(relative)
        else:
            path = dataset / relative
            if sha(path) != spec["images"][key]:
                raise ValueError(f"Competition image does not match the frozen split: {key}")
            source = path.open("rb")
        with source, Image.open(source) as image:
            tensor = pil_to_tensor(image.convert("RGB")).float().div_(255).to(device)
        rows[key] = tile_predictions(model, tensor, spec["tiles"])
        if count % 100 == 0:
            print(f"{count}/{len(keys)} images")
meta = {"weights_sha256": conditions["sha256"], "config": conditions["config"],
        "tiles": spec["tiles"], "split_sha256": spec["split_sha256"],
        "torch": torch.__version__, "device": device.type, "supplement": spec.get("supplement")}
out = Path("/kaggle/working") / (spec["partition"] + "-tiles.json")
out.write_text(json.dumps({"meta": meta, "images": rows}, allow_nan=False) + "\\n")
print(f"Wrote {len(rows)} proposal sets to {out.name}.")
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partition", choices=["validation", "holdout", "field"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Use the specified evaluated checkpoint.")
    parser.add_argument("--label", default="d2")
    parser.add_argument("--save-notebook", action="store_true")
    args = parser.parse_args()
    config = json.loads((PROJECT / "configs/experiment-tile-quality.json").read_text())
    spec = json.loads((PROJECT / config["baseline"]).read_text())
    split_bytes = (PROJECT / "data/prepared/split.json").read_bytes()
    split = json.loads(split_bytes)
    name = f"wheat-benchmark-{args.label}-tiles-{args.partition}"
    spec.update(notebook=f"yutarotaro/{name}", source_notebook=f"{name}.ipynb",
                description=f"固定{args.partition}の画像でEfficientDet-D2の4タイル候補を採取する。"
                "枠の評価・採用判定はローカルの固定評価で実施する。比較元は提出58370638。",
                partition=args.partition, tiles=config["tiles"],
                split_sha256=sha256(split_bytes).hexdigest(), csv_sha256=split["csv_sha256"])
    if args.model:
        conditions = json.loads(args.model.read_text())
        spec.update(models=[conditions], tiles=dict(spec["tiles"], model_index=0),
                    description=f"{conditions['config']['run_label']} の学習済み重みで、"
                                f"固定{args.partition}の4タイル候補を採取する。")
    if args.partition == "field":
        manifest_bytes = (PROJECT / "data/gwhd2021/field385/manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest["split_sha256"] != spec["split_sha256"]:
            raise ValueError("GWHD 2021 manifest references another split.")
        spec.update(archive_file=manifest["archive_file"], archive_sha256=manifest["archive_sha256"],
                    annotation_file=manifest["annotation_file"], image_count=manifest["images"],
                    supplement=sha256(manifest_bytes).hexdigest())
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as path:
        scratch = Path(path)
        notebook = build(spec, scratch)
        document = json.loads(notebook.read_text())
        metadata = json.loads((scratch / "kernel-metadata.json").read_text())
        if args.partition == "field":
            metadata["dataset_sources"].append("yutarotaro/wheat-gwhd2021-field385")
        else:
            keys = split["partitions"][args.partition]
            images = {key: split["images"][key] for key in keys}
            spec_cell = document["cells"][1]["source"]
            # Keep the manifest readable without a single line containing all image hashes.
            spec_cell[4:4] = ["images = {\n", *[f"    {key!r}: {value!r},\n" for key, value in images.items()],
                              "}\n"]
            insert_at = next(index for index, line in enumerate(spec_cell) if line.startswith("Path("))
            spec_cell.insert(insert_at, "spec['images'] = images\n")
        document["cells"][-2]["source"] = ["## 固定分割のタイル候補を保存する"]
        document["cells"][-1]["source"] = CELL.splitlines(keepends=True)
        payload = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
        (args.output / notebook.name).write_text(payload)
        (args.output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if args.save_notebook:
            (PROJECT / "notebooks" / notebook.name).write_text(payload)


if __name__ == "__main__":
    main()
