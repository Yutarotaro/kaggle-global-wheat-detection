"""Build a readable, offline Kaggle inference notebook from a frozen submission spec."""

import argparse
import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MODULES = ("data", "metric", "fusion", "model", "submission")


def build(spec, destination):
    slug = spec["notebook"].split("/")[-1]
    notebook_name = spec["source_notebook"]
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "id": f"cell-{len(cells):03d}", "metadata": {},
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", f"# {slug}\n\n{spec['description']}")
    add("code", '''import hashlib
import json
import subprocess
import sys
from pathlib import Path

spec = json.loads(Path("submission-spec.json").read_text())
wheel_paths = []
for name, expected in spec["wheels"].items():
    (wheel,) = Path("/kaggle/input").rglob(name)
    with wheel.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
            raise ValueError(f"Wheel checksum mismatch: {name}")
    wheel_paths.append(str(wheel))
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                "--force-reinstall", *wheel_paths], check=True)
'''.replace('spec = json.loads(Path("submission-spec.json").read_text())',
            f"spec = json.loads({json.dumps(spec, ensure_ascii=False)!r})\n"
            'Path("submission-spec.json").write_text(json.dumps(spec))'))
    for module in MODULES:
        source = (PROJECT / "src/wheat" / f"{module}.py").read_text()
        add("markdown", f"## {module}.py")
        add("code", f"%%writefile {module}.py\n{source}")
    add("markdown", "## 推論と提出用CSVの作成")
    add("code", 'from submission import run_kaggle\n\nrun_kaggle(Path("submission-spec.json"))\n')
    document = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "language": "python",
                "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 5}
    destination.mkdir(parents=True, exist_ok=True)
    (destination / notebook_name).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    metadata = {"id": spec["notebook"], "title": slug, "code_file": notebook_name, "language": "python",
                "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": False,
                "dataset_sources": [spec["wheel_dataset"]], "competition_sources": ["global-wheat-detection"],
                "kernel_sources": [model["notebook"].rsplit("/", 1)[0] for model in spec["models"]]}
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return destination / notebook_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-notebook", action="store_true")
    args = parser.parse_args()
    result = build(json.loads(args.spec.read_text()), args.output)
    if args.save_notebook:
        shutil.copyfile(result, PROJECT / "notebooks" / result.name)


if __name__ == "__main__":
    main()
