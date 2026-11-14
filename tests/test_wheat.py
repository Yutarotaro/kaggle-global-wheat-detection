import csv
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "src/wheat"))

from data import load_dataset, sha, split_dataset
from evaluate import evaluate
from metric import image_score, match


def test_metric_penalizes_duplicates_and_respects_confidence():
    truth = [[0, 0, 10, 10]]
    assert image_score(truth, [[0.9, 0, 0, 10, 10]]) == 1
    assert image_score(truth, [[0.9, 0, 0, 10, 10], [0.8, 0, 0, 10, 10]]) == 0.5
    assert image_score(truth, []) == 0
    assert image_score([], [[0.9, 0, 0, 10, 10]]) == 0
    assert match(truth, [[0.4, 0, 0, 10, 10], [0.9, 0, 0, 10, 10]], 0.5) == ([1], [], [0], [])
    assert match(truth, [[0.4, 0, 0, 5, 10]], 0.5) == ([], [0], [], [0])


@pytest.fixture
def dataset(tmp_path):
    (tmp_path / "train").mkdir()
    rows = []
    for i in range(8):
        key = str(i)
        Image.new("RGB", (32, 32), (i * 20, 0, 0)).save(tmp_path / "train" / f"{key}.jpg")
        rows.append(dict(image_id=key, source=f"source-{i // 2}", width=32, height=32, bbox="[2,3,10,8]"))
    with (tmp_path / "train.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def test_group_split_and_exact_duplicates(dataset):
    (dataset / "train/2.jpg").write_bytes((dataset / "train/0.jpg").read_bytes())
    images, hashes = load_dataset(dataset)
    split = split_dataset(images, hashes)
    assert split == split_dataset(dict(reversed(list(images.items()))), hashes)
    sources = [{images[k]["source"] for k in ids} for ids in split.values()]
    assert not any(a & b for i, a in enumerate(sources) for b in sources[i + 1:])
    assert any({"0", "2"} <= set(ids) for ids in split.values())


def test_independent_evaluator_rejects_missing_and_invalid_predictions(dataset):
    rows, hashes = load_dataset(dataset)
    partitions = split_dataset(rows, hashes)
    manifest = {"partitions": partitions, "csv_sha256": sha(dataset / "train.csv"), "images": hashes}
    out = dataset / "result"
    out.mkdir()
    predictions = out / "predictions.json"
    ids = partitions["validation"]
    predictions.write_text(json.dumps({"images": {}, "environment": {}}))
    with pytest.raises(ValueError, match="IDs"):
        evaluate(dataset, manifest, out, "validation")
    predictions.write_text(json.dumps({"images": {k: [[0.9, 2, 3, 12, 11]] for k in ids}, "environment": {}}))
    assert evaluate(dataset, manifest, out, "validation")["score"] == 1
    predictions.write_text(json.dumps({"images": {k: [[float("nan"), 2, 3, 12, 11]] for k in ids}, "environment": {}}))
    with pytest.raises(ValueError, match="finite"):
        evaluate(dataset, manifest, out, "validation")
