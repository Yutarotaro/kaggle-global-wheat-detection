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


def test_public_archive_conversion_preserves_sources_and_excludes_unlabelled(tmp_path):
    import importlib.util
    import io
    import zipfile

    spec = importlib.util.spec_from_file_location(
        "prepare_gwhd2020", Path(__file__).parents[1] / "scripts/prepare_gwhd2020.py")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    image_bytes = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image_bytes, format="PNG")
    output = tmp_path / "result"
    (output / "train").mkdir(parents=True)
    archive_path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for source in converter.SOURCES:
            archive.writestr(f"{source}.json", json.dumps({
                "categories": [{"id": 1, "name": "wheat_head"}],
                "images": [{"id": 1, "file_name": "image.png", "width": 32, "height": 32},
                           {"id": 2, "file_name": "no-box.png", "width": 32, "height": 32}],
                "annotations": [{"category_id": 1, "image_id": 1, "bbox": [28, 2, 8, 6]}],
            }))
            archive.writestr(f"{source}/image.png", image_bytes.getvalue())
    with zipfile.ZipFile(archive_path) as archive:
        counts = converter.convert(archive, output)
    images, _ = load_dataset(output)
    assert {row["source"] for row in images.values()} == set(converter.SOURCES)
    assert {tuple(row["boxes"][0]) for row in images.values()} == {(28, 2, 32, 8)}
    assert counts["unannotated_images"] == counts["clipped_boxes"] == len(converter.SOURCES)
