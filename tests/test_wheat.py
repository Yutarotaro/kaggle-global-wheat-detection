import csv
import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "src/wheat"))

from data import load_dataset, sha, split_dataset
from evaluate import evaluate
from metric import image_score, match
from preprocess import mosaic, zoom_crop
from render_previews import render_previews


def test_mosaic_clips_and_translates_boxes(monkeypatch):
    monkeypatch.setattr("preprocess.random.randint", lambda a, b: 4)
    samples = [
        (torch.full((3, 8, 8), i), {"boxes": torch.tensor([[1., 1., 6., 6.]]),
                                  "labels": torch.tensor([i + 1])})
        for i in range(4)
    ]
    image, target = mosaic(samples)
    assert [image[0, y, x].item() for y, x in ((0, 0), (0, 7), (7, 0), (7, 7))] == [0, 1, 2, 3]
    assert target["boxes"].tolist() == [[0, 0, 2, 2], [5, 0, 8, 2], [0, 5, 2, 8], [5, 5, 8, 8]]
    assert target["labels"].tolist() == [1, 2, 3, 4]


def test_zoom_crop_preserves_visible_boxes_and_labels():
    class FixedCrop:
        positions = iter((20, 10))

        def uniform(self, _low, _high):
            return 0.60

        def randrange(self, _stop):
            return next(self.positions)

    image = torch.arange(3 * 100 * 100).reshape(3, 100, 100)
    boxes = torch.tensor([[23., 17., 44., 39.], [10., 25., 30., 40.], [70., 40., 99., 60.]])
    crop, target = zoom_crop(image, {"boxes": boxes, "labels": torch.tensor([1, 2, 3])}, FixedCrop())
    assert torch.equal(crop, image[:, 10:70, 20:80])
    assert target["boxes"].tolist() == [[3, 7, 24, 29], [0, 15, 10, 30]]
    assert target["labels"].tolist() == [1, 2]
    assert boxes[1, 0].item() == 10


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


def test_epoch_previews_require_all_epochs_and_use_the_same_samples(dataset):
    project = dataset / "project"
    raw = project / "data/raw"
    raw.mkdir(parents=True)
    (dataset / "train").rename(raw / "train")
    (dataset / "train.csv").rename(raw / "train.csv")
    rows, hashes = load_dataset(raw)
    partitions = split_dataset(rows, hashes)
    manifest = {"partitions": partitions, "csv_sha256": sha(raw / "train.csv"), "images": hashes}
    prepared = project / "data/prepared"
    prepared.mkdir()
    (prepared / "split.json").write_text(json.dumps(manifest))
    artifacts = dataset / "result"
    epochs = artifacts / "epoch-previews"
    epochs.mkdir(parents=True)
    ids = partitions["validation"]
    (epochs / "epoch-001.json").write_text(json.dumps(dict.fromkeys(ids, [])))
    with pytest.raises(ValueError, match="Expected 2"):
        render_previews(project, artifacts, 2)
    (epochs / "epoch-002.json").write_text(json.dumps(dict.fromkeys(ids, [[0.9, 2, 3, 12, 11]])))
    render_previews(project, artifacts, 2)
    for epoch in (1, 2):
        paths = list((artifacts / f"validation-previews/epoch-{epoch:03d}").glob("*.jpg"))
        assert {path.stem for path in paths} == set(ids)
        for path in paths:
            with Image.open(path) as image:
                image.verify()


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
