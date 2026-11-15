"""YOLOX-S using the same wheat training and prediction contract as the other detectors."""

import hashlib
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import torch
from torch import nn
from torchvision.ops import nms
from torchvision.transforms.functional import resize

SOURCE_REVISION = "419778480ab6ec0590e5d3831b3afb3b46ab2aa3"
SOURCE_URL = f"https://codeload.github.com/Megvii-BaseDetection/YOLOX/zip/{SOURCE_REVISION}"
SOURCE_SHA256 = "68e5d3abea0337c92e051816edd9182d78aa67424ac3cfea32664ea8f32e7e81"
WEIGHTS_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
WEIGHTS_SHA256 = "f55ded7181e1b0c13285c56e7790b8f0e8f8db590fe4edb37f0b7f345c913a30"


def fetch(path, url, expected):
    if not path.is_file():
        temporary = path.with_suffix(".download")
        urlretrieve(url, temporary)
        temporary.replace(path)
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
            raise ValueError(f"Unexpected SHA-256 for {path.name}.")
    return path


def prepare_yolox():
    missing = [package for module, package in (("cv2", "opencv-python-headless==4.13.0.92"),
                                                ("loguru", "loguru==0.7.3"),
                                                ("thop", "thop==0.1.1.post2209072238"))
               if find_spec(module) is None]
    if missing:
        if os.environ.get("ML_LOOP_REMOTE") != "kaggle":
            raise RuntimeError("Install YOLOX dependencies with uv sync --group wheat --group yolo.")
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        *missing], check=True)
    project = Path(os.environ.get("ML_LOOP_PROJECT", Path(__file__).resolve().parents[2]))
    cache = project / ".cache/yolox"
    cache.mkdir(parents=True, exist_ok=True)
    source = fetch(cache / "source.zip", SOURCE_URL, SOURCE_SHA256)
    source_root = f"YOLOX-{SOURCE_REVISION}"
    with ZipFile(source) as archive:
        archive.extractall(cache, (name for name in archive.namelist()
                                   if name.startswith(f"{source_root}/yolox/")))
    sys.path.insert(0, str(cache / source_root))
    return cache


class YoloWheat(nn.Module):
    def __init__(self, config, pretrained):
        super().__init__()
        cache = prepare_yolox()
        from yolox.models import YOLOPAFPN, YOLOXHead

        self.size = config["image_size"]
        self.confidence = config["confidence"]
        self.nms = config["nms"]
        if self.size % 32:
            raise ValueError("YOLOX image size must be a multiple of 32.")
        self.backbone = YOLOPAFPN(depth=0.33, width=0.5)
        self.loss_bench = YOLOXHead(num_classes=1, width=0.5)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = 1e-3
                module.momentum = 0.03
        self.loss_bench.initialize_biases(1e-2)
        if pretrained:
            path = fetch(cache / "yolox_s.pth", WEIGHTS_URL, WEIGHTS_SHA256)
            state = torch.load(path, map_location="cpu", weights_only=True)["model"]
            backbone = {key.removeprefix("backbone."): value for key, value in state.items()
                        if key.startswith("backbone.")}
            head = {key.removeprefix("head."): value for key, value in state.items()
                    if key.startswith("head.") and not key.startswith("head.cls_preds.")}
            self.backbone.load_state_dict(backbone)
            result = self.loss_bench.load_state_dict(head, strict=False)
            expected = {key for key in self.loss_bench.state_dict() if key.startswith("cls_preds.")}
            if set(result.missing_keys) != expected or result.unexpected_keys:
                raise ValueError(f"YOLOX pretrained head does not match: {result}.")

    def forward(self, images, labels=None):
        shapes = [image.shape[-2:] for image in images]
        # The published YOLOX-S weights use BGR inputs on the 0–255 scale.
        inputs = torch.stack([resize(image.flip(0), [self.size, self.size], antialias=True)
                              for image in images]) * 255
        features = self.backbone(inputs)
        if labels is not None:
            count = max(1, max(len(target["boxes"]) for target in labels))
            targets = inputs.new_zeros((len(labels), count, 5))
            for i, (target, (h, w)) in enumerate(zip(labels, shapes, strict=True)):
                boxes = target["boxes"] * inputs.new_tensor([self.size / w, self.size / h] * 2)
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                targets[i, :len(boxes), 1:3] = centers
                targets[i, :len(boxes), 3:] = boxes[:, 2:] - boxes[:, :2]
            loss, *_ = self.loss_bench(features, targets, inputs)
            return {"loss": loss}

        outputs = []
        for prediction, (h, w) in zip(self.loss_bench(features), shapes, strict=True):
            if not torch.isfinite(prediction).all():
                raise ValueError("Non-finite YOLOX prediction.")
            scores = prediction[:, 4] * prediction[:, 5]
            qualified = scores >= self.confidence
            scores = scores[qualified]
            centers = prediction[qualified, :2]
            half_sizes = prediction[qualified, 2:4] / 2
            boxes = torch.cat((centers - half_sizes, centers + half_sizes), dim=1)
            boxes = boxes.clamp(0, self.size) * boxes.new_tensor([w / self.size, h / self.size] * 2)
            valid = (boxes[:, 2:] > boxes[:, :2]).all(dim=1)
            boxes, scores = boxes[valid], scores[valid]
            keep = nms(boxes, scores, self.nms)[:300]
            outputs.append({"boxes": boxes[keep], "scores": scores[keep]})
        return outputs
