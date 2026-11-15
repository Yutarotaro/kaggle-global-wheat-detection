import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import torch
from torch import nn
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import normalize, resize

EFFDET_PACKAGES = {"effdet": "0.4.1", "timm": "1.0.22"}


def prepare_efficientdet():
    missing = []
    for name, expected in EFFDET_PACKAGES.items():
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = None
        if installed != expected:
            missing.append(f"{name}=={expected}")
    if missing:
        if os.environ.get("ML_LOOP_REMOTE") != "kaggle":
            raise RuntimeError("Install the wheat group with uv sync --group wheat.")
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        *missing], check=True)


class EfficientWheat(nn.Module):
    def __init__(self, config, pretrained):
        super().__init__()
        prepare_efficientdet()
        from effdet import DetBenchPredict, DetBenchTrain, create_model

        self.size = config["image_size"]
        self.confidence = config["confidence"]
        if config["nms"] != 0.5:
            raise ValueError("EfficientDet uses NMS at IoU 0.50.")
        if self.size % 128:
            raise ValueError("EfficientDet image size must be a multiple of 128.")
        network = create_model("tf_efficientdet_d2", num_classes=1, pretrained=pretrained,
                               pretrained_backbone=False, image_size=(self.size, self.size),
                               max_det_per_image=300)
        self.loss_bench = DetBenchTrain(network)
        self.predict_bench = DetBenchPredict(network)

    def forward(self, images, labels=None):
        shapes = [image.shape[-2:] for image in images]
        inputs = torch.stack([
            normalize(resize(image, [self.size, self.size], antialias=True),
                      [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            for image in images
        ])
        if labels is not None:
            boxes = []
            for label, (h, w) in zip(labels, shapes, strict=True):
                gt = label["boxes"][:, [1, 0, 3, 2]]
                boxes.append(gt * gt.new_tensor([self.size / h, self.size / w] * 2))
            info = {
                "bbox": boxes, "cls": [label["labels"] for label in labels],
                "img_scale": inputs.new_ones(len(images)),
                "img_size": inputs.new_full((len(images), 2), self.size),
            }
            return {"loss": self.loss_bench(inputs, info)["loss"]}

        detections = self.predict_bench(inputs)
        outputs = []
        for prediction, (h, w) in zip(detections, shapes, strict=True):
            if not torch.isfinite(prediction).all():
                raise ValueError("Non-finite EfficientDet prediction.")
            scores = prediction[:, 4]
            boxes = prediction[:, :4].clamp(0, self.size)
            boxes = boxes * boxes.new_tensor([w / self.size, h / self.size] * 2)
            keep = (scores >= self.confidence) & (boxes[:, 2:] > boxes[:, :2]).all(dim=1)
            outputs.append({"boxes": boxes[keep], "scores": scores[keep]})
        return outputs


def create_model(config, pretrained=True):
    detector = config.get("detector", "fasterrcnn-mobilenet-v3")
    if detector == "yolox-s":
        from yolox_model import YoloWheat

        return YoloWheat(config, pretrained)
    if detector == "efficientdet-d2":
        return EfficientWheat(config, pretrained)
    if detector != "fasterrcnn-mobilenet-v3":
        raise ValueError(f"Unknown detector: {detector}")
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights, weights_backbone=None, min_size=config["image_size"],
        max_size=config["image_size"], box_score_thresh=config["confidence"],
        box_nms_thresh=config["nms"], box_detections_per_img=300,
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    return model
