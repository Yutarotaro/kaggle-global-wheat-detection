import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

import torch
from torch import nn
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_convert
from torchvision.transforms.functional import normalize, resize

EFFDET_PACKAGES = {"effdet": "0.4.1", "timm": "1.0.22"}
RTDETR_PACKAGES = {"scipy": "1.16.3", "transformers": "4.57.6"}
EFFICIENTDET_MODELS = {"efficientdet-d2": "tf_efficientdet_d2", "efficientdet-d4": "tf_efficientdet_d4"}


def prepare_packages(packages, group):
    missing = []
    for name, expected in packages.items():
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = None
        if installed != expected:
            missing.append(f"{name}=={expected}")
    if missing:
        if os.environ.get("ML_LOOP_REMOTE") != "kaggle":
            raise RuntimeError(f"Install the {group} group with uv sync --group wheat --group {group}.")
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        *missing], check=True)


class EfficientWheat(nn.Module):
    def __init__(self, config, pretrained):
        super().__init__()
        prepare_packages(EFFDET_PACKAGES, "wheat")
        from effdet import DetBenchPredict, DetBenchTrain, create_model

        self.size = config["image_size"]
        self.confidence = config["confidence"]
        self.freeze_backbone_bn = config.get("freeze_backbone_bn", False)
        if config["nms"] != 0.5:
            raise ValueError("EfficientDet uses NMS at IoU 0.50.")
        if self.size % 128:
            raise ValueError("EfficientDet image size must be a multiple of 128.")
        network = create_model(EFFICIENTDET_MODELS[config["detector"]], num_classes=1, pretrained=pretrained,
                               pretrained_backbone=False, image_size=(self.size, self.size),
                               max_det_per_image=300)
        self.loss_bench = DetBenchTrain(network)
        self.predict_bench = DetBenchPredict(network)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_backbone_bn:
            for layer in self.loss_bench.model.backbone.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.training = False
        return self

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


class RTDetrWheat(nn.Module):
    def __init__(self, config, pretrained):
        super().__init__()
        prepare_packages(RTDETR_PACKAGES, "detr")
        from transformers import RTDetrV2Config, RTDetrV2ForObjectDetection

        self.size = config["image_size"]
        self.confidence = config["confidence"]
        if self.size % 32:
            raise ValueError("RT-DETR requires a multiple-of-32 input size.")
        conditions = RTDetrV2Config.from_dict(config["detr_architecture"])
        conditions.num_labels = 1
        conditions.id2label = {0: "wheat_head"}
        conditions.label2id = {"wheat_head": 0}
        if pretrained:
            self.detector = RTDetrV2ForObjectDetection.from_pretrained(
                config["pretrained_model"], revision=config["pretrained_revision"], config=conditions,
                use_safetensors=True, ignore_mismatched_sizes=True,
            )
        else:
            self.detector = RTDetrV2ForObjectDetection(conditions)

    def parameter_groups(self, backbone_lr):
        backbone = list(self.detector.model.backbone.parameters())
        ids = {id(param) for param in backbone}
        return [{"params": backbone, "lr": backbone_lr},
                {"params": [param for param in self.parameters() if id(param) not in ids]}]

    def forward(self, images, labels=None):
        shapes = [image.shape[-2:] for image in images]
        pixels = torch.stack([resize(image, [self.size, self.size], antialias=True) for image in images])
        targets = None
        if labels is not None:
            targets = []
            for label, (height, width) in zip(labels, shapes, strict=True):
                boxes = label["boxes"]
                scaled = boxes / boxes.new_tensor([width, height, width, height])
                targets.append({"boxes": box_convert(scaled, "xyxy", "cxcywh"),
                                "class_labels": label["labels"] - 1})
        outputs = self.detector(pixel_values=pixels, labels=targets)
        if labels is not None:
            return {"loss": outputs.loss}
        detections = []
        for scores, boxes, (height, width) in zip(
            outputs.logits[..., 0].sigmoid(), outputs.pred_boxes, shapes, strict=True
        ):
            boxes = box_convert(boxes, "cxcywh", "xyxy").clamp(0, 1)
            boxes = boxes * boxes.new_tensor([width, height, width, height])
            if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
                raise ValueError("Non-finite RT-DETR detection.")
            keep = (scores >= self.confidence) & (boxes[:, 2:] > boxes[:, :2]).all(dim=1)
            detections.append({"boxes": boxes[keep], "scores": scores[keep]})
        return detections


def create_model(config, pretrained=True):
    detector = config.get("detector", "fasterrcnn-mobilenet-v3")
    if detector in EFFICIENTDET_MODELS:
        return EfficientWheat(config, pretrained)
    if detector == "rtdetr-v2-r18":
        return RTDetrWheat(config, pretrained)
    if detector == "fasterrcnn-mobilenet-v3":
        build, weights = fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    elif detector == "fasterrcnn-resnet50-v2":
        build, weights = fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    else:
        raise ValueError(f"Unknown detector: {detector}")
    model = build(
        weights=weights if pretrained else None, weights_backbone=None, min_size=config["image_size"],
        max_size=config["image_size"], box_score_thresh=config["confidence"],
        box_nms_thresh=config["nms"], box_detections_per_img=300,
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    return model
