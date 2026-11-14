from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(config, pretrained=True):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights, weights_backbone=None, min_size=config["image_size"],
        max_size=config["image_size"], box_score_thresh=config["confidence"],
        box_nms_thresh=config["nms"], box_detections_per_img=300,
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 2)
    return model
