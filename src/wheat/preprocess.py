import random

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


def read_image(path, boxes, flip, vertical_flip=0, color_jitter=None):
    with Image.open(path) as handle:
        image = pil_to_tensor(handle.convert("RGB")).float().div_(255)
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if flip and random.random() < flip:
        image = image.flip(-1)
        boxes = boxes.clone()
        boxes[:, [0, 2]] = image.shape[-1] - boxes[:, [2, 0]]
    if vertical_flip and random.random() < vertical_flip:
        image = image.flip(-2)
        boxes = boxes.clone()
        boxes[:, [1, 3]] = image.shape[-2] - boxes[:, [3, 1]]
    if color_jitter is not None:
        image = color_jitter(image)
    return image, {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}
