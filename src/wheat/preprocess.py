import random

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


def read_image(path, boxes, flip):
    with Image.open(path) as handle:
        image = pil_to_tensor(handle.convert("RGB")).float().div_(255)
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if flip and random.random() < flip:
        image = image.flip(-1)
        boxes = boxes.clone()
        boxes[:, [0, 2]] = image.shape[-1] - boxes[:, [2, 0]]
    return image, {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}
