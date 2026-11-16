import random

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, resize


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


def zoom_crop(image, target, rng):
    """Crop at a random field of view, preserving the visible portions of boxes."""
    height, width = image.shape[-2:]
    scale = rng.uniform(0.60, 0.82)
    crop_h, crop_w = max(1, round(height * scale)), max(1, round(width * scale))
    x, y = rng.randrange(width - crop_w + 1), rng.randrange(height - crop_h + 1)
    boxes = target["boxes"]
    clipped = boxes - boxes.new_tensor([x, y, x, y])
    clipped[:, 0::2].clamp_(0, crop_w)
    clipped[:, 1::2].clamp_(0, crop_h)
    size = clipped[:, 2:] - clipped[:, :2]
    original_area = (boxes[:, 2:] - boxes[:, :2]).prod(dim=1)
    keep = (size >= 2).all(dim=1) & (size.prod(dim=1) >= 0.40 * original_area)
    return image[:, y:y + crop_h, x:x + crop_w], {
        "boxes": clipped[keep], "labels": target["labels"][keep],
    }


def zoom_out(image, target, background, rng):
    """Resize the whole view onto a canvas verified to contain no labelled heads."""
    height, width = image.shape[-2:]
    scale = rng.uniform(0.68, 0.85)
    scaled_h, scaled_w = max(1, round(height * scale)), max(1, round(width * scale))
    x, y = rng.randrange(width - scaled_w + 1), rng.randrange(height - scaled_h + 1)
    canvas = resize(background, [height, width], antialias=True).clone()
    canvas[:, y:y + scaled_h, x:x + scaled_w] = resize(image, [scaled_h, scaled_w], antialias=True)
    boxes = target["boxes"]
    transformed = boxes * boxes.new_tensor([scaled_w / width, scaled_h / height] * 2)
    transformed += boxes.new_tensor([x, y] * 2)
    return canvas, {"boxes": transformed, "labels": target["labels"]}


def mosaic(samples):
    """Combine four training crops and clip and translate their boxes."""
    size = samples[0][0].shape[-1]
    xc, yc = (random.randint(size // 4, 3 * size // 4) for _ in range(2))
    canvas = samples[0][0].new_empty((3, size, size))
    boxes, labels = [], []
    quadrants = [(0, 0, xc, yc), (xc, 0, size, yc), (0, yc, xc, size), (xc, yc, size, size)]
    for (image, target), (left, top, right, bottom) in zip(samples, quadrants, strict=True):
        height, width = image.shape[-2:]
        crop_width, crop_height = right - left, bottom - top
        x = size - crop_width if left == 0 else 0
        y = size - crop_height if top == 0 else 0
        image = resize(image, [size, size], antialias=True)
        canvas[:, top:bottom, left:right] = image[:, y:y + crop_height, x:x + crop_width]
        transformed = target["boxes"] * image.new_tensor([size / width, size / height] * 2)
        transformed[:, 0::2] = (transformed[:, 0::2] - x).clamp(0, crop_width) + left
        transformed[:, 1::2] = (transformed[:, 1::2] - y).clamp(0, crop_height) + top
        keep = (transformed[:, 2:] > transformed[:, :2]).all(dim=1)
        boxes.append(transformed[keep])
        labels.append(target["labels"][keep])
    return canvas, {"boxes": torch.cat(boxes), "labels": torch.cat(labels)}
