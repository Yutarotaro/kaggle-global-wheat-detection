"""Select the same validation images across epochs and model runs."""

import random


def select_preview_ids(validation_ids, count=4):
    ids = sorted(validation_ids)
    return sorted(random.Random(42).sample(ids, min(count, len(ids))))
