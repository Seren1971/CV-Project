"""Pure-Python geometry for the variant-B augmentations (no torch, unit-testable).

Boxes are [x1, y1, x2, y2] in pixels.
"""
from __future__ import annotations

import random

MIN_VISIBLE = 0.4      # a box cut by a border crop is kept if >= 40% of its area stays visible
MIN_CROP_SIDE = 0.5    # the crop window keeps at least 50% of the image width / height


def area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def border_crop_window(w: int, h: int, boxes: list, rng: random.Random = random):
    """Crop window whose edge cuts through one animal, so 40-80% of it stays visible.

    Returns (window, new_boxes, kept_indices) or None if no valid crop exists.
    window = (x0, y0, x1, y1); new_boxes are shifted into window coordinates.
    """
    cand = [i for i, b in enumerate(boxes) if b[2] - b[0] >= 8 and b[3] - b[1] >= 8]
    if not cand:
        return None
    i = rng.choice(cand)
    x1, y1, x2, y2 = boxes[i]
    bw, bh = x2 - x1, y2 - y1
    v = rng.uniform(MIN_VISIBLE, 0.8)            # visible share of the chosen box
    sides = ["left", "right", "top", "bottom"]
    rng.shuffle(sides)
    for side in sides:
        win = [0.0, 0.0, float(w), float(h)]
        if side == "left":
            win[0] = x2 - v * bw
        elif side == "right":
            win[2] = x1 + v * bw
        elif side == "top":
            win[1] = y2 - v * bh
        else:
            win[3] = y1 + v * bh
        if win[2] - win[0] < MIN_CROP_SIDE * w or win[3] - win[1] < MIN_CROP_SIDE * h:
            continue
        win = [round(c) for c in win]
        new_boxes, kept = [], []
        for j, b in enumerate(boxes):
            c = [max(b[0], win[0]), max(b[1], win[1]), min(b[2], win[2]), min(b[3], win[3])]
            if c[2] - c[0] < 2 or c[3] - c[1] < 2:
                continue
            if area(c) < MIN_VISIBLE * area(b):
                continue
            new_boxes.append([c[0] - win[0], c[1] - win[1], c[2] - win[0], c[3] - win[1]])
            kept.append(j)
        if i in kept:
            return tuple(win), new_boxes, kept
    return None


def erase_rect(box, rng: random.Random = random, min_frac: float = 0.2, max_frac: float = 0.5):
    """Rectangle inside `box` covering 20-50% of its area (never the whole box)."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return None
    frac = rng.uniform(min_frac, max_frac)
    ratio = rng.uniform(0.5, 2.0)                # width / height of the occluder
    target = frac * bw * bh
    ew = min(bw, (target * ratio) ** 0.5)
    eh = min(bh, target / ew)
    ew = min(bw, target / eh)                    # re-balance if the height was clipped
    ex = x1 + rng.uniform(0, bw - ew)
    ey = y1 + rng.uniform(0, bh - eh)
    return [round(ex), round(ey), round(ex + ew), round(ey + eh)]
