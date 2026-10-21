"""Optional stylized focus guidance in linear light; zero adds no render nodes.

Use the lens's prepared disparity (including foreground spread/subject hold).
Normalize blur size once across the take, never separately per layer, then use
smoothstep. Grade each colour layer after its lens and before the subject blend,
look and shoulders. A held subject's zero mask therefore remains unchanged.
"""
from pathlib import Path
import math

import numpy as np
from PIL import Image

import round_lens

VERSION = "guide-linear-blur-smoothstep-v1"


def amount(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError("Guide must be a number from 0 to 100.")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError("Guide must be a number from 0 to 100.")
    return value


def grade_inputs(value):
    scale = amount(value) / 100
    gain = 2 ** (-0.7 * scale)
    return {"preset": "None (Custom)", "preset_strength": 0.0,
            "lift_r": 0.0, "lift_g": 0.0, "lift_b": 0.0,
            "gamma_r": 1.0, "gamma_g": 1.0, "gamma_b": 1.0,
            "gain_r": gain, "gain_g": gain, "gain_b": gain,
            "offset_r": 0.0, "offset_g": 0.0, "offset_b": 0.0,
            "contrast": 1.0, "pivot": 0.5, "saturation": 1 - .25 * scale}


def prepare_masks(disparities, output_prefix, blur, focus_d):
    """Save replayable uint8 masks with one shared normalization for every layer."""
    maps = {role: round_lens.blur_fraction(path, blur, focus_d)
            for role, path in disparities.items()}
    for role, mask in maps.items():
        spec = round_lens.describe(blur, mask.shape[1], mask.shape[0])
        # Tiny discrete lens levels can also be identities, despite a nonzero B.
        active = [level["level"] for level in spec["levels"] if not level["identity"]]
        if not active:
            maps[role] = np.zeros_like(mask)
        else:
            maps[role] = np.maximum(mask - (min(active) - 1) / round_lens.LEVELS, 0)
    maximum = max(float(mask.max()) for mask in maps.values())
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {}
    for role, mask in maps.items():
        t = mask / maximum if maximum > 0 else np.zeros_like(mask)
        weight = t * t * (3 - 2 * t)
        path = str(prefix) + f"_{role}.png"
        Image.fromarray(np.round(weight * 255).astype(np.uint8)).save(path)
        paths[role] = path
    return paths, {"version": VERSION, "normalization": "shared maximum of prepared layer blur-size maps",
                   "maximum_blur_fraction": maximum, "falloff": "smoothstep, round to uint8 /255",
                   "maximum_stops_at_100": -0.7, "maximum_saturation_reduction_at_100": .25,
                   "stage": "linear light after each colour lens, before layer composite and look"}


def graph(image_ref, mask_name, value, role):
    value = amount(value)
    if value == 0:
        return {}, list(image_ref)
    if not isinstance(mask_name, str) or not mask_name.endswith(".png"):
        raise ValueError("Guide requires a saved PNG mask.")
    stem = f"guide_{role}"
    load, mask, grade, mix = (stem + suffix for suffix in ("_load", "_mask", "_grade", "_mix"))
    return {
        load: {"class_type": "LoadImage", "inputs": {"image": mask_name}},
        mask: {"class_type": "ImageToMask", "inputs": {"image": [load, 0], "channel": "red"}},
        grade: {"class_type": "RadianceGrade", "_meta": {"title": VERSION, "guide": value},
                "inputs": {"image": list(image_ref), **grade_inputs(value)}},
        mix: {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": list(image_ref), "source": [grade, 0], "mask": [mask, 0],
            "x": 0, "y": 0, "resize_source": False}},
    }, [mix, 0]
