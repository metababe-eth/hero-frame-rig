"""Portable still lens: round Gaussian with the legacy blur footprint.

Radiance 2.3.3 caps Gaussian support at 31 pixels even for sigma 50. Its
nearly flat square kernel creates square highlights. This graph uses open core
ImageBlur with ~3-sigma support. Per-level variance matches the old kernel so
this correction does not silently expand the established blur footprint.
This is a soft Gaussian approximation, not a physical disc/octagonal aperture.
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image

import optics

VERSION = "core-round-gaussian-legacy-footprint-v1"
GRAPH_VERSION = VERSION
LEVELS = 5
# Explicit opt-in for unlayered uploaded photographs. The V1 API and graph are
# deliberately unchanged, including their legacy foreground-spread input maps.
PHOTO_VERSION = "core-round-gaussian-photo-v2"
PHOTO_HIGHLIGHT_RECOVERY = 0.0


def _number(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number.")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _parameters(blur, focus):
    blur = _number(blur, "Blur", 0, optics.MAX_BLUR)
    focus = _number(focus, "Focus disparity", 0, 1)
    return optics.dof_inputs(blur, focus)


def legacy_sigma(sigma):
    """Standard deviation of the actual normalized Radiance 2.3.3 1D kernel."""
    if sigma < 0.1:
        return 0.0
    size = max(3, min(int(sigma * 6) | 1, 31))
    x = np.arange(size, dtype=np.float64) - size // 2
    kernel = np.exp(-(x * x) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    return float(np.sqrt((kernel * x * x).sum()))


def describe(blur, width, height):
    if any(type(v) is not int or v < 2 for v in (width, height)):
        raise ValueError("Round lens needs image dimensions of at least 2 pixels.")
    amount = _parameters(blur, 0)["blur_amount"]
    sigmas = [legacy_sigma(amount * level / LEVELS) for level in range(1, LEVELS + 1)]
    scale = max(1, math.ceil(max(sigmas) / 8))
    sw, sh = max(2, round(width / scale)), max(2, round(height / scale))
    actual_scale = (width / sw + height / sh) / 2
    levels = []
    for level, sigma in enumerate(sigmas, 1):
        reduced = sigma / actual_scale
        radius = min(31, sw - 1, sh - 1, max(1, math.ceil(3 * reduced)))
        # Vanishing discrete kernels are identity; core sigma has a 0.1 minimum.
        identity = reduced < 0.1 or reduced / radius < 0.1
        levels.append({"level": level, "legacy_nominal_sigma_px": amount * level / LEVELS,
                       "matched_sigma_px": sigma, "radius": radius,
                       "core_sigma": min(10.0, max(0.1, reduced / radius)), "identity": identity})
    return {"version": VERSION, "shape": "soft round Gaussian (legacy footprint)",
            "width": width, "height": height, "downsample": [sw, sh],
            "node_blur_amount": amount, "levels": levels,
            "weights": "five saved uint8 masks; round to nearest /255; exact zero remains zero",
            "limits": "Gather blur and legacy footprint calibration; not physical aperture bokeh."}


def blur_fraction(disparity_path, blur, focus_d):
    """The lens's continuous blur-size map, including its exact zero-focus band."""
    params = _parameters(blur, focus_d)
    d = optics.load_disparity(disparity_path)
    mask = np.clip((np.abs(d - params["focus_distance"]) - params["focus_range"]) /
                   (1 - params["focus_range"] + 1e-6), 0, 1)
    return np.zeros_like(mask) if params["blur_amount"] < 0.1 else mask


def prepare_weights(disparity_path, output_prefix, blur, focus_d):
    """Return five saved mask paths, in blend order, using the exact saved disparity."""
    mask = blur_fraction(disparity_path, blur, focus_d)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for level in range(1, LEVELS + 1):
        weight = np.clip((mask - (level - 1) / LEVELS) * LEVELS, 0, 1)
        path = str(prefix) + f"_{level}.png"
        Image.fromarray(np.round(weight * 255).astype(np.uint8)).save(path)
        paths.append(path)
    return paths


def describe_photo(blur, width, height):
    """V2 photograph policy; its blur kernels and finish remain the V1 ones.

    The caller must disable synthetic HDR highlight recovery in the linear-light
    conversion. This is a versioned tone change, including at zero blur, rather
    than a claim to preserve the V1 treatment of nearly clipped white surfaces.
    """
    spec = describe(blur, width, height)
    spec.update({
        "version": PHOTO_VERSION,
        "input_disparity": "raw saved disparity; no legacy foreground dilation",
        "foreground_weights": "Gaussian spill per blur level; 3-sigma finite support; uint8",
        "highlight_recovery": PHOTO_HIGHLIGHT_RECOVERY,
        "scope": "unlayered uploaded still photographs only",
        "limits": "Soft gather approximation; depth errors, thin-wire halos and hidden background remain unresolved.",
    })
    return spec


def photo_weight_arrays(disparity, blur, focus_d):
    """Five uint8 weights with smoothly decaying foreground spill.

    Input is the *original* normalized disparity, not spread_foreground output.
    The old twelve binary dilations extend blur with hard depth steps. Here each
    blur level spills only its foreground weight with that kernel's actual
    standard deviation. Subtracting the local weight keeps constant-depth
    interiors unchanged; doubling the outward half makes the edge continuous.
    No colour, source pixel, disparity or input array is modified.
    """
    import cv2
    d = np.asarray(disparity, dtype=np.float32)
    if d.ndim != 2 or min(d.shape) < 2 or not np.isfinite(d).all():
        raise ValueError("Photo lens needs a finite 2D disparity of at least 2 pixels per side.")
    if d.min() < 0 or d.max() > 1:
        raise ValueError("Photo disparity must be normalized to 0–1, bright = near.")
    params = _parameters(blur, focus_d)
    if params["blur_amount"] < 0.1:
        return [np.zeros(d.shape, dtype=np.uint8) for _ in range(LEVELS)]
    spec = describe_photo(blur, d.shape[1], d.shape[0])
    fraction = np.clip((np.abs(d - params["focus_distance"]) - params["focus_range"]) /
                       (1 - params["focus_range"] + 1e-6), 0, 1)
    near = d > params["focus_distance"] + params["focus_range"]
    weights = []
    for level in spec["levels"]:
        weight = np.clip((fraction - (level["level"] - 1) / LEVELS) * LEVELS, 0, 1)
        sigma = level["matched_sigma_px"]
        if not level["identity"]:
            foreground = np.where(near, weight, 0).astype(np.float32)
            radius = min(d.shape[0] - 1, d.shape[1] - 1, math.ceil(3 * sigma))
            smooth = cv2.GaussianBlur(foreground, (2 * radius + 1, 2 * radius + 1),
                                      sigma, borderType=cv2.BORDER_REFLECT_101)
            spill = np.clip(2 * smooth - foreground, 0, 1)
            weight = np.maximum(weight, spill)
        weights.append(np.round(weight * 255).astype(np.uint8))
    return weights


def prepare_photo_weights(disparity_path, output_prefix, blur, focus_d):
    """Persist V2 weights from a raw saved disparity PNG; return the five paths."""
    arrays = photo_weight_arrays(optics.load_disparity(disparity_path), blur, focus_d)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for level, array in enumerate(arrays, 1):
        path = str(prefix) + f"_{level}.png"
        Image.fromarray(array).save(path)
        paths.append(path)
    return paths


def photo_graph(image_ref, width, height, blur, mask_names, id_prefix="round"):
    """Core-only V2 branch. Its input must be unexpanded linear-light RGB.

    The supplied masks must be prepare_photo_weights output. They are saved with
    the recipe, so execution never depends on recomputing an unknown algorithm.
    """
    nodes, result = graph(image_ref, width, height, blur, mask_names, id_prefix)
    for node in nodes.values():
        if node.get("_meta", {}).get("title", "").startswith(VERSION):
            node["_meta"]["title"] = node["_meta"]["title"].replace(VERSION, PHOTO_VERSION, 1)
    return nodes, result


def graph(image_ref, width, height, blur, mask_names, id_prefix="round"):
    """Return (API nodes, output reference). HDR values remain unclamped in blending.

    The caller persists/uploads every mask, and can reuse the subject's masks for
    its colour and matte passes. IDs are prefixed to allow independent branches.
    """
    if len(mask_names) != LEVELS or not all(isinstance(n, str) and n for n in mask_names):
        raise ValueError("Round lens requires exactly five saved mask filenames.")
    if not isinstance(id_prefix, str) or not id_prefix:
        raise ValueError("Round lens requires a nonempty node ID prefix.")
    spec = describe(blur, width, height)
    if spec["node_blur_amount"] < 0.1:
        return {}, list(image_ref)
    sw, sh = spec["downsample"]
    low = f"{id_prefix}_scale"
    nodes = {low: {"class_type": "ImageScale", "_meta": {"title": VERSION + " / area scale"},
                   "inputs": {"image": list(image_ref), "upscale_method": "area", "width": sw,
                              "height": sh, "crop": "disabled"}}}
    previous = list(image_ref)
    for level, name in zip(spec["levels"], mask_names):
        if level["identity"]:
            continue
        tag = f"{id_prefix}_{level['level']}"
        load, mask, blur_id, up, mix = (tag + suffix for suffix in ("_load", "_mask", "_blur", "_up", "_mix"))
        nodes[load] = {"class_type": "LoadImage", "inputs": {"image": name}}
        nodes[mask] = {"class_type": "ImageToMask", "inputs": {"image": [load, 0], "channel": "red"}}
        nodes[blur_id] = {"class_type": "ImageBlur", "inputs": {
            "image": [low, 0], "blur_radius": level["radius"], "sigma": level["core_sigma"]}}
        nodes[up] = {"class_type": "ImageScale", "inputs": {
            "image": [blur_id, 0], "upscale_method": "bilinear", "width": width, "height": height, "crop": "disabled"}}
        nodes[mix] = {"class_type": "ImageCompositeMasked", "inputs": {
            "destination": previous, "source": [up, 0], "x": 0, "y": 0,
            "resize_source": False, "mask": [mask, 0]}}
        previous = [mix, 0]
    return nodes, previous
