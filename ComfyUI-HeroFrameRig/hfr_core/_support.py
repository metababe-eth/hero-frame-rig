import math
import numpy as np
from numbers import Integral
MIN_FOCUS_RANGE=.01
MAX_SEGMENTS=50
BLUR_STEP_PX=.15
LEVELS=5
VERSION="hfr-shared-round-v1"
def smooth(values, window=9):
    """Moving average with edge padding: a focus puller's hand, not a jittery sensor."""
    v = np.asarray(values, dtype=np.float64)
    pad = window // 2
    return np.convolve(np.pad(v, pad, mode="edge"), np.ones(window) / window, mode="valid")


def smoothstep(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def patch_level(d, x, y, r=4):
    """Trimmed mean of a small patch: sub-level precision (the 10-bit clip depth steps are
    ~2.5% of distance up close, so a median would snap between levels)."""
    p = np.sort(d[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].ravel())
    k = len(p) // 4
    return float(p[k:len(p) - k].mean())


def segments(curve, step):
    segs, start = [], 0
    for i in range(1, len(curve) + 1):
        if i == len(curve) or abs(curve[i] - curve[start]) >= step:
            seg = curve[start:i]
            segs.append({"start": start, "length": i - start, "focus": round(sum(seg) / len(seg), 4)})
            start = i
    return segs


def _anchor(point, width, height, name="Anchor"):
    if not isinstance(point, (tuple, list, np.ndarray)) or len(point) != 2:
        raise ValueError(f"{name} must contain x and y pixel coordinates.")
    if any(isinstance(p, (bool, np.bool_)) or not isinstance(p, Integral) for p in point):
        raise ValueError(f"{name} coordinates must be integers.")
    x, y = map(int, point)
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"{name} ({x}, {y}) is outside the {width}x{height} frame.")
    return x, y

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
    amount = round(min(blur * .99, 50), 2)
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

