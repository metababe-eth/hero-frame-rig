"""Read-only presentation helpers for saved frames; no rendering or Cloud calls.

An optional take['focus_point'] stores normalized image coordinates in UI state
and new recipe exports. Legacy/automatic-focus takes have no invented marker.
"""
from __future__ import annotations

import math
from pathlib import Path


def _take(state, number=None):
    takes = state.get("takes", []) if state else []
    if not takes:
        return None
    if number is None:
        number = state.get("selected_take", takes[-1]["n"])
        if number is None:
            number = takes[-1]["n"]
    for take in takes:
        if take["n"] == number:
            return take
    raise ValueError("Choose one of this frame's saved takes.")


def _number(value, *, decimal=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "?"
    if not math.isfinite(number):
        return "?"
    text = f"{number:.2f}".rstrip("0").rstrip(".")
    return text + ".0" if decimal and "." not in text else text


def take_readout(state, take_number=None):
    """Use this take's saved settings, including its actual adaptive multiplier."""
    take = _take(state, take_number)
    if take is None:
        return "Bring a frame or open the saved study."
    settings = take.get("settings", {})
    mode = settings.get("mode", "Mode not recorded")
    strength = _number(settings.get("strength"), decimal=True)
    if mode.startswith("Cinematic"):
        mode = cinematic_label(settings)
    elif mode.startswith("True to lens"):
        mode = f"True to lens {strength}×"
    else:
        mode = f"{mode} {strength}×"
    guide = f" · guide {_number(settings['guide'])}" if "guide" in settings else ""
    return (f"#{take['n']} · {_number(state.get('lens'))}mm · "
            f"f/{_number(settings.get('f_stop'))} · "
            f"focus {_number(settings.get('focus_m'))} m (est.) · "
            f"{mode} · {settings.get('look', 'Look not recorded')}{guide}")


def cinematic_label(settings):
    """Expose saved effective strength, including sub-unit values caused by caps."""
    strength = _number(settings.get('strength'), decimal=True)
    suffix = 'capped' if settings.get('blur_capped') else 'stylized'
    return f'Cinematic {strength}× ({suffix})'


def take_choices(state):
    """Newest-first metadata labels paired with stable take IDs, never indices."""
    return [(take_readout(state, take["n"]), take["n"])
            for take in reversed(state.get("takes", []))] if state else []


def default_pair(state):
    """Previous saved take versus selected; a singleton compares with itself.

    Previous means the preceding entry in the recorded history, not ID minus one
    and not the next item in the newest-first selector. First take pairs with itself.
    """
    selected = _take(state)
    if selected is None:
        return None, None
    takes = state["takes"]
    index = next(i for i, take in enumerate(takes) if take["n"] == selected["n"])
    return takes[max(0, index - 1)]["n"], selected["n"]


def pair_paths(state, take_a, take_b):
    """Get the two full-resolution saved images without selecting either take."""
    if take_a is None or take_b is None:
        raise ValueError("Choose two saved takes to compare.")
    a, b = _take(state, take_a), _take(state, take_b)
    if a is None or b is None:
        raise ValueError("Open a frame before comparing takes.")
    return a["hero"], b["hero"]


def swap_pair(state, take_a, take_b):
    """Validate and swap only A/B IDs; never change the selected take or controls."""
    pair_paths(state, take_a, take_b)
    return take_b, take_a


def stage_path(state, view="Frame", take_number=None):
    """Expose the actual saved image used by the frame/map views; no I/O."""
    if view not in ("Frame", "Depth", "Subject"):
        raise ValueError("Choose Frame, Depth or Subject.")
    if not state:
        return None
    if view == "Depth":
        return state.get("disparity")
    if view == "Subject":
        return state.get("subject_view") or str(Path(state["dir"]) / "subject_view.png")
    take = _take(state, take_number)
    return take["hero"] if take else state.get("still")


def _dimensions(width, height):
    if (isinstance(width, bool) or isinstance(height, bool)
            or not isinstance(width, int) or not isinstance(height, int)
            or width < 1 or height < 1):
        raise ValueError("Image dimensions must be positive integers.")
    return width, height


def orientation_metadata(width, height):
    """Intrinsic layout hint; responsive CSS still stacks either layout on mobile."""
    width, height = _dimensions(width, height)
    orientation = "portrait" if height > width else "landscape" if width > height else "square"
    return {"width": width, "height": height, "aspect_ratio": width / height,
            "orientation": orientation,
            "layout": "controls-beside" if orientation == "portrait" else "controls-below"}


def normalized_click(x, y, width, height):
    """Clamp image-pixel coordinates, mapping endpoints with x/(width-1).

    For a one-pixel axis, its only pixel maps to zero. This is click-location
    metadata, not a claim that subject-median focus lies exactly at that pixel.
    """
    width, height = _dimensions(width, height)
    x, y = float(x), float(y)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Focus coordinates must be finite.")
    return (min(max(x, 0), width - 1) / max(1, width - 1),
            min(max(y, 0), height - 1) / max(1, height - 1))


def _valid_point(point):
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 <= value <= 1 for value in point):
        return None
    return float(point[0]), float(point[1])


def focus_point(state, take_number=None):
    """Return only the chosen take's recorded point; never infer a legacy marker."""
    take = _take(state, take_number)
    return _valid_point(take.get("focus_point")) if take else None


def focus_point_pixels(point, width, height):
    """Map a recorded point to this actual image rectangle, excluding letterboxing."""
    width, height = _dimensions(width, height)
    point = _valid_point(point)
    return ((point[0] * (width - 1), point[1] * (height - 1))
            if point is not None else None)


def focus_point_status(state, take_number=None):
    """A short honest marker label for recorded points and older takes."""
    take = _take(state, take_number)
    if take is None:
        return ""
    if focus_point(state, take_number) is None:
        return "Point not recorded"
    return f"{_number(take.get('settings', {}).get('focus_m'))} m (est.)"
