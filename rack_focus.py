"""Rack focus after animation: plan per-frame focus on an animated clip (lens v2).

Inputs come from rig_workflow.clip_depth(): the clip's Depth Anything 3 Metric depth
(10-bit AVIF, raw units / 32) and its sky mask. All planning runs locally:

  1. Track two points (focus targets) through the sharp clip by template matching.
  2. Convert depth to metres for the chosen lens, then to one disparity map per frame,
     D = z_near / z with a single z_near for the whole clip (no pumping). Blur on a
     disparity map is physically scaled: CoC radius = B * |D - D_focus| (see optics.py).
  3. Read each target's disparity along its track (trimmed-mean patch, 9-frame moving
     average) and build the focus curve: hold A, smoothstep A -> B, then follow B.
  4. Per frame, let out-of-focus foreground blur spill over what is behind it
     (optics.spread_foreground), and write the result as the depth video to upload.
  5. Merge frames whose focus moves by less than ~0.15 px of blur into segments, so the
     Comfy graph needs one depth-of-field node per segment (<= 50).

rig_workflow.rack_focus_video() turns the segments into a ComfyUI graph.

Usage:
  python3 rack_focus.py SHARP.mp4 DEPTH.avif SKY.avif OUTDIR AX,AY BX,BY RACK_START RACK_END
          [--lens 35] [--fstop 1.4] [--strength 4.5]
Street test (Oct 12): A = lamp post (182,560), B = musician (686,477), rack 20-48.
"""
import argparse
from contextlib import contextmanager
import json
import math
from numbers import Integral
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import optics  # noqa: E402

DEPTH_SCALE = 32.0          # clip_depth() saves raw / 32, clamped, as 10-bit video range
MAX_SEGMENTS = 50
BLUR_STEP_PX = 0.15         # merge frames whose blur radius differs by less than this


def read_frames(clip, mode):
    with tempfile.TemporaryDirectory(prefix="hero-sharp-") as directory:
        d = Path(directory)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(clip), str(d / "f_%04d.png")], check=True)
        frames = []
        for f in sorted(d.glob("f_*.png")):
            with Image.open(f) as image:
                frames.append(np.array(image.convert(mode)))
        if not frames:
            raise ValueError("Sharp clip decoded to no frames.")
        return frames


@contextmanager
def _as_sequence(avif):
    """Animated AVIF from SaveImageAdvanced: ffmpeg reads only its still item unless the
    ftyp brand says 'avis', so patch a copy and read the frame track."""
    b = bytearray(Path(avif).read_bytes())
    if b[8:12] == b"avif":
        b[8:12] = b"avis"
    with tempfile.TemporaryDirectory(prefix="hero-avif-") as directory:
        out = Path(directory) / "seq.mp4"
        out.write_bytes(b)
        yield out


def _read_y(path, bits, expected_frames=None):
    if bits not in (8, 10):
        raise ValueError("Depth video bit depth must be 8 or 10.")
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v",
                            "-count_frames", "-show_entries", "stream=index,width,height,nb_read_frames",
                            "-of", "json", str(path)],
                           check=True, capture_output=True, text=True).stdout)["streams"]
    candidates = [s for s in probe if int(s.get("nb_read_frames", 0)) > 1]
    if expected_frames is not None:
        candidates = [s for s in candidates if int(s["nb_read_frames"]) == expected_frames]
    if not candidates and expected_frames is None and len(probe) == 1:
        candidates = probe
    if len(candidates) != 1:
        raise ValueError("Depth container must identify one matching sequence track.")
    selected = candidates[0]
    w, h = int(selected["width"]), int(selected["height"])
    if w <= 0 or h <= 0:
        raise ValueError("Depth video dimensions must be positive.")
    fmt, dt, bpp = ("yuv420p10le", "<u2", 2) if bits == 10 else ("yuv420p", "u1", 1)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:{selected['index']}", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", fmt, "-"],
                         check=True, capture_output=True).stdout
    fsz = (w * h + 2 * ((w + 1) // 2) * ((h + 1) // 2)) * bpp
    if not raw or len(raw) % fsz:
        raise ValueError("Depth video decoded to no frames or an incomplete raw frame.")
    return np.stack([np.frombuffer(raw[i * fsz:i * fsz + w * h * bpp], dtype=dt).reshape(h, w)
                     for i in range(len(raw) // fsz)]).astype(np.float32)


def read_clip_depth(depth_avif, sky_avif=None, depth_scale=DEPTH_SCALE, expected_frames=None):
    """Raw DA3-Metric depth per frame (video range: 64-940 for 10-bit) and sky probability."""
    with _as_sequence(depth_avif) as depth_sequence:
        y = _read_y(depth_sequence, 10, expected_frames)
    _positive("Depth packing scale", depth_scale)
    z_raw = depth_scale * np.clip((y - 64.0) / 876.0, 0, 1)
    sky = None
    if sky_avif:
        with _as_sequence(sky_avif) as sky_sequence:
            s = _read_y(sky_sequence, 8, expected_frames)
        sky = np.clip((s - 16.0) / 219.0, 0, 1)
    return z_raw, sky


def _positive(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a positive finite number.")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")


def _frame_shape(frames, name):
    if len(frames) == 0:
        raise ValueError(f"{name} must contain at least one frame.")
    shape = np.asarray(frames[0]).shape
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError(f"{name} frames must be nonempty two-dimensional arrays.")
    for frame in frames:
        if np.asarray(frame).shape != shape:
            raise ValueError(f"{name} frames must have matching dimensions.")
        if not np.isfinite(frame).all():
            raise ValueError(f"{name} frames must contain finite values.")
    return shape


def _anchor(point, width, height, name="Anchor"):
    if not isinstance(point, (tuple, list, np.ndarray)) or len(point) != 2:
        raise ValueError(f"{name} must contain x and y pixel coordinates.")
    if any(isinstance(p, (bool, np.bool_)) or not isinstance(p, Integral) for p in point):
        raise ValueError(f"{name} coordinates must be integers.")
    x, y = map(int, point)
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"{name} ({x}, {y}) is outside the {width}x{height} frame.")
    return x, y


def _rack_bounds(rack_start, rack_end, frames):
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
           for v in (rack_start, rack_end)):
        raise ValueError("Rack bounds must be integer frame indices.")
    if not 0 <= rack_start < rack_end < frames:
        raise ValueError(f"Rack bounds must satisfy 0 <= start < end < {frames} frames.")


def track(frames_gray, start_xy, half=24, search=28):
    """Follow a patch frame to frame (sum of squared differences, recentred each frame)."""
    h, w = _frame_shape(frames_gray, "Tracking clip")
    for name, value, minimum in (("half", half, 1), ("search", search, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"Tracking {name} must be an integer >= {minimum}.")
    x, y = _anchor(start_xy, w, h)
    pts = [(x, y)]
    # Keep the clicked coordinate, adapting a fixed-size patch at frame edges.
    # Reusing these offsets keeps every candidate the same shape as the template.
    left, right = min(half, x), min(half, w - x)
    top, bottom = min(half, y), min(half, h - y)
    tmpl = frames_gray[0][y - top:y + bottom, x - left:x + right].astype(np.float32)
    for g in frames_gray[1:]:
        g = g.astype(np.float32)
        best = ((g[y - top:y + bottom, x - left:x + right] - tmpl) ** 2).mean()
        bx, by = x, y
        for dy in range(-search, search + 1, 2):
            for dx in range(-search, search + 1, 2):
                cx, cy = x + dx, y + dy
                if cx - left < 0 or cy - top < 0 or cx + right > w or cy + bottom > h:
                    continue
                ssd = ((g[cy - top:cy + bottom, cx - left:cx + right] - tmpl) ** 2).mean()
                if ssd < best:
                    best, bx, by = ssd, cx, cy
        x, y = bx, by
        pts.append((x, y))
        tmpl = g[y - top:y + bottom, x - left:x + right]
    return pts


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


def focus_curve(disp, track_a, track_b, rack_start, rack_end):
    h, w = _frame_shape(disp, "Disparity clip")
    _rack_bounds(rack_start, rack_end, len(disp))
    if len(track_a) < rack_end + 1 or len(track_a) > len(disp):
        raise ValueError("A track must cover frame zero through the rack end.")
    if len(track_b) != len(disp):
        raise ValueError("B track must cover every disparity frame.")
    for name, points in (("A track", track_a), ("B track", track_b)):
        for point in points:
            _anchor(point, w, h, name)
    # A has no role after the rack. Its smoothing tail intentionally repeats its
    # last needed sample, rather than requiring an unrelated full-length track.
    fa = smooth([patch_level(disp[i], *track_a[i]) for i in range(rack_end + 1)])
    fb = smooth([patch_level(disp[i], *track_b[i]) for i in range(len(disp))])
    curve = []
    for i in range(len(disp)):
        if i <= rack_start:
            f = fa[i]
        elif i >= rack_end:
            f = fb[i]
        else:
            t = smoothstep((i - rack_start) / (rack_end - rack_start))
            f = (1 - t) * fa[i] + t * fb[i]
        curve.append(round(float(f), 4))
    return curve


def segments(curve, step):
    segs, start = [], 0
    for i in range(1, len(curve) + 1):
        if i == len(curve) or abs(curve[i] - curve[start]) >= step:
            seg = curve[start:i]
            segs.append({"start": start, "length": i - start, "focus": round(sum(seg) / len(seg), 4)})
            start = i
    return segs


def write_depth_video(frames01, out, fps=25):
    _positive("FPS", fps)
    h, w = _frame_shape(frames01, "Disparity clip")
    if w % 2 or h % 2:
        raise ValueError("Disparity video dimensions must be even for yuv420p encoding.")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hero-depth-", dir=out.parent) as directory:
        d = Path(directory)
        for i, f in enumerate(frames01):
            optics.save_disparity(f, d / f"d_{i:04d}.png")
        encoded = d / "disparity.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i",
                        str(d / "d_%04d.png"), "-c:v", "libx264", "-crf", "1", "-pix_fmt", "yuv420p",
                        str(encoded)], check=True)
        encoded.replace(out)


def plan(sharp, depth_avif, sky_avif, outdir, a_xy, b_xy, rack_start, rack_end,
         lens_mm=35, f_stop=1.4, strength=4.5, fps=25):
    for name, value in (("FPS", fps), ("Lens", lens_mm), ("F-stop", f_stop), ("Strength", strength)):
        _positive(name, value)
    gray = read_frames(sharp, "L")
    z_raw, sky = read_clip_depth(depth_avif, sky_avif)
    h, w = _frame_shape(gray, "Sharp clip")
    if _frame_shape(z_raw, "Depth clip") != (h, w) or len(z_raw) != len(gray):
        raise ValueError("Sharp and depth clips must have matching frame counts and dimensions.")
    if sky is not None and (_frame_shape(sky, "Sky clip") != (h, w) or len(sky) != len(gray)):
        raise ValueError("Sky and sharp clips must have matching frame counts and dimensions.")
    if w % 2 or h % 2:
        raise ValueError("Clip dimensions must be even for yuv420p disparity encoding.")
    z_raw = np.asarray(z_raw)
    if (z_raw < 0).any():
        raise ValueError("Depth values must be nonnegative.")
    if sky is not None and ((np.asarray(sky) < 0).any() or (np.asarray(sky) > 1).any()):
        raise ValueError("Sky probabilities must be between zero and one.")
    n = len(gray)
    _rack_bounds(rack_start, rack_end, n)
    a_xy, b_xy = _anchor(a_xy, w, h, "A anchor"), _anchor(b_xy, w, h, "B anchor")
    z = z_raw * optics.metric_scale(lens_mm, w, h)
    ground = [np.where(sky[i] < 0.5, z[i], np.inf) if sky is not None else z[i] for i in range(n)]
    z_near = optics.z_near_for(ground)
    if not math.isfinite(z_near) or z_near <= 0:
        raise ValueError("The clip needs finite non-sky depth to establish its focus scale.")
    disp = [optics.disparity(z[i], z_near, None if sky is None else sky[i]) for i in range(n)]
    blur = min(optics.blur_scale(lens_mm, f_stop, w, z_near, strength), optics.MAX_BLUR)
    ta = track(gray[:rack_end + 1], a_xy)       # A only needs tracking until the rack ends
    tb = track(gray, b_xy)
    curve = focus_curve(disp, ta, tb, rack_start, rack_end)
    step = max(BLUR_STEP_PX / blur, 0.002)
    segs = segments(curve, step)
    while len(segs) > MAX_SEGMENTS:
        step *= 1.25
        segs = segments(curve, step)
    # Spread foreground per frame with the focus that frame is rendered at.
    seg_focus = np.empty(n)
    for s in segs:
        seg_focus[s["start"]:s["start"] + s["length"]] = s["focus"]
    spread = [optics.spread_foreground(disp[i], seg_focus[i], blur) for i in range(n)]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_depth_video(spread, outdir / "disparity.mp4", fps)
    info = {"frames": n, "width": w, "height": h, "fps": fps,
            "lens_mm": lens_mm, "f_stop": f_stop, "strength": strength,
            "z_near_m": z_near, "blur": round(blur, 2), "segment_step": round(step, 4),
            "rack": [rack_start, rack_end], "track_a": ta, "track_b": tb,
            "focus_curve": curve, "focus_m": [round(optics.focus_metres(c, z_near), 2) for c in curve],
            "segments": segs}
    (outdir / "rack_plan.json").write_text(json.dumps(info))
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sharp"); ap.add_argument("depth"); ap.add_argument("sky"); ap.add_argument("outdir")
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("rack_start", type=int); ap.add_argument("rack_end", type=int)
    ap.add_argument("--lens", type=int, default=35)
    ap.add_argument("--fstop", type=float, default=1.4)
    ap.add_argument("--strength", type=float, default=4.5)
    a = ap.parse_args()
    p = plan(a.sharp, a.depth, a.sky, a.outdir, tuple(int(v) for v in a.a.split(",")),
             tuple(int(v) for v in a.b.split(",")), a.rack_start, a.rack_end, a.lens, a.fstop, a.strength)
    c = p["focus_curve"]
    print(f"frames {p['frames']}  z_near {p['z_near_m']} m  blur B {p['blur']} px  segments {len(p['segments'])}")
    print(f"focus A@0 {c[0]} ({p['focus_m'][0]} m)  mid {c[(a.rack_start + a.rack_end) // 2]}  "
          f"B@end {c[-1]} ({p['focus_m'][-1]} m)")
