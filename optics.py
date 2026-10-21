"""Lens physics for Hero Frame Rig: estimated metric depth -> physically shaped blur.

A real lens blurs a point at distance z, focused at z_f, into a circle of confusion:

    CoC radius (px) = f^2 * W / (2 * N * S) * |1/z_f - 1/z|

f = focal length, N = f-stop, S = sensor width (full frame 36 mm), W = image width (px),
distances in metres. Blur is linear in inverse distance, so we feed the depth-of-field
node a DISPARITY map D = z_near / z (0 = infinity, 1 = the nearest distance we encode).
Its linear blur ramp then has the right shape, and

    blur_amount B = f^2 * W / (2 * N * S * z_near)        (CoC radius at |dD| = 1)

Depth: Depth Anything 3 Metric. Its raw output is "canonical" depth for a 300 px focal
length at the resolution the network ran at, so metres = raw * focal_px / 300 with
focal_px = processed_width * f / S (checked: running at 504 vs 1008 px changes raw by
2.04x, as this predicts). The focal length comes from the lens framing used to make the
frame, so distances (and blur) follow the chosen lens.

Honest naming: "physically scaled blur from estimated depth". Depth is an estimate on a
generated image. `strength` > 1 is a stylized exaggeration and must be labelled as such.

Subject layer (hero shots): a BiRefNet matte finds the subject; `subject_layers` splits the
frame into the subject (sharp when focused, including thin things it holds; blurred as one
piece when in front of focus) and a clean background plate with the subject removed, so the
bokeh never picks up the subject's colours.
"""
import numpy as np
from PIL import Image

SENSOR_W = 0.036            # full-frame width, metres
DA3_CANONICAL_FOCAL = 300.0
DA3_RESOLUTION = 1008       # DA3Inference resolution used for stills and clips
MIN_FOCUS_RANGE = 0.01      # node minimum: a dead zone of 1% of B (< 0.2 px at B = 16)
MAX_BLUR = 50.0             # RadianceDepthOfField blur_amount maximum


# ---------- depth I/O ----------

def read_exr_depth(path):
    import OpenEXR
    px = next(iter(OpenEXR.File(str(path)).channels().values())).pixels
    return (px[..., 0] if px.ndim == 3 else px).astype(np.float32)


def read_mask(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def processed_width(w, h, res=DA3_RESOLUTION):
    """Width DA3 ran at (upper_bound_resize: longest side -> res)."""
    return res * w / max(w, h)


def metric_scale(focal_mm, w, h, res=DA3_RESOLUTION):
    """Multiply DA3-Metric raw output by this to get metres for the given lens."""
    return processed_width(w, h, res) * (focal_mm / 1000.0) / SENSOR_W / DA3_CANONICAL_FOCAL


# ---------- disparity ----------

def z_near_for(depths_m, floor=0.3):
    """One z_near (metres) for a still or a whole clip, so focus never pumps."""
    lo = min(float(np.percentile(z, 0.5)) for z in depths_m)
    return max(floor, round(lo * 0.95, 3))


def disparity(z_m, z_near, sky=None):
    """D = z_near / z in [0, 1]; sky probability pushes toward infinity (D = 0)."""
    d = np.clip(z_near / np.maximum(z_m, 1e-3), 0.0, 1.0)
    if sky is not None:
        d = d * (1.0 - np.clip(sky, 0.0, 1.0))
    return d.astype(np.float32)


def save_disparity(d, path):
    Image.fromarray(np.round(np.clip(d, 0, 1) * 255).astype(np.uint8)).save(path)
    return path


def load_disparity(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def sample(d, x, y, r=4):
    """Median disparity in a small patch around a click or a tracked point."""
    return float(np.median(d[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]))


def auto_focus(d, pct=98):
    """Default focus: the nearest thing in the middle of the frame (98th percentile of
    disparity in the central half-width, 20-80% height box). The frame centre alone landed
    on the far street (67 m) instead of the sax player; p98 hit the player within 0.02 on
    three test frames (0.914 vs 0.933, 0.627 vs 0.629, 0.800 vs 0.791)."""
    h, w = d.shape
    box = d[h // 5:4 * h // 5, w // 4:3 * w // 4]
    return float(np.percentile(box, pct))


def focus_metres(focus_d, z_near):
    return z_near / max(focus_d, 1e-4)


# ---------- blur ----------

def blur_scale(focal_mm, f_stop, width_px, z_near, strength=1.0):
    """B: CoC radius in pixels at |D - D_focus| = 1 (node max is 50)."""
    f = focal_mm / 1000.0
    return strength * f * f * width_px / (2.0 * f_stop * SENSOR_W * z_near)


def dof_inputs(blur, focus_d):
    return {"blur_amount": round(min(blur * (1 - MIN_FOCUS_RANGE), MAX_BLUR), 2),
            "focus_distance": round(float(focus_d), 4), "focus_range": MIN_FOCUS_RANGE}


def spread_foreground(d, focus_d, blur, levels=12, share=0.8):
    """Let out-of-focus foreground blur spill over what is behind it.

    The depth-of-field node gathers: each pixel is blurred by its own depth, so a blurred
    foreground object keeps a hard outline against a sharp background. A real lens
    spreads the foreground's blur circle outward. Approximation: grow every nearer-than-
    focus region by `share` of its own blur radius, so pixels just outside it take its
    depth and pick up its blur. Tested on the street frame (lamp in front of a sharp
    wall): the lamp's silhouette softens instead of staying cut out.
    """
    import cv2
    out = d.copy()
    near = d - focus_d
    top = float(near.max())
    if top <= 0 or blur <= 0:
        return out
    for t in np.linspace(top / levels, top, levels):
        r = int(round(share * blur * t))
        if r < 1:
            continue
        mask = (near >= t).astype(np.uint8)
        if not mask.any():
            continue
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        out = np.where(cv2.dilate(mask, k) > 0, np.maximum(out, focus_d + t), out)
    return out


# ---------- subject plane (matte) ----------

def subject_components(matte, thresh=0.5, min_frac=0.003):
    """Connected subjects in the BiRefNet matte; returns (labels, [label ids by area desc])."""
    import cv2
    core = (matte > thresh).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    total = matte.size
    ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_frac * total]
    ids.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    return labels, ids


def subject_at(labels, x, y, r=6):
    """Subject at a caller-clamped click; nearby fallback for background pixels only."""
    label = int(labels[y, x])
    if label:
        return label
    patch = labels[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    vals = patch[patch > 0]
    return int(np.bincount(vals).argmax()) if vals.size else 0


def subject_focus(d, labels, label):
    """Focus disparity for a subject: median over its core."""
    return float(np.median(d[labels == label]))


def default_subject(matte, min_frac=0.01):
    """Matte subjects and the largest one if it covers >= `min_frac` of the frame:
    (labels, ids, label or 0)."""
    labels, ids = subject_components(matte)
    if ids and (labels == ids[0]).mean() >= min_frac:
        return labels, ids, ids[0]
    return labels, ids, 0


def subject_alpha(matte, labels, label, grow=6):
    """Soft matte of the chosen subject(s) only: the matte inside their (grown) components."""
    import cv2
    comp = np.isin(labels, np.atleast_1d(label)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    region = cv2.dilate(comp, k) > 0
    return np.where(region, np.clip(matte, 0, 1), 0.0).astype(np.float32)


def fill_holes(img, hole):
    """Fill `hole` (bool HxW) from its surroundings by push-pull: a mask-weighted image
    pyramid, coarse levels filling what fine levels lack. Smooth, fast, and only ever seen
    through heavy blur."""
    import cv2
    x = img.astype(np.float32)
    w = (~hole).astype(np.float32)
    chan = x.ndim == 3
    xs, ws = [x * (w[..., None] if chan else w)], [w]
    while min(ws[-1].shape) > 8:
        xs.append(cv2.pyrDown(xs[-1]))
        ws.append(cv2.pyrDown(ws[-1]))
    norm = lambda a, b: a / np.maximum(b, 1e-6)[..., None] if chan else a / np.maximum(b, 1e-6)
    known = ~hole
    mean = x[known].mean(axis=0) if known.any() else np.zeros(x.shape[2:], np.float32)
    t = np.clip(ws[-1] * 4, 0, 1)
    t = t[..., None] if chan else t
    filled = t * norm(xs[-1], ws[-1]) + (1 - t) * mean   # far from anything known: the mean
    for a, b in zip(reversed(xs[:-1]), reversed(ws[:-1])):
        up = cv2.resize(filled, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_LINEAR)
        t = np.clip(b * 4, 0, 1)
        t = t[..., None] if chan else t
        filled = t * norm(a, b) + (1 - t) * up
    return np.where(hole[..., None] if chan else hole, filled, x).astype(img.dtype)


def layer_subjects(d, labels, ids, focus_d, blur, max_px=2.0):
    """Matte subjects at or in front of the focus plane: they become the subject layer.
    Subjects behind focus stay in the background and blur with it."""
    tol = max_px / max(blur, 1e-6)
    return [i for i in ids if float(np.median(d[labels == i])) >= focus_d - tol]


def subject_layers(still_rgb, d, matte, labels, layer_ids, focus_d, blur, max_px=2.0,
                   margin=4):
    """Split the frame into a subject layer and a clean background plate.

    Subject layer: the subject's colours pushed outward past its edge (so blurring it never
    mixes in background colours) and a flat disparity per subject: the focus value if the
    subject is within `max_px` of blur of the focus plane (held sharp, including thin
    things it holds), else its own median depth (blurred as one piece, the way an
    out-of-focus foreground melts over a sharp background). Its matte (`alpha`) is blurred
    with the same map, then the layer is composited over the background.

    Background plate: the layer's subjects removed (filled from around them) from the still
    and from the disparity map. Blurring it separately keeps the subject's colours out of
    the bokeh (no glow; Oct 12 hero test).

    Returns (fg_rgb, fg_disparity, alpha, plate_rgb, plate_disparity, n_held).
    """
    import cv2
    alpha = subject_alpha(matte, labels, layer_ids)
    tol = max_px / max(blur, 1e-6)
    flat = np.zeros_like(d, dtype=np.float32)
    core = np.zeros(d.shape, bool)
    held = 0
    for i in layer_ids:
        comp = labels == i
        v = float(np.median(d[comp]))
        if abs(v - focus_d) <= tol:
            v, held = float(focus_d), held + 1
        flat[comp] = v
        core |= comp
    fg_disp = fill_holes(flat, ~core)
    solid = (alpha > 0.9) & core
    fg_rgb = fill_holes(still_rgb, ~(solid if solid.any() else core))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    hole = cv2.dilate((alpha > 0.02).astype(np.uint8), k) > 0
    return (fg_rgb, fg_disp, alpha, fill_holes(still_rgb, hole), fill_holes(d, hole), held)


def background_gap(d, focus_d, pct=10):
    """Disparity between the focus plane and the far background (10th percentile)."""
    return max(float(focus_d) - float(np.percentile(d, pct)), 0.05)


def cinematic_strength(focal_mm, width_px, z_near, gap, target_frac=0.025, max_strength=4.0,
                       ref_stop=1.4):
    """Stylized multiplier for 'Cinematic': aim for a background blur radius of
    `target_frac` of the frame width at f/1.4, never below the lens (1x), at most 4x.

    A fixed 4x asked for ~107 px of background blur on an 85 mm portrait (the node caps
    B at 50, so the "4x" label was wrong) while a 35 mm full-body shot stayed nearly sharp;
    real fast lenses are already strong up close.
    Computed once per frame (with the subject in focus) and kept, so refocusing and
    stopping down still behave like one lens: f/2.8 gives half the blur of f/1.4.
    """
    b1 = blur_scale(focal_mm, ref_stop, width_px, z_near, 1.0)
    return float(np.clip(target_frac * width_px / (b1 * gap), 1.0, max_strength))


# ---------- still helpers ----------

def still_disparity(exr_path, sky_path, focal_mm, out_path):
    """Metric EXR + sky mask -> disparity PNG. Returns (path, z_near, D array)."""
    z_raw = read_exr_depth(exr_path)
    h, w = z_raw.shape
    z = z_raw * metric_scale(focal_mm, w, h)
    sky = read_mask(sky_path) if sky_path else None
    zn = z_near_for([z if sky is None else np.where(sky < 0.5, z, np.inf)])
    d = disparity(z, zn, sky)
    save_disparity(d, out_path)
    return out_path, zn, d
