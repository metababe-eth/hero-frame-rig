"""Local, immutable A/B exports of already developed takes. No Cloud dependency."""

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid
import zipfile

from PIL import Image, ImageDraw, ImageFont


FPS = 30
HOLD_FRAMES = 36
FADE_FRAMES = 6
FRAME_COUNT = 120
BACKGROUND = (14, 14, 16)
TEXT = (236, 232, 225)
MUTED = (154, 149, 140)


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number.") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return number


def _take(state, number):
    if isinstance(number, bool) or number is None:
        raise ValueError("Choose one of this frame's saved takes for A and B.")
    matches = [take for take in state.get("takes", []) if take["n"] == number]
    if len(matches) != 1:
        raise ValueError("Choose one of this frame's saved takes for A and B.")
    return matches[0]


def take_label(state, take):
    """Build the comparison label from saved settings, never a free-text caption."""
    settings = take["settings"]
    lens = _positive(state["lens"], "Lens")
    aperture = _positive(settings["f_stop"], "Aperture")
    focus = _positive(settings["focus_m"], "Estimated focus")
    strength = _positive(settings["strength"], "Depth-of-field strength")
    mode = settings["mode"]
    if not isinstance(mode, str):
        raise ValueError("Saved depth-of-field mode is missing.")
    if mode.startswith("Cinematic"):
        mode_label = f"Cinematic {strength:.1f}x (stylized)"
    elif mode.startswith("True to lens"):
        mode_label = f"True to lens {strength:.1f}x"
    else:
        raise ValueError("Saved depth-of-field mode is not recognized.")
    look = settings["look"]
    if look not in ("digital", "film", "cinestill"):
        raise ValueError("Saved look is not recognized.")
    layer = settings.get("subject_layer_enabled")
    # Do not silently turn legacy, unknown settings into an asserted off state.
    layer_label = "unknown" if layer is None else ("on" if layer else "off")
    return (f"Take #{take['n']} | {lens:g}mm | f/{aperture:g} | "
            f"focus {focus:.1f} m (est.) | {mode_label} | {look} | "
            f"subject layer {layer_label}")


def _read_image(path, expected_size):
    path = Path(path)
    data = path.read_bytes()
    with Image.open(BytesIO(data)) as image:
        if image.size != expected_size:
            raise ValueError("Both takes must match this frame's saved dimensions.")
        if image.mode != "RGB":
            raise ValueError("Pair export needs saved RGB take images.")
        image.load()
        result = image.copy()
    return result, {"file": path.name, "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest()}


def _font(size):
    # Pillow's bundled font adds no downloaded/system-font dependency.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow 10.0 predates the scalable bundled font.
        return ImageFont.load_default()


def _wrap(draw, text, font, width):
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=font) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _side_by_side(a, b, labels, path):
    width, height = a.size
    pad = max(12, round(width * .025))
    # A small input still gets a readable label outside the original image rectangle.
    panel_width = max(240, width)
    font_size = max(14, min(32, round(panel_width * .027)))
    title_font, body_font = _font(round(font_size * 1.4)), _font(font_size)
    line_height = round(font_size * 1.45)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    wrapped = [_wrap(measure, label, body_font, panel_width) for label in labels]
    header = pad * 2 + line_height * (2 + max(map(len, wrapped)))
    result = Image.new("RGB", (panel_width * 2 + pad * 3, header + height + pad), BACKGROUND)
    draw = ImageDraw.Draw(result)
    boxes = []
    for index, (image, lines) in enumerate(zip((a, b), wrapped)):
        x = pad + index * (panel_width + pad)
        draw.text((x, pad), "A" if index == 0 else "B", font=title_font, fill=TEXT)
        for row, line in enumerate(lines):
            draw.text((x, pad + line_height * (2 + row)), line, font=body_font, fill=MUTED)
        image_x = x + (panel_width - width) // 2
        result.paste(image, (image_x, header))
        boxes.append([image_x, header, image_x + width, header + height])
    result.save(path)
    return {"size": list(result.size), "image_boxes_xyxy": boxes,
            "image_transform": "none; original RGB pixels pasted at full resolution"}


def _pad_even(image):
    width, height = image.size
    target = (width + width % 2, height + height % 2)
    if target == image.size:
        return image
    padded = Image.new("RGB", target, BACKGROUND)
    padded.paste(image, (0, 0))
    return padded


def _frames(a, b):
    """120 frames: 36 A, 6 A->B, 36 B, 6 B->A, 36 A.

    The six transition samples are strictly between the two endpoint images; the
    adjacent holds supply each endpoint. Both ends of the clip are identical A.
    """
    for first, second in ((a, b), (b, a)):
        for _ in range(HOLD_FRAMES):
            yield first
        for index in range(1, FADE_FRAMES + 1):
            yield Image.blend(first, second, index / (FADE_FRAMES + 1))
    for _ in range(HOLD_FRAMES):
        yield a


def _encode(a, b, path, ffmpeg):
    width, height = a.size
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-framerate", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
               "-frames:v", str(FRAME_COUNT), "-movflags", "+faststart", str(path)]
    # File-backed stderr cannot fill a PIPE while raw frames are being written.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=errors)
        try:
            for frame in _frames(a, b):
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            returncode = process.wait(timeout=120)
        except Exception as exc:
            process.kill()
            process.wait()
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            errors.seek(0)
            detail = errors.read().decode("utf-8", errors="replace")[-1500:]
            raise RuntimeError(f"Local pair video export failed. {detail}") from exc
        if returncode:
            errors.seek(0)
            detail = errors.read().decode("utf-8", errors="replace")[-1500:]
            raise RuntimeError(f"Local pair video export failed. {detail}")


def _file_record(path):
    data = path.read_bytes()
    return {"file": path.name, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def export_pair(state, a_number, b_number, output_dir=None, *, bundle=True):
    """Export two existing takes without changing state, takes, source or recipes.

    Returns absolute `png`, `mp4`, `manifest`, `bundle` (or None) and `directory`
    paths. A unique child directory is committed only after every output succeeds;
    failed exports clean their temporary files and never replace existing exports.
    `output_dir` is a parent directory, defaulting to state['dir'].
    """
    if not state or not state.get("takes"):
        raise ValueError("Load or develop a frame before exporting a pair.")
    width = _positive(state["width"], "Frame width")
    height = _positive(state["height"], "Frame height")
    if not width.is_integer() or not height.is_integer() or width * height > 16_000_000:
        raise ValueError("Saved frame dimensions are not supported.")
    size = (int(width), int(height))
    takes = [_take(state, a_number), _take(state, b_number)]
    labels = [take_label(state, take) for take in takes]
    # JSON copy before any output ensures metadata is serializable and independent.
    settings = [json.loads(json.dumps(take["settings"], allow_nan=False)) for take in takes]
    a, a_source = _read_image(takes[0]["hero"], size)
    b, b_source = _read_image(takes[1]["hero"], size)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Local pair export requires ffmpeg on the server PATH.")
    source_frame = _file_record(Path(state["still"])) if state.get("still") else None
    parent = Path(output_dir if output_dir is not None else state["dir"]).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"pair_{a_number}_{b_number}_{uuid.uuid4().hex[:12]}"
    with tempfile.TemporaryDirectory(prefix=".pair-export-", dir=parent) as temporary:
        work = Path(temporary) / "ready"
        work.mkdir()
        png, mp4 = work / "pair.png", work / "pair.mp4"
        png_info = _side_by_side(a, b, labels, png)
        even_a, even_b = _pad_even(a), _pad_even(b)
        _encode(even_a, even_b, mp4, ffmpeg)
        manifest = {
            "schema": "hero-frame-rig-pair-export/v1", "app": "Hero Frame Rig",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "processing": "Local Pillow composition and ffmpeg encoding; no Cloud job",
            "source_size": list(size), "working_source": source_frame,
            "source_recipe_sha256": state.get("source_recipe_sha256"),
            "takes": [{"slot": slot, "take": take["n"], "label": label,
                       "settings": saved, "source": source}
                      for slot, take, label, saved, source in zip(
                          ("A", "B"), takes, labels, settings, (a_source, b_source))],
            "png": png_info,
            "video": {"size": list(even_a.size), "fps": FPS, "frames": FRAME_COUNT,
                      "duration_seconds": FRAME_COUNT / FPS,
                      "timeline": [
                          {"kind": "hold", "take": "A", "start_frame": 0, "frames": 36},
                          {"kind": "crossfade", "from": "A", "to": "B", "start_frame": 36, "frames": 6},
                          {"kind": "hold", "take": "B", "start_frame": 42, "frames": 36},
                          {"kind": "crossfade", "from": "B", "to": "A", "start_frame": 78, "frames": 6},
                          {"kind": "hold", "take": "A", "start_frame": 84, "frames": 36}],
                      "padding_right_bottom": [size[0] % 2, size[1] % 2],
                      "image_transform": "No resize or crop; odd dimensions padded right/bottom",
                      "labels": "Clean image pixels; take labels are in pair.png and this manifest",
                      "codec": "H.264, CRF 16, yuv420p (lossy)",
                      "loop": "MP4 player loop setting controls repeated playback; encoded 4s A-B-A has matching loop endpoints"},
            "files": [_file_record(png), _file_record(mp4)],
            "bundle": "pair.zip" if bundle else None,
            "integrity_note": "File hashes cover PNG, MP4 and source images. Manifest and ZIP do not hash themselves."
        }
        manifest_path = work / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if bundle:
            with zipfile.ZipFile(work / "pair.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in (png, mp4, manifest_path):
                    archive.write(path, path.name)
        work.rename(destination)
    return {"directory": str(destination), "png": str(destination / "pair.png"),
            "mp4": str(destination / "pair.mp4"), "manifest": str(destination / "manifest.json"),
            "bundle": str(destination / "pair.zip") if bundle else None}
