"""Review saved takes and load the registered, bundled study without a cloud job.

This is deliberately not a user-recipe import API. Only the checked-in study path and
its recorded SHA-256 are accepted; archive extraction is bounded and path checked.
"""

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

import numpy as np
from PIL import Image


BUNDLED_STUDY = Path(__file__).resolve().parent / "examples" / "nola_study.zip"
NOLA_RECIPE_SHA256 = "15194add2032e925961c66201a4f84807177bf44e0e75642edece54c1e5ac48b"
NOLA_PRODUCER_REVISION = "c85dc96af0e8ed14f99159095a17dd1b698b8d7a"
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_MEMBERS = 128


def selected_take(state):
    """Return the selected take, or the last take for older/unselected frame states."""
    if not state or not state.get("takes"):
        return None
    number = state.get("selected_take")
    if number is None:
        return state["takes"][-1]
    for take in state["takes"]:
        if take["n"] == number:
            return take
    raise ValueError("The selected take is not present in this frame.")


def select_take(state, take_number):
    """Select an existing take and restore its focus, without modifying saved takes."""
    if not state or not state.get("takes"):
        raise ValueError("Load or develop a frame before selecting a take.")
    for take in state["takes"]:
        if take["n"] == take_number:
            settings = take["settings"]
            return {**state, "selected_take": take["n"],
                    "focus": float(settings["focus_disparity"]),
                    "label": int(settings.get("subject_label", 0))}
    raise ValueError(f"Take {take_number} is not present in this frame.")


def _member_path(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("Study archive contains an unsafe path.")
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in ("", ".", "..") for p in name.rstrip("/").split("/")):
        raise ValueError("Study archive contains an unsafe path.")
    return path


def _checked_members(archive):
    infos = archive.infolist()
    if len(infos) > MAX_MEMBERS or sum(i.file_size for i in infos) > MAX_ARCHIVE_BYTES:
        raise ValueError("Study archive exceeds its size limits.")
    seen = set()
    for info in infos:
        path = _member_path(info.filename)
        key = str(path).casefold()
        if key in seen:
            raise ValueError("Study archive contains duplicate paths.")
        seen.add(key)
        if info.file_size > MAX_MEMBER_BYTES or stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError("Study archive contains an unsupported member.")
    return infos


def _positive(value, label):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Study {label} must be finite and positive.")
    return value


def _caption(take):
    settings = take["settings"]
    mode = (f"Cinematic {settings['strength']:.1f}x"
            if settings["mode"].startswith("Cinematic") else "True to lens")
    name = {4: "Building focus", 5: "Road focus"}.get(take["n"])
    title = f"#{take['n']}" + (f" · {name}" if name else "")
    return (f"{title} · f/{settings['f_stop']:g} · {mode} · "
            f"{settings['focus_m']:.1f} m · {settings['look']}")


def load_study(recipe_path, directory):
    """Restore the bundled NOLA study in a fresh directory, selecting building take 4.

    Reuses saved images, depth maps, optical scale and graphs. It does not estimate
    depth, regenerate images, import the Comfy SDK, or submit jobs. Legacy subject-layer
    switches are recovered from the original acceptance script, not guessed from pixels.
    """
    recipe_path = Path(recipe_path).resolve()
    if recipe_path != BUNDLED_STUDY.resolve():
        raise ValueError("Only the registered bundled study can be loaded.")
    if recipe_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Study archive exceeds its size limit.")
    digest = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
    if digest != NOLA_RECIPE_SHA256:
        raise ValueError("Bundled study does not match its recorded recipe hash.")

    destination = Path(directory).resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Load the study into a new or empty directory.")
    with zipfile.ZipFile(recipe_path) as archive:
        infos = _checked_members(archive)
        recipe = json.loads(archive.read("recipe.json"))
        if recipe.get("app") != "Hero Frame Rig":
            raise ValueError("This archive is not a Hero Frame Rig recipe.")
        destination.mkdir(parents=True, exist_ok=True)
        for info in infos:
            target = destination.joinpath(*_member_path(info.filename).parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError("Study archive path escapes its destination.")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)

    def asset(name):
        path = destination.joinpath(*_member_path(name).parts)
        if not path.is_file():
            raise ValueError(f"Study is missing {name}.")
        return str(path)

    def graph(name):
        return json.loads(Path(asset(name)).read_text(encoding="utf-8"))

    frame, optical = recipe["frame"], recipe["optics"]
    width, height = map(int, frame["size"])
    if width <= 0 or height <= 0 or width * height > 16_000_000:
        raise ValueError("Study image size is unsupported.")
    still = asset("still.png")
    matte_path = asset("subject_matte.png")
    with Image.open(still) as image:
        if image.size != (width, height):
            raise ValueError("Study frame dimensions do not match its recipe.")
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    with Image.open(matte_path) as image:
        if image.size != (width, height):
            raise ValueError("Study matte dimensions do not match its frame.")
        matte = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    Image.fromarray(np.clip(rgb * (0.3 + 0.7 * matte[..., None]), 0, 255).astype(np.uint8)).save(
        destination / "subject_view.png")

    takes = []
    seen = set()
    # Recorded in tests/hero-builder-20261012/run_hero_acceptance.py, lines 92-96.
    known_subject_layer = {1: True, 2: False, 3: False, 4: False, 5: False}
    for saved in recipe["takes"]:
        number = int(saved["take"])
        if number in seen:
            raise ValueError("Study contains duplicate take numbers.")
        seen.add(number)
        settings = dict(saved["settings"])
        if "subject_layer_enabled" not in settings:
            if number in known_subject_layer:
                settings["subject_layer_enabled"] = known_subject_layer[number]
                settings["subject_layer_setting_source"] = "recorded acceptance script"
            else:
                settings["subject_layer_enabled"] = bool(settings.get("layered", 0))
                settings["subject_layer_setting_source"] = "inferred from saved layered count"
                settings["subject_layer_enabled_inferred"] = True
        take = {"n": number, "hero": asset(saved["hero"]),
                "files": {key: asset(value) for key, value in saved["inputs"].items()},
                "settings": settings, "graph": graph(saved["graph"]),
                "job": saved.get("cloud_job")}
        take["caption"] = _caption(take)
        takes.append(take)

    state = {"dir": str(destination), "still": still, "depth_exr": asset("depth.exr"),
             "sky": asset("sky.png"), "matte": matte_path, "disparity": asset("disparity.png"),
             "z_near": _positive(optical["z_near_m"], "near scale"),
             "cine": _positive(optical["cinematic_multiplier"], "cinematic multiplier"),
             "lens": int(frame["lens_mm_full_frame"]), "width": width, "height": height,
             "source": frame["source"], "shot": frame.get("shot"), "scene": frame.get("scene"),
             "prompt": frame.get("prompt"), "seed": frame.get("seed"),
             "frame_graph": graph(frame["graph"]), "frame_job": frame.get("cloud_job"),
             "takes": takes, "recipe_models": recipe.get("models", {}),
             "source_recipe_sha256": digest, "producer_revision": NOLA_PRODUCER_REVISION,
             "study": {"id": "nola", "title": "NOLA focus study", "saved_example": True,
                       "source_recipe_sha256": digest, "producer_revision": NOLA_PRODUCER_REVISION,
                       "saved_at": recipe.get("made"), "selected_default": 4,
                       "original_upload_retained": bool(recipe.get("original_upload"))}}
    original = recipe.get("original_upload")
    if original:
        state.update(original=original, original_name=original["filename"],
                     original_file=asset(original["archive_path"]))
    return select_take(state, 4)
