"""Bounded, offline reopening of saved Hero Frame Rig image recipes.

Saved graphs are validated and retained as data, never executed. Further development
uses the application's current renderer and the saved working frame/depth. This is
separate from review_state.load_study's pinned, trusted bundled-study loader.
"""

import copy
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import struct
import tempfile
import unicodedata
import zipfile

import numpy as np
from PIL import Image

from review_state import select_take


MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_MEMBERS = 512
MAX_TAKES = 64
MAX_PIXELS = 16_000_000
MAX_ORIGINAL_PIXELS = 64_000_000
MAX_DECODED_PIXELS = 300_000_000
MODES = ("True to lens (1x)", "Cinematic (stylized, up to 4x)")
_RESERVED = re.compile(r"^(?:con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])(?:\.|$)", re.I)


def _path(name):
    if (not isinstance(name, str) or not name or len(name) > 240
            or any(ord(c) < 32 for c in name) or any(c in name for c in '\\:<>"|?*')):
        raise ValueError("Recipe contains an unsafe archive path.")
    parts = name.split("/")
    if any(p in ("", ".", "..") or p.endswith((".", " ")) or _RESERVED.match(p) for p in parts):
        raise ValueError("Recipe contains an unsafe archive path.")
    return PurePosixPath(name)


def _read_archive(path):
    archive_size = path.stat().st_size
    if archive_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Image recipe ZIP exceeds 256 MiB.")
    # Bound the central-directory entry count before ZipFile constructs one Python
    # object per entry. Our bounded exports never require ZIP64 or spanning disks.
    with path.open("rb") as source:
        source.seek(max(0, archive_size - 65557))
        tail = source.read(65557)
    end = tail.rfind(b"PK\x05\x06")
    if end < 0 or len(tail) - end < 22:
        raise ValueError("Image recipe has no supported ZIP directory.")
    _, disk, central_disk, count, total, length, offset, comment = struct.unpack("<4s4H2IH", tail[end:end + 22])
    absolute_end = max(0, archive_size - 65557) + end
    if (disk or central_disk or count != total or total > MAX_MEMBERS or total == 65535
            or offset + length != absolute_end or len(tail) - end != 22 + comment):
        raise ValueError("Recipe ZIP directory exceeds limits or uses an unsupported layout.")
    contents = {}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_MEMBERS or sum(i.file_size for i in infos) > MAX_ARCHIVE_BYTES:
            raise ValueError("Image recipe exceeds its expanded size or member limit.")
        seen = set()
        for info in infos:
            # App exports contain files only. Reject directory records and special files.
            if info.orig_filename != info.filename:
                raise ValueError("Recipe contains an unsafe archive path.")
            name = str(_path(info.filename))
            key = unicodedata.normalize("NFC", name).casefold()
            mode = stat.S_IFMT(info.external_attr >> 16)
            if key in seen or mode not in (0, stat.S_IFREG) or info.is_dir() or info.flag_bits & 1:
                raise ValueError("Recipe contains duplicate, encrypted, or unsupported members.")
            seen.add(key)
            if info.file_size > MAX_MEMBER_BYTES or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError("Recipe member exceeds its limit or uses unsupported compression.")
        for key in seen:
            if any("/".join(key.split("/")[:i]) in seen for i in range(1, len(key.split("/")))):
                raise ValueError("Recipe has conflicting file and directory paths.")
        for info in infos:
            limit = MAX_JSON_BYTES if info.filename.endswith(".json") else MAX_MEMBER_BYTES
            if info.file_size > limit:
                raise ValueError("Recipe JSON exceeds its size limit.")
            with archive.open(info) as source:
                data = source.read(limit + 1)  # Includes ZIP CRC validation; never unbounded.
            if len(data) != info.file_size or len(data) > limit:
                raise ValueError("Recipe member size does not match its declaration.")
            contents[info.filename] = data
    return contents


def _json(data, label):
    if len(data) > MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds its JSON size limit.")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON keys.")
            result[key] = value
        return result

    def constant(_):
        raise ValueError(f"{label} contains a non-finite number.")

    try:
        result = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
        # Bound nesting and reject numbers such as 1e999 (not caught by parse_constant).
        pending = [(result, 0)]
        while pending:
            value, depth = pending.pop()
            if depth > 64:
                raise ValueError(f"{label} is too deeply nested.")
            if isinstance(value, float) and not math.isfinite(value):
                constant(value)
            if isinstance(value, dict):
                pending.extend((v, depth + 1) for v in value.values())
            elif isinstance(value, list):
                pending.extend((v, depth + 1) for v in value)
        return result
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not supported JSON.") from exc


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    return value


def _number(value, label, minimum, maximum, integer=False):
    if (type(value) not in ((int,) if integer else (int, float))
            or not math.isfinite(value) or not minimum <= value <= maximum):
        raise ValueError(f"{label} is outside its supported range.")
    return value


def _size(value, label, limit=MAX_PIXELS):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} needs width and height.")
    width, height = (_number(v, label, 1, 16384, integer=True) for v in value)
    if width * height > limit:
        raise ValueError(f"{label} exceeds the pixel limit.")
    return width, height


def _settings(value):
    settings = copy.deepcopy(_object(value, "Take settings"))
    for key, lo, hi in (("f_stop", .5, 64), ("focus_disparity", 0, 1),
                        ("focus_m", 0, 1_000_000), ("strength", 0, 4)):
        _number(settings.get(key), key, lo, hi)
    # Earlier exports used these short labels. Preserve saved settings verbatim;
    # only the UI representation is normalized by app._outputs.
    if settings.get("mode") not in (*MODES, "True to lens", "Cinematic"):
        raise ValueError("Take has an unsupported depth-of-field mode.")
    if settings.get("look") not in ("digital", "film", "cinestill"):
        raise ValueError("Take has an unsupported look.")
    for key, lo, hi in (("blur_px", 0, 100), ("guide", 0, 100), ("edge_cleanup", 0, 1)):
        if key in settings:
            _number(settings[key], key, lo, hi)
    for key in ("held", "layered", "subject_label", "grain_seed"):
        if key in settings:
            _number(settings[key], key, 0, 2**63 - 1, integer=True)
    for key in ("subject_layer_enabled", "blur_capped"):
        if key in settings and type(settings[key]) is not bool:
            raise ValueError(f"{key} must be a boolean.")
    return settings


def _graph(value, contents):
    graph = _object(value, "Saved graph")
    if not graph or len(graph) > 2048:
        raise ValueError("Saved graph must contain 1–2048 nodes.")
    dependencies = {}
    for node_id, entry in graph.items():
        _object(entry, "Graph node")
        if not node_id or not isinstance(entry.get("class_type"), str) or not entry["class_type"]:
            raise ValueError("Graph has an invalid node.")
        inputs = _object(entry.get("inputs"), "Graph inputs")
        dependencies[node_id] = set()
        for value in inputs.values():
            if isinstance(value, list):
                if (len(value) != 2 or not isinstance(value[0], str) or value[0] not in graph
                        or type(value[1]) is not int or not 0 <= value[1] <= 1024):
                    raise ValueError("Graph contains a dangling or invalid connection.")
                dependencies[node_id].add(value[0])
        if entry["class_type"] == "LoadImage":
            name = inputs.get("image")
            _path(name)
            if name not in contents or not name.endswith(".png"):
                raise ValueError("LoadImage must reference a bundled PNG.")
    # Iterative topological check avoids recursion on an adversarial graph.
    pending = {key: set(value) for key, value in dependencies.items()}
    while pending:
        ready = {key for key, value in pending.items() if not value}
        if not ready:
            raise ValueError("Saved graph contains a dependency cycle.")
        pending = {key: value - ready for key, value in pending.items() if key not in ready}
    return graph


def _validate(contents):
    def member(name):
        _path(name)
        if name not in contents:
            raise ValueError(f"Image recipe is missing {name}.")
        return contents[name]

    recipe = _object(_json(member("recipe.json"), "recipe.json"), "Recipe")
    version = recipe.get("schema_version", 1)
    if recipe.get("app") != "Hero Frame Rig" or type(version) is not int or version not in (1, 2):
        raise ValueError("Choose a supported Hero Frame Rig image recipe (schema 1 or 2).")
    verified = "integrity.json" in contents
    if version == 2 and not verified:
        raise ValueError("Schema 2 requires integrity.json.")
    if verified:
        manifest = _object(_json(member("integrity.json"), "integrity.json"), "Integrity manifest")
        hashes = _object(manifest.get("files"), "Integrity files")
        if manifest.get("algorithm") != "sha256" or set(hashes) != set(contents) - {"integrity.json"}:
            raise ValueError("Integrity manifest does not cover the complete archive.")
        if any(hashes[name] != hashlib.sha256(contents[name]).hexdigest() for name in hashes):
            raise ValueError("Image recipe checksum mismatch.")

    frame = _object(recipe.get("frame"), "Image frame")
    size = _size(frame.get("size"), "Working image size")
    if frame.get("source") not in ("uploaded", "generated"):
        raise ValueError("This is not a saved image recipe.")
    _number(frame.get("lens_mm_full_frame"), "Lens", 1, 2000)
    for key in ("shot", "scene", "prompt", "engine", "expanded_prompt", "prompt_system_source",
                "cloud_job", "image_format", "resolution"):
        if frame.get(key) is not None and not isinstance(frame[key], str):
            raise ValueError(f"Frame {key} must be text.")
    optical = _object(recipe.get("optics"), "Optics")
    _number(optical.get("z_near_m"), "Depth scale", .000001, 1_000_000)
    _number(optical.get("cinematic_multiplier"), "Cinematic multiplier", .01, 4)
    saved = recipe.get("takes")
    if not isinstance(saved, list) or not 1 <= len(saved) <= MAX_TAKES:
        raise ValueError("Image recipe must contain 1–64 takes.")
    takes, numbers = [], set()
    referenced = {"still.png", "depth.exr", "sky.png", "subject_matte.png", "disparity.png"}
    for take in saved:
        _object(take, "Take")
        n = _number(take.get("take"), "Take number", 1, 1_000_000, integer=True)
        if n in numbers:
            raise ValueError("Recipe contains duplicate take numbers.")
        numbers.add(n)
        if take.get("hero") != f"take{n:02d}_hero.png":
            raise ValueError("Take hero names must match their saved take numbers.")
        inputs = _object(take.get("inputs"), "Take inputs")
        if len(inputs) > 128:
            raise ValueError("Take has too many input maps.")
        for name in (take.get("hero"), *inputs.values()):
            _path(name)
            # Saved image recipes use root-level PNGs; preserving names keeps saved
            # graphs reproducible and prevents basename collisions on the next save.
            if "/" in name or not name.endswith(".png"):
                raise ValueError("Take image inputs must be root-level PNGs.")
            referenced.add(name)
        settings = _settings(take.get("settings"))
        graph = _graph(_json(member(take.get("graph")), "Take graph"), contents)
        point = take.get("focus_point")
        if point is not None:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("Focus point must contain normalized [x, y].")
            for coordinate in point:
                _number(coordinate, "Focus point", 0, 1)
        takes.append({**take, "settings": settings, "parsed_graph": graph})
    selected = recipe.get("selected_take", saved[-1]["take"] if version == 1 else None)
    if type(selected) is not int or selected not in numbers:
        raise ValueError("The selected take is not present in the recipe.")
    frame_graph = _graph(_json(member(frame.get("graph", "frame_graph.json")), "Frame graph"), contents)
    for graph in (frame_graph, *(take["parsed_graph"] for take in takes)):
        bindings = {node["inputs"]["image"] for node in graph.values() if node["class_type"] == "LoadImage"}
        if not bindings.issubset(referenced):
            raise ValueError("Graph image bindings must be recorded frame assets, heroes, or take inputs.")

    total_pixels = 0
    for name in referenced:
        data = member(name)
        if name == "depth.exr":
            continue  # Header and full depth decode below, after bounded extraction.
        with Image.open(io.BytesIO(data)) as image:
            _size(image.size, "Saved image size")
            total_pixels += image.width * image.height
            if total_pixels > MAX_DECODED_PIXELS:
                raise ValueError("Recipe exceeds its combined decoded pixel limit.")
            if image.format != "PNG" or getattr(image, "n_frames", 1) != 1 or image.size != size:
                raise ValueError(f"{name} must be a single PNG matching the working frame.")
            image.load()

    original = recipe.get("original_upload")
    if original is not None:
        _object(original, "Original upload")
        name = original.get("archive_path")
        data = member(name)
        if not name.startswith("source/") or original.get("sha256") != hashlib.sha256(data).hexdigest() or original.get("bytes") != len(data):
            raise ValueError("Original upload bytes do not match their recorded hash and size.")
        if not isinstance(original.get("filename"), str) or not original["filename"]:
            raise ValueError("Original upload needs its saved filename.")
        with Image.open(io.BytesIO(data)) as image:
            _size(image.size, "Original image size", MAX_ORIGINAL_PIXELS)
            orientation = image.getexif().get(274, 1)
            display = image.size[::-1] if orientation in (5, 6, 7, 8) else image.size
            if (original.get("format") != image.format or original.get("stored_size") != list(image.size)
                    or original.get("display_size") != list(display)
                    or original.get("exif_orientation_applied") != orientation
                    or original.get("working_size") != list(size)):
                raise ValueError("Original upload dimensions or encoding differ from its provenance.")
            image.load()
    return recipe, takes, frame_graph, selected, verified


def _depth(path):
    import OpenEXR

    header_file = OpenEXR.File(str(path), header_only=True)
    if len(header_file.parts) != 1:
        raise ValueError("Depth must be a single-part EXR.")
    header = header_file.header()
    if header.get("type") != OpenEXR.scanlineimage or not 1 <= len(header["channels"]) <= 4:
        raise ValueError("Depth must be a scanline EXR with at most four channels.")
    for key in ("dataWindow", "displayWindow"):
        lo, hi = header[key]
        _size([int(hi[0]) - int(lo[0]) + 1, int(hi[1]) - int(lo[1]) + 1], "Depth size")
    decoded = OpenEXR.File(str(path))
    for channel in decoded.channels().values():
        pixels = channel.pixels
        if (pixels.dtype.kind != "f" or not np.isfinite(pixels).all()
                or np.any(pixels < 0) or not np.any(pixels > 0)):
            raise ValueError("Depth must contain finite, nonnegative floating-point values.")


def load_image_recipe(recipe_path, directory=None):
    """Restore a validated image recipe in a fresh directory without inference.

    Hashes detect inconsistent packages, not trusted authorship. schema 1 archives
    without a manifest remain readable and are explicitly marked as unchecked.
    No saved graph is evaluated or external resource fetched, even after validation.
    """
    path = Path(recipe_path)
    destination = None
    created = False
    try:
        contents = _read_archive(path)
        recipe, saved, graph, selected, verified = _validate(contents)
        if directory is None:
            destination = Path(tempfile.mkdtemp(prefix="rig_image_recipe_"))
        else:
            destination = Path(directory).absolute()
            # Require a new destination so failed imports cannot touch caller files.
            destination.mkdir(parents=True, exist_ok=False)
        created = True
        imported = destination / "imported"
        imported.mkdir()
        for name, data in contents.items():
            target = imported.joinpath(*_path(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        _depth(imported / "depth.exr")

        def asset(name):
            return str(imported.joinpath(*_path(name).parts))

        with Image.open(asset("still.png")) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        with Image.open(asset("subject_matte.png")) as image:
            matte = np.asarray(image.convert("L"), dtype=np.float32) / 255
        Image.fromarray(np.clip(rgb * (.3 + .7 * matte[..., None]), 0, 255).astype(np.uint8)).save(destination / "subject_view.png")
        takes = []
        for take in saved:
            settings = take["settings"]
            mode = f"Cinematic {settings['strength']:.1f}x" if settings["mode"].startswith("Cinematic") else "True to lens"
            caption = f"#{take['take']} · f/{settings['f_stop']:g} · {mode} · {settings['focus_m']:.1f} m · {settings['look']}"
            takes.append({"n": take["take"], "hero": asset(take["hero"]),
                          "files": {key: asset(value) for key, value in take["inputs"].items()},
                          "settings": settings, "graph": take["parsed_graph"], "job": take.get("cloud_job"),
                          "provenance": take.get("provenance", recipe.get("provenance", {})),
                          "focus_point": take.get("focus_point"), "caption": caption})
        frame, optical = recipe["frame"], recipe["optics"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        state = {"dir": str(destination), "still": asset("still.png"), "depth_exr": asset("depth.exr"),
                 "sky": asset("sky.png"), "matte": asset("subject_matte.png"), "disparity": asset("disparity.png"),
                 "z_near": optical["z_near_m"], "cine": optical["cinematic_multiplier"],
                 "lens": frame["lens_mm_full_frame"], "width": frame["size"][0], "height": frame["size"][1],
                 "source": frame["source"], "frame_graph": graph, "frame_job": frame.get("cloud_job"),
                 "frame_provenance": frame.get("provenance", recipe.get("provenance", {})),
                 "recipe_models": recipe.get("models", {}), "takes": takes, "source_recipe_sha256": digest,
                 "imported_recipe": {"sha256": digest, "schema_version": recipe.get("schema_version", 1),
                                     "integrity_verified": verified, "saved_at": recipe.get("made")}}
        state.update({key: frame[key] for key in ("shot", "scene", "prompt", "seed", "engine", "expanded_prompt", "prompt_system_source") if key in frame})
        if frame["source"] == "generated":
            state.update({key: frame[key] for key in ("image_format", "resolution") if frame.get(key) is not None})
        if recipe.get("original_upload"):
            original = recipe["original_upload"]
            state.update(original=original, original_name=original["filename"], original_file=asset(original["archive_path"]))
        return select_take(state, selected)
    except Exception as exc:
        if created and destination is not None and destination.is_dir():
            # Only a directory created successfully by this call can reach here.
            shutil.rmtree(destination)
        if isinstance(exc, ValueError):
            raise
        raise ValueError("The image recipe could not be read or contains an invalid asset.") from exc
