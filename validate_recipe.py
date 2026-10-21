"""Offline checks and replay preparation for a Hero Frame Rig recipe ZIP.

Usage:
    python validate_recipe.py hero_frame_recipe.zip --report validation.json
    python validate_recipe.py hero_frame_recipe.zip --prepare replay_bundle

This never contacts ComfyUI, submits jobs, installs nodes, or changes source images.
Preparation writes a NEW directory with namespaced ComfyUI/input files and API graphs.
A passing report confirms package consistency, NOT a successful desktop replay or
pixel-identical output. EXR is checked by signature only; PNGs are fully decoded.
Requires Pillow, already in the app requirements.
"""

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

from PIL import Image, ImageOps


def safe_name(name):
    path = PurePosixPath(name)
    return bool(name and not path.is_absolute() and ".." not in path.parts
                and "\\" not in name and ":" not in name)


def validate_integrity(contents, report, required=False):
    """Check all archive file bytes against a manifest; this is not a signature."""
    errors = report["errors"]
    if "integrity.json" not in contents:
        if required:
            errors.append("Schema v2 requires integrity.json.")
        return
    first_error = len(errors)
    result = report["integrity"] = {"verified": False, "files_checked": 0}
    try:
        manifest = json.loads(contents["integrity.json"])
    except (ValueError, UnicodeDecodeError) as exc:
        errors.append(f"integrity.json cannot be decoded: {exc}")
        return
    if not isinstance(manifest, dict) or manifest.get("algorithm") != "sha256":
        errors.append("integrity.json must specify algorithm 'sha256'.")
        return
    result["algorithm"] = "sha256"
    recorded = manifest.get("files")
    if not isinstance(recorded, dict):
        errors.append("integrity.json.files must map archive paths to SHA-256 hashes.")
        return
    expected = set(contents) - {"integrity.json"}
    for name in sorted(expected - set(recorded)):
        errors.append(f"integrity.json is missing archive member: {name}")
    for name in sorted(set(recorded) - expected):
        errors.append(f"integrity.json lists a non-member or itself: {name}")
    for name in sorted(expected & set(recorded)):
        actual = hashlib.sha256(contents[name]).hexdigest()
        if recorded[name] != actual:
            errors.append(f"Integrity SHA-256 mismatch: {name}")
        result["files_checked"] += 1
    result["verified"] = len(errors) == first_error


def validate(path):
    report = {"zip": str(Path(path).resolve()), "errors": [], "warnings": [],
              "files": {}, "graphs": {}, "desktop_replay": "not run"}
    errors, warnings = report["errors"], report["warnings"]
    contents = {}
    with zipfile.ZipFile(path) as archive:
        names = [i.filename for i in archive.infolist() if not i.is_dir()]
        if len(names) != len(set(names)):
            errors.append("ZIP contains duplicate filenames.")
        for name in names:
            if not safe_name(name):
                errors.append(f"Non-portable archive path: {name!r}")
                continue
            data = archive.read(name)  # includes ZIP CRC verification
            contents[name] = data
            info = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            report["files"][name] = info
            if name.lower().endswith(".png"):
                try:
                    with Image.open(io.BytesIO(data)) as im:
                        info.update(format=im.format, size=list(im.size), mode=im.mode)
                        if im.format != "PNG":
                            errors.append(f"{name}: encoded as {im.format}, not PNG.")
                        im.load()
                except Exception as exc:
                    errors.append(f"{name}: image decode failed: {exc}")
            elif name.lower().endswith(".exr"):
                if data[:4] != b"\x76\x2f\x31\x01":
                    errors.append(f"{name}: not an OpenEXR signature.")
    if "recipe.json" not in contents:
        errors.append("recipe.json is missing.")
        report["ok"] = False
        return report, contents
    try:
        recipe = json.loads(contents["recipe.json"])
    except (ValueError, UnicodeDecodeError) as exc:
        errors.append(f"recipe.json cannot be decoded: {exc}")
        report["ok"] = False
        return report, contents
    if not isinstance(recipe, dict):
        errors.append("recipe.json must contain an object.")
        report["ok"] = False
        return report, contents
    version = recipe.get("schema_version", 1)
    report["schema_version"] = version
    if type(version) is not int or version not in (1, 2):
        errors.append(f"Unsupported recipe schema_version: {version!r}")
    elif version == 1:
        warnings.append("Legacy schema v1: selection and a complete integrity manifest are not required.")
    validate_integrity(contents, report, required=(version == 2))
    expected_size = recipe.get("frame", {}).get("size")
    original = recipe.get("original_upload")
    if original:
        original_name = original.get("archive_path")
        if not isinstance(original_name, str) or not safe_name(original_name) or original_name not in contents:
            errors.append("original_upload.archive_path does not identify a bundled original.")
        else:
            data = contents[original_name]
            actual_sha = hashlib.sha256(data).hexdigest()
            report["original_upload"] = {"archive_path": original_name, "sha256": actual_sha,
                                         "bytes": len(data), "verified": False}
            if original.get("sha256") != actual_sha:
                errors.append("Original upload SHA-256 differs from its provenance record.")
            if original.get("bytes") != len(data):
                errors.append("Original upload byte count differs from its provenance record.")
            try:
                with Image.open(io.BytesIO(data)) as im:
                    if original.get("stored_size") != list(im.size):
                        errors.append("Original upload stored dimensions differ from its provenance record.")
                    if original.get("format") != im.format:
                        errors.append("Original upload encoding differs from its provenance record.")
                    upright = ImageOps.exif_transpose(im)
                    if original.get("display_size") != list(upright.size):
                        errors.append("Original upload oriented dimensions differ from its provenance record.")
                    if original.get("exif_orientation_applied") != im.getexif().get(274, 1):
                        errors.append("Original upload EXIF orientation differs from its provenance record.")
                    im.load()
            except Exception as exc:
                errors.append(f"Original upload decode failed: {exc}")
            if original.get("working_size") != expected_size:
                errors.append("Original upload working dimensions differ from recipe.frame.size.")
            report["original_upload"]["verified"] = not any(
                "original upload" in e.lower() for e in errors)
    elif recipe.get("frame", {}).get("source") == "uploaded":
        warnings.append("Uploaded original is not bundled with provenance; only the working image is retained.")
    for name in ("still.png", "depth.exr", "sky.png", "subject_matte.png", "disparity.png"):
        if name not in contents:
            errors.append(f"Required frame input missing: {name}")
    for name in ("still.png", "subject_matte.png", "disparity.png"):
        if name in report["files"] and report["files"][name].get("size") != expected_size:
            errors.append(f"{name}: dimensions differ from recipe.frame.size {expected_size}.")
    takes = recipe.get("takes", [])
    if not takes:
        errors.append("No takes in recipe.")
    if len({t["take"] for t in takes}) != len(takes):
        errors.append("Duplicate take numbers.")
    if version == 2 and any(type(t["take"]) is not int or t["take"] < 1 for t in takes):
        errors.append("Schema v2 take numbers must be positive integers.")
    if version == 2 or "selected_take" in recipe:
        selected = recipe.get("selected_take")
        if type(selected) is not int or selected not in {t["take"] for t in takes}:
            errors.append("selected_take must be an integer matching one takes[].take value.")
        else:
            report["selected_take"] = selected
    graph_names = [recipe.get("frame", {}).get("graph", "frame_graph.json")]
    for take in takes:
        graph_names.append(take["graph"])
        for name in [take["hero"], *take.get("inputs", {}).values()]:
            if name not in contents:
                errors.append(f"Take {take['take']}: missing {name}")
            elif report["files"][name].get("size") != expected_size:
                errors.append(f"Take {take['take']}: {name} dimensions differ from the frame.")
        if not take.get("cloud_job"):
            warnings.append(f"Take {take['take']}: no cloud job provenance.")
    for name in graph_names:
        if name not in contents:
            errors.append(f"Graph missing: {name}")
            continue
        graph = json.loads(contents[name])
        if not isinstance(graph, dict) or not all(isinstance(n, dict) and "class_type" in n for n in graph.values()):
            errors.append(f"{name}: expected ComfyUI API-format node graph.")
            continue
        classes, inputs, dependencies = set(), {}, {}
        for node_id, node in graph.items():
            classes.add(node["class_type"])
            dependencies[node_id] = []
            for key, value in node.get("inputs", {}).items():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int):
                    dependencies[node_id].append(value[0])
                    if value[0] not in graph or value[1] < 0:
                        errors.append(f"{name}: dangling connection {node_id}.{key} -> {value}")
            if node["class_type"] == "LoadImage":
                image = node.get("inputs", {}).get("image")
                inputs[node_id] = image
                if not isinstance(image, str) or not safe_name(image) or image not in contents:
                    errors.append(f"{name}: LoadImage {node_id} has no bundled local input: {image!r}")
        seen, active = set(), set()
        def visit(node_id):
            if node_id in active:
                raise ValueError("cyclic node connections")
            if node_id in seen or node_id not in graph:
                return
            active.add(node_id)
            for dep in dependencies[node_id]:
                visit(dep)
            active.remove(node_id)
            seen.add(node_id)
        try:
            for node_id in graph:
                visit(node_id)
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
        report["graphs"][name] = {"nodes": len(graph), "class_types": sorted(classes), "load_images": inputs}
    warnings.append("Custom-node availability, revisions, GPU behavior and pixel replay are not checked offline.")
    warnings.append("EXR payload decode and metric-depth calibration are not checked by this helper.")
    provenance = recipe.get("provenance") or {}
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object when provided.")
        provenance = {}
    if not recipe.get("app_revision") and not provenance.get("source_files"):
        warnings.append("Recipe has no app_revision; preserve the tested Space commit separately.")
    if provenance:
        report["provenance"] = provenance
        warnings.append("Source/package provenance is recorded, not independently checked against the producing runtime.")
    report["take_count"] = len(takes)
    report["ok"] = not errors
    return report, contents


def prepare(path, destination, report, contents):
    if not report["ok"]:
        raise ValueError("Refusing to prepare an inconsistent recipe; see validation errors.")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    namespace = "hero_recipe_" + hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    input_dir = destination / "input" / namespace
    input_dir.mkdir(parents=True)
    graphs_dir = destination / "api_graphs"
    graphs_dir.mkdir()
    copied = set()
    for name, info in report["graphs"].items():
        graph = copy.deepcopy(json.loads(contents[name]))
        for node_id, image in info["load_images"].items():
            if image not in copied:
                target = input_dir / image
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(contents[image])
                copied.add(image)
            graph[node_id]["inputs"]["image"] = f"{namespace}/{image}"
        (graphs_dir / Path(name).name).write_text(json.dumps(graph, indent=2), encoding="utf-8")
    (destination / "README.txt").write_text(
        "PREPARED ONLY: no job was run.\n"
        f"Copy input/{namespace} into your ComfyUI/input/ directory.\n"
        "Load/queue a takeNN_develop_graph.json with ComfyUI's API graph support.\n"
        "Install the graph's required core/Radiance nodes first (listed in validation.json).\n"
        "A develop graph reuses bundled pixels; frame_graph.json reruns generation/depth/models.\n"
        "Do not queue frame_graph.json merely to reproduce a saved take.\n"
        "The recipe does not pin custom-node revisions; output may vary between installs.\n",
        encoding="utf-8")
    (destination / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(destination.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("zip")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--prepare", type=Path, help="Create a new, namespaced replay folder; never submits jobs.")
    args = parser.parse_args()
    report, contents = validate(args.zip)
    if args.prepare:
        report["prepared"] = prepare(args.zip, args.prepare, report, contents)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("files", "graphs")}, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
