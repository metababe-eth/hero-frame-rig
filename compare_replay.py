"""Compare a supplied replay PNG with a recipe's saved take, without running jobs.

    python compare_replay.py recipe.zip local_output.png --report comparison.json
    python compare_replay.py recipe.zip local_output.png --take 4 --report take4.json

The default take is recipe.selected_take; legacy recipes require --take. Exit 0 means
pixel-exact or within the stated numerical tolerance, NOT that local execution has
been verified. Keep the ComfyUI prompt/history and environment record separately.
Defaults allow MAE <= 0.5 and maximum error <= 2 in 8-bit samples. These conservative
engineering tolerances are not perceptual, optical, or artistic acceptance criteria.
No resizing, channel conversion, EXIF rotation, registration or image edits occur.
"""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from validate_recipe import validate


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def decode(data):
    """Decode original sample values and record their actual shape/channel layout."""
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        pixels = np.asarray(image).copy()
        icc = image.info.get("icc_profile")
        record = {
            "file_sha256": sha256(data), "file_bytes": len(data),
            "pixel_sha256": sha256(pixels.tobytes()),
            "format": image.format, "mode": image.mode,
            "bands": list(image.getbands()), "channels": len(image.getbands()),
            "size": list(image.size), "array_shape": list(pixels.shape),
            "dtype": str(pixels.dtype), "frames": getattr(image, "n_frames", 1),
            # Pillow can down-convert 16-bit RGB PNGs while decoding; inspect IHDR
            # rather than mistaking that uint8 array for the file's actual precision.
            "png_bit_depth": data[24] if data[:8] == b"\x89PNG\r\n\x1a\n" else None,
            "orientation": image.getexif().get(274, 1),
            "color_metadata": {
                "icc_sha256": sha256(icc) if icc else None,
                "gamma": image.info.get("gamma"), "srgb": image.info.get("srgb"),
                "chromaticity": image.info.get("chromaticity"),
            },
        }
    return pixels, record


def compare_image_bytes(reference_bytes, output_bytes, max_mae=0.5, max_abs=2.0):
    """Measure two images as delivered; incompatible layouts are never normalized."""
    if not all(math.isfinite(v) and 0 <= v <= 255 for v in (max_mae, max_abs)):
        raise ValueError("Tolerance values must be finite and between 0 and 255.")
    reference, ref_info = decode(reference_bytes)
    output, out_info = decode(output_bytes)
    report = {
        "comparison_status": "incompatible", "reference": ref_info, "output": out_info,
        "file_bytes_equal": reference_bytes == output_bytes,
        "thresholds": {"sample_range": [0, 255], "max_mae": max_mae, "max_abs": max_abs},
        "errors": [], "metrics": None, "within_tolerance": False,
        "execution_evidence": "Not checked. An image comparison does not prove a local render occurred.",
        "processing": "Original decoded samples; no resize, color conversion, alignment, EXIF transform or edits.",
    }
    errors = report["errors"]
    for label, info in (("Reference", ref_info), ("Output", out_info)):
        if info["format"] != "PNG":
            errors.append(f"{label} is encoded as {info['format']}; the saved workflow outputs PNG.")
        if info["dtype"] != "uint8" or info["mode"] not in {"L", "LA", "RGB", "RGBA"}:
            errors.append(f"{label} is not a supported 8-bit sample layout: {info['mode']}/{info['dtype']}.")
        if info["format"] == "PNG" and info["png_bit_depth"] != 8:
            errors.append(f"{label} PNG has {info['png_bit_depth']}-bit samples; comparison requires 8-bit files.")
        if info["frames"] != 1:
            errors.append(f"{label} contains multiple frames; this comparison requires one still.")
    for field in ("size", "mode", "bands", "array_shape", "dtype", "png_bit_depth", "orientation", "color_metadata"):
        if ref_info[field] != out_info[field]:
            errors.append(f"Reference/output {field} differ.")
    if reference.shape != output.shape or reference.dtype != np.uint8 or output.dtype != np.uint8:
        return report
    # Metrics remain useful for diagnosing equal-shape images with different metadata,
    # but such a result remains incompatible and can never pass the tolerance gate.
    delta = np.abs(reference.astype(np.float64) - output.astype(np.float64))
    per_pixel = delta if delta.ndim == 2 else delta.max(axis=2)
    channel_values = delta.reshape(-1, 1 if delta.ndim == 2 else delta.shape[2])
    mse = float(np.mean(delta ** 2))
    metrics = {
        "pixel_exact": bool(np.array_equal(reference, output)),
        "mae": float(delta.mean()), "rmse": math.sqrt(mse), "max_abs": float(delta.max()),
        "p99_abs": float(np.percentile(delta, 99)), "p99_9_abs": float(np.percentile(delta, 99.9)),
        "changed_pixels": int(np.count_nonzero(per_pixel)), "total_pixels": int(per_pixel.size),
        "changed_pixel_percent": float(np.count_nonzero(per_pixel) / per_pixel.size * 100),
        "per_channel_mae": dict(zip(ref_info["bands"], map(float, channel_values.mean(axis=0)))),
        "psnr_db": 10 * math.log10(255 ** 2 / mse) if mse else None,
        "psnr_note": "Null means samples are exactly equal (zero MSE).",
    }
    report["metrics"] = metrics
    if not errors:
        report["within_tolerance"] = metrics["mae"] <= max_mae and metrics["max_abs"] <= max_abs
        report["comparison_status"] = (
            "pixel_exact" if metrics["pixel_exact"] else
            "within_tolerance" if report["within_tolerance"] else "different")
    return report


def compare_recipe(recipe_path, output_path, take_number=None, max_mae=0.5, max_abs=2.0):
    """Validate the real archive, resolve the selected take, then compare its image."""
    recipe_path, output_path = Path(recipe_path), Path(output_path)
    validation, contents = validate(recipe_path)
    if not validation["ok"]:
        return {"comparison_status": "invalid_recipe", "errors": validation["errors"],
                "recipe": str(recipe_path.resolve()), "within_tolerance": False,
                "execution_evidence": "Not checked."}
    recipe = json.loads(contents["recipe.json"])
    number = take_number if take_number is not None else recipe.get("selected_take")
    if type(number) is not int:
        raise ValueError("This recipe has no integer selected_take; specify --take.")
    matches = [take for take in recipe["takes"] if take["take"] == number]
    if len(matches) != 1:
        raise ValueError(f"Take {number} is not uniquely present in this recipe.")
    take = matches[0]
    report = compare_image_bytes(contents[take["hero"]], output_path.read_bytes(), max_mae, max_abs)
    graph = json.loads(contents[take["graph"]])
    inputs = {node["inputs"]["image"] for node in graph.values() if node["class_type"] == "LoadImage"}
    report["recipe"] = {
        "path": str(recipe_path.resolve()), "file_sha256": sha256(recipe_path.read_bytes()),
        "schema_version": recipe.get("schema_version", 1), "take": number,
        "archive_selected_take": recipe.get("selected_take"),
        "reference_member": take["hero"], "cloud_job": take.get("cloud_job"),
        "graph_member": take["graph"], "graph_sha256": sha256(contents[take["graph"]]),
        "input_sha256": {name: sha256(contents[name]) for name in sorted(inputs)},
        "working_source_sha256": sha256(contents["still.png"]),
        "archive_validation": {"ok": True, "integrity": validation.get("integrity"),
                               "warnings": validation["warnings"]},
    }
    report["output"]["path"] = str(output_path.resolve())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipe")
    parser.add_argument("output")
    parser.add_argument("--take", type=int)
    parser.add_argument("--max-mae", type=float, default=0.5)
    parser.add_argument("--max-abs", type=float, default=2.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = compare_recipe(args.recipe, args.output, args.take, args.max_mae, args.max_abs)
    except (OSError, ValueError, KeyError) as exc:
        report = {"comparison_status": "comparison_error", "errors": [str(exc)],
                  "within_tolerance": False, "execution_evidence": "Not checked."}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report.get("within_tolerance") else 1)


if __name__ == "__main__":
    main()
