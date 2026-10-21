"""Record the actual application sources and export environment, without secrets."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

SOURCE_FILES = ("app.py", "rig_workflow.py", "round_lens.py", "guide_eye.py", "optics.py", "frame_input.py",
                "review_state.py", "provenance.py", "validate_recipe.py",
                "animation_workflow.py", "animation_input.py", "rack_focus.py",
                "workflows/ltx_sharp_recovered.json",
                "requirements.txt", "requirements-local.txt", "README.md", "DEPENDENCIES.md")


def capture():
    root = Path(__file__).resolve().parent
    packages = {}
    for name in ("gradio", "comfy-sdk", "pillow", "numpy", "OpenEXR", "opencv-python-headless"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    return {
        "source_files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                         for name in SOURCE_FILES if (root / name).is_file()},
        "packages": packages,
        "base_revision": "740392f22e6c88279c6ed57abdad02b49ea8da98",
        "scope": "Application source hashes and export environment. Base revision is the "
                 "pre-patch ancestor, not the deployed commit. Cloud worker package/model "
                 "hashes are not exposed here; see DEPENDENCIES.md.",
    }
