"""Serialized UI must never offer missing bundled media; no backend calls."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import app


class EmptyState(unittest.TestCase):
    def test_study_offers_follow_each_archive_independently(self):
        root = Path(app.__file__).resolve().parent
        image_archive = root / "examples/nola_study.zip"
        clip_archive = root / "examples/arrival/arrival_recipe.zip"
        real_is_file = Path.is_file
        for image_exists, clip_exists in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(image=image_exists, clip=clip_exists):
                def is_file(path):
                    if path == image_archive:
                        return image_exists
                    if path == clip_archive:
                        return clip_exists
                    return real_is_file(path)

                name = "_hfr_empty_state_fixture"
                spec = importlib.util.spec_from_file_location(name, root / "app.py")
                module = importlib.util.module_from_spec(spec)
                with patch.dict(sys.modules, {name: module}), patch.object(Path, "is_file", is_file), \
                        patch.object(app.backend, "client", side_effect=AssertionError("No jobs in UI checks")):
                    spec.loader.exec_module(module)
                    props = {c["props"].get("elem_id"): c["props"] for c in module.demo.config["components"]}
                    for elem_id, available in (("study-button", image_exists), ("stage-study-button", image_exists),
                                               ("clip-study-button", clip_exists), ("stage-motion-study-button", clip_exists)):
                        self.assertEqual(props[elem_id]["visible"], available, elem_id)
                        self.assertEqual(props[elem_id]["interactive"], available, elem_id)
                    self.assertEqual(props["stage-empty-hint"]["value"],
                                     "Drop your own image above, or open a saved image recipe.")
                    self.assertNotIn("saved study", props["frame-readout"]["value"])
                    if not image_exists and not clip_exists:
                        visible = " ".join(str(c.get("value", "")) for c in props.values() if c.get("visible", True))
                        for promise in ("Explore saved study", "Open the saved study", "open the saved clip study", "No clip? Try ours"):
                            self.assertNotIn(promise, visible)

    def test_missing_take_guidance_uses_user_media_or_recipe(self):
        with self.assertRaises(app.gr.Error) as caught:
            app.choose_take(None, None)
        self.assertIn("saved image recipe", str(caught.exception))
        self.assertNotIn("saved study", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
