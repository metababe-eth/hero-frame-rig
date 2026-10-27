"""Offline image-recipe trust boundary and continued-editing regressions."""
import copy
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np
import OpenEXR
from PIL import Image

import app
import image_recipe
import review_state


def png(size=(16, 12), color=128):
    stream = io.BytesIO()
    Image.new("RGB", size, (color, color, color)).save(stream, format="PNG")
    return stream.getvalue()


class ImageRecipeOpen(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "recipe.zip"
        self.graph = {"1": {"class_type": "LoadImage", "inputs": {"image": "still.png"}},
                      "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}
        self.recipe = {
            "app": "Hero Frame Rig", "schema_version": 2, "selected_take": 7,
            "frame": {"source": "generated", "size": [16, 12], "lens_mm_full_frame": 50,
                      "graph": "frame_graph.json", "engine": "moss", "image_format": "Landscape",
                      "resolution": "1K", "seed": 117, "prompt": "A saved scene",
                      "provenance": {"fixture": "saved frame producer"}},
            "optics": {"z_near_m": 1, "cinematic_multiplier": 2},
            "takes": [{"take": n, "hero": f"take{n:02d}_hero.png", "inputs": {"disparity": "map.png"},
                       "graph": f"take{n:02d}_develop_graph.json", "cloud_job": f"original-job-{n}",
                       "settings": {"mode": app.CINE if n == 7 else app.TRUE, "strength": 2 if n == 7 else 1,
                                    "f_stop": 2.8, "focus_disparity": .4 if n == 7 else .8,
                                    "focus_m": 2.5 if n == 7 else 1.25, "look": "film", "guide": 35,
                                    "subject_layer_enabled": False, "subject_label": 0},
                       "focus_point": [.2, .7] if n == 7 else None,
                       "provenance": {"fixture": f"producer {n}"}}
                      for n in (2, 7)]}
        depth_path = self.root / "fixture.exr"
        OpenEXR.File({}, {"Z": np.ones((12, 16), dtype=np.float32)}).write(str(depth_path))
        self.contents = {"depth.exr": depth_path.read_bytes()}
        for name in ("still.png", "sky.png", "subject_matte.png", "disparity.png", "map.png",
                     "take02_hero.png", "take07_hero.png"):
            self.contents[name] = png()
        for name in ("frame_graph.json", "take02_develop_graph.json", "take07_develop_graph.json"):
            self.contents[name] = json.dumps(self.graph).encode()

    def write(self, manifest=True, extra=None):
        contents = {**self.contents, "recipe.json": json.dumps(self.recipe).encode()}
        if manifest:
            contents["integrity.json"] = json.dumps({"algorithm": "sha256", "files": {
                key: hashlib.sha256(value).hexdigest() for key, value in contents.items()}}).encode()
        with zipfile.ZipFile(self.archive, "w", zipfile.ZIP_DEFLATED) as z:
            for name, value in contents.items():
                z.writestr(name, value)
            if extra:
                z.writestr(*extra)
        return self.archive

    def load(self):
        return image_recipe.load_image_recipe(self.write(), self.root / "opened")

    def test_roundtrip_restores_all_takes_selection_original_settings_and_generation_metadata(self):
        state = self.load()
        self.assertEqual([t["n"] for t in state["takes"]], [2, 7])
        self.assertEqual((state["selected_take"], state["focus"], state["label"]), (7, .4, 0))
        self.assertEqual(state["focus_point"], [.2, .7])
        self.assertEqual(state["engine"], "moss")
        self.assertEqual(state["frame_provenance"], self.recipe["frame"]["provenance"])
        exported = app.save_recipe(state)
        reopened = image_recipe.load_image_recipe(exported, self.root / "reopened")
        self.assertEqual(reopened["selected_take"], 7)
        self.assertEqual(reopened["image_format"], "Landscape")
        self.assertEqual(reopened["resolution"], "1K")
        for before, after in zip(state["takes"], reopened["takes"]):
            for key in ("settings", "graph", "job", "provenance", "focus_point"):
                self.assertEqual(before[key], after[key])
            self.assertEqual(Path(before["hero"]).read_bytes(), Path(after["hero"]).read_bytes())
        # A shared input is retained once, with no duplicate ZIP entries.
        with zipfile.ZipFile(exported) as z:
            self.assertEqual(len(z.namelist()), len(set(z.namelist())))
            self.assertEqual(z.namelist().count("map.png"), 1)

    def test_ui_open_does_not_contact_backend_or_execute_saved_graph(self):
        arbitrary = {"1": {"class_type": "NeverExecuteThisSavedNode", "inputs": {"text": "https://example.invalid"}}}
        self.contents["take07_develop_graph.json"] = json.dumps(arbitrary).encode()
        path = self.write()
        # Keep the importer-created session directory inside this test's temporary root.
        with patch.object(app, "_client", side_effect=AssertionError("backend contacted")), \
             patch.object(app.backend, "client", side_effect=AssertionError("backend contacted")), \
             patch.object(app, "load_image_recipe", side_effect=lambda p: image_recipe.load_image_recipe(p, self.root / "ui")):
            outputs = app.open_image_recipe(str(path))
        self.assertEqual(len(outputs), 30)
        self.assertIn("no backend job", outputs[5])
        self.assertEqual(outputs[4]["takes"][-1]["graph"], arbitrary)
        self.assertEqual(outputs[9]["value"], 2.8)
        self.assertEqual(outputs[10:13], ("film", app.CINE, False))
        self.assertEqual(outputs[29], 35)

    def test_legacy_missing_manifest_selection_and_optional_settings(self):
        self.recipe.pop("schema_version")
        self.recipe.pop("selected_take")
        self.recipe["frame"].pop("image_format")
        self.recipe["frame"].pop("resolution")
        for take in self.recipe["takes"]:
            take["settings"].pop("subject_layer_enabled")
            take["settings"].pop("guide")
            take["settings"]["mode"] = "Cinematic"
        state = image_recipe.load_image_recipe(self.write(manifest=False), self.root / "legacy")
        self.assertFalse(state["imported_recipe"]["integrity_verified"])
        self.assertEqual(state["selected_take"], 7)
        self.assertNotIn("image_format", state)
        self.assertNotIn("subject_layer_enabled", state["takes"][-1]["settings"])
        self.assertEqual(app._outputs(state)[11], app.CINE)

    def test_new_development_uses_new_graph_and_noncolliding_take_number(self):
        state = self.load()
        client = Mock()
        client.workflows.from_json.side_effect = lambda graph: Mock(graph=graph)
        with patch.object(app, "_run_many", return_value=[SimpleNamespace(id="mock-only")]), \
             patch.object(app, "_save", return_value=state["still"]):
            developed = app._develop(client, state, [(1.4, "digital", app.TRUE)], False, 0)
        self.assertEqual(developed["takes"][-1]["n"], 8)
        self.assertEqual(len(developed["takes"]), 3)
        self.assertNotEqual(developed["takes"][-1]["graph"], state["takes"][-1]["graph"])
        self.assertEqual(Path(state["takes"][-1]["hero"]).read_bytes(), self.contents["take07_hero.png"])
        self.assertTrue(Path(developed["takes"][-1]["files"]["disparity"]).is_file())

    def test_changing_selection_restores_focus_and_marker(self):
        state = self.load()
        older = review_state.select_take(state, 2)
        self.assertEqual(older["focus"], .8)
        self.assertIsNone(older["focus_point"])
        self.assertEqual(state["selected_take"], 7)

    def test_existing_destination_is_never_deleted(self):
        directory = self.root / "keep"
        directory.mkdir()
        (directory / "precious.txt").write_text("preserve me")
        with self.assertRaises(ValueError):
            image_recipe.load_image_recipe(self.write(), directory)
        self.assertEqual((directory / "precious.txt").read_text(), "preserve me")

    def test_pinned_study_loader_still_rejects_arbitrary_archives(self):
        with self.assertRaisesRegex(ValueError, "registered bundled study"):
            review_state.load_study(self.write(), self.root / "not-study")

    def test_unsafe_paths_and_aliases_rejected_before_extraction(self):
        for name in ("../escape", "/absolute", "C:/drive", "a\\b", "a:stream", "con.txt",
                     "folder./x", "folder /x", "a//b", "Still.png", "a/../b"):
            with self.subTest(name=name):
                path = self.write(extra=(name, b"bad"))
                with self.assertRaises(ValueError):
                    image_recipe.load_image_recipe(path, self.root / "opened")
                self.assertFalse((self.root / "opened").exists())

    def test_symlink_member_rejected(self):
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaisesRegex(ValueError, "unsupported"):
            image_recipe.load_image_recipe(self.write(extra=(info, b"target")), self.root / "opened")

    def test_declared_size_and_member_limits_are_enforced(self):
        path = self.write()
        for limit in ("MAX_MEMBER_BYTES", "MAX_MEMBERS", "MAX_ARCHIVE_BYTES", "MAX_JSON_BYTES"):
            with self.subTest(limit=limit), patch.object(image_recipe, limit, 1):
                with self.assertRaises(ValueError):
                    image_recipe.load_image_recipe(path, self.root / "opened")

    def test_hash_mismatch_and_missing_manifest_rejected(self):
        path = self.write()
        with zipfile.ZipFile(path) as z:
            contents = {name: z.read(name) for name in z.namelist()}
        contents["still.png"] = png(color=129)
        with zipfile.ZipFile(path, "w") as z:
            for name, data in contents.items():
                z.writestr(name, data)
        with self.assertRaisesRegex(ValueError, "checksum"):
            image_recipe.load_image_recipe(path, self.root / "opened")
        with self.assertRaisesRegex(ValueError, "integrity.json"):
            image_recipe.load_image_recipe(self.write(manifest=False), self.root / "opened")

    def test_invalid_selected_take_settings_graphs_and_images_rejected(self):
        good_recipe, good_contents = copy.deepcopy(self.recipe), dict(self.contents)
        mutations = (
            lambda: self.recipe.update(selected_take=99),
            lambda: self.recipe["takes"][1]["settings"].update(focus_disparity=float("nan")),
            lambda: self.recipe["takes"][1]["settings"].update(f_stop=0),
            lambda: self.recipe["takes"][1]["settings"].update(subject_layer_enabled="false"),
            lambda: self.recipe["takes"][1].update(focus_point=[2, 0]),
            lambda: self.recipe["frame"].update(size=[1000000, 1000000]),
            lambda: self.contents.update({"still.png": b"bad image"}),
            lambda: self.contents.update({"disparity.png": png(size=(15, 12))}),
            lambda: self.contents.update({"depth.exr": b"invalid EXR"}),
            lambda: self.contents.update({"take07_develop_graph.json": json.dumps({"1": {"class_type": "LoadImage", "inputs": {"image": "https://bad.invalid/image.png"}}}).encode()}),
            lambda: self.contents.update({"take07_develop_graph.json": json.dumps({"1": {"class_type": "Node", "inputs": {"image": ["9", 0]}}}).encode()}),
            lambda: self.contents.update({"take07_develop_graph.json": json.dumps({"1": {"class_type": "Node", "inputs": {"image": ["1", 0]}}}).encode()}),
            lambda: self.contents.update({"frame_graph.json": b'{"1": {}, "1": {}}'}),
        )
        for index, mutate in enumerate(mutations):
            self.recipe, self.contents = copy.deepcopy(good_recipe), dict(good_contents)
            mutate()
            with self.subTest(index=index), self.assertRaises(ValueError):
                image_recipe.load_image_recipe(self.write(), self.root / "opened")
            self.assertFalse((self.root / "opened").exists())

    def test_nonfinite_depth_rejected_and_new_directory_cleaned(self):
        path = self.root / "nan.exr"
        OpenEXR.File({}, {"Z": np.full((12, 16), np.nan, dtype=np.float32)}).write(str(path))
        self.contents["depth.exr"] = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "finite"):
            image_recipe.load_image_recipe(self.write(), self.root / "opened")
        self.assertFalse((self.root / "opened").exists())

    def test_unrecorded_graph_asset_cannot_disappear_on_resave(self):
        self.contents["extra.png"] = png()
        graph = copy.deepcopy(self.graph)
        graph["1"]["inputs"]["image"] = "extra.png"
        self.contents["take07_develop_graph.json"] = json.dumps(graph).encode()
        with self.assertRaisesRegex(ValueError, "bindings"):
            image_recipe.load_image_recipe(self.write(), self.root / "opened")
        # Once explicitly recorded as a take input, it is safely retained.
        self.recipe["takes"][1]["inputs"]["extra"] = "extra.png"
        state = self.load()
        saved = app.save_recipe(state)
        with zipfile.ZipFile(saved) as z:
            self.assertEqual(z.read("extra.png"), self.contents["extra.png"])
        image_recipe.load_image_recipe(saved, self.root / "reopened")

    def test_gradio_image_recipe_event_is_private_and_restores_existing_outputs(self):
        config = app.demo.config
        event = next(event for event in config["dependencies"]
                     if any(target == (app.image_recipe_open._id, "click") or target == [app.image_recipe_open._id, "click"]
                            for target in event["targets"]))
        self.assertEqual(event["inputs"], [app.image_recipe_upload._id])
        self.assertEqual(event["outputs"], [component._id for component in app.outs])
        self.assertEqual(event["api_visibility"], "private")
        self.assertEqual(sorted(event["api_name"] for event in config["dependencies"]
                                if event.get("api_visibility") == "public"), ["direct_clip", "shoot_still"])

    def test_real_five_take_royal_praline_recipe_roundtrip(self):
        fixture = Path(__file__).resolve().parents[3] / "own-media-20261023/photo_demo/ROYAL_PRALINE_FIVE_TAKES_recipe.zip"
        if not fixture.is_file():
            self.skipTest("Private Royal Praline evidence stays outside the source repository.")
        with patch.object(app, "_client", side_effect=AssertionError("no inference")):
            state = image_recipe.load_image_recipe(fixture, self.root / "royal")
            exported = app.save_recipe(state)
            reopened = image_recipe.load_image_recipe(exported, self.root / "royal_again")
        self.assertEqual(reopened["selected_take"], 5)
        self.assertEqual(len(reopened["takes"]), 5)
        self.assertEqual((reopened["width"], reopened["height"]), (1200, 1600))
        self.assertEqual(hashlib.sha256(Path(reopened["original_file"]).read_bytes()).hexdigest(),
                         "403b5c7f3fb28d9bfc6d03d83cc358ee7e299ad5081d68a0aadf5a1bede916f2")
        with zipfile.ZipFile(fixture) as before, zipfile.ZipFile(exported) as after:
            original_recipe = json.loads(before.read("recipe.json"))
            for name in before.namelist():
                if name.endswith((".png", ".exr", ".jpg")):
                    self.assertEqual(before.read(name), after.read(name), name)
            for expected, actual in zip(original_recipe["takes"], reopened["takes"]):
                self.assertEqual(expected["settings"], actual["settings"])
                self.assertEqual(expected["cloud_job"], actual["job"])


if __name__ == "__main__":
    unittest.main()
