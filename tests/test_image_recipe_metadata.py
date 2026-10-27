"""Exercise real recipe ZIP writing/reopening with optional generation metadata."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

import app
import review_state


class ImageRecipeMetadata(unittest.TestCase):
    def state(self, root, source='generated', **metadata):
        for name in ('still.png', 'sky.png', 'subject_matte.png', 'disparity.png', 'hero.png'):
            Image.new('RGB', (16, 16), (110, 120, 130)).save(root / name)
        (root / 'depth.exr').write_bytes(b'fixture: this test does not estimate depth')
        settings = {'mode': 'Cinematic', 'strength': 1, 'f_stop': 1.4, 'focus_m': 2,
                    'focus_disparity': .5, 'look': 'digital', 'subject_layer_enabled': False}
        take = {'n': 4, 'hero': str(root / 'hero.png'), 'job': 'fixture-develop',
                'settings': settings, 'files': {}, 'graph': {}}
        return {'dir': str(root), 'source': source, 'lens': 50, 'width': 16, 'height': 16,
                'z_near': 1, 'cine': 1, 'frame_job': 'fixture-frame', 'frame_graph': {},
                'still': str(root / 'still.png'), 'sky': str(root / 'sky.png'),
                'matte': str(root / 'subject_matte.png'), 'disparity': str(root / 'disparity.png'),
                'depth_exr': str(root / 'depth.exr'), 'takes': [take], **metadata}

    def save_and_reopen(self, state, root):
        archive = Path(app.save_recipe(state))
        with zipfile.ZipFile(archive) as z:
            recipe = json.loads(z.read('recipe.json'))
        with patch.object(review_state, 'BUNDLED_STUDY', archive), \
             patch.object(review_state, 'NOLA_RECIPE_SHA256', hashlib.sha256(archive.read_bytes()).hexdigest()):
            reopened = review_state.load_study(archive, root / 'reopened')
        return recipe, reopened

    def test_generated_format_and_size_survive_actual_zip_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recipe, reopened = self.save_and_reopen(
                self.state(root, image_format='Square', resolution='2K'), root)
            self.assertEqual(recipe['frame']['image_format'], 'Square')
            self.assertEqual(recipe['frame']['resolution'], '2K')
            self.assertEqual(reopened['image_format'], 'Square')
            self.assertEqual(reopened['resolution'], '2K')

    def test_legacy_generated_recipe_does_not_invent_missing_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recipe, reopened = self.save_and_reopen(self.state(root), root)
            for field in ('image_format', 'resolution'):
                self.assertNotIn(field, recipe['frame'])
                self.assertNotIn(field, reopened)

    def test_uploaded_image_does_not_acquire_generation_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            recipe, reopened = self.save_and_reopen(
                self.state(root, source='uploaded', image_format='Portrait', resolution='2K'), root)
            for field in ('image_format', 'resolution'):
                self.assertNotIn(field, recipe['frame'])
                self.assertNotIn(field, reopened)


if __name__ == '__main__':
    unittest.main()
