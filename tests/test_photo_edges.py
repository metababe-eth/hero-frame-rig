"""Still-only edge policy: meaningful invariants and a synthetic occlusion edge."""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import optics
import round_lens as lens


class PhotoEdges(unittest.TestCase):
    def test_zero_blur_is_zero_and_inputs_unchanged(self):
        d = np.random.default_rng(5).random((48, 64)).astype(np.float32)
        before = d.copy()
        self.assertTrue(all(not w.any() for w in lens.photo_weight_arrays(d, 0, .4)))
        self.assertTrue(np.array_equal(d, before))

    def test_focused_flat_plane_is_exactly_sharp(self):
        d = np.full((48, 64), .6, np.float32)
        self.assertTrue(all(not w.any() for w in lens.photo_weight_arrays(d, 45, .6)))

    def test_flat_defocused_plane_retains_legacy_weights(self):
        for value, focus in [(.8, .2), (.2, .8)]:
            d = np.full((96, 128), value, np.float32)
            arrays = lens.photo_weight_arrays(d, 11.52, focus)
            f = max(0, (abs(value - focus) - .01) / (.99 + 1e-6))
            for n, array in enumerate(arrays):
                wanted = round(np.clip((f - n / 5) * 5, 0, 1) * 255)
                self.assertTrue(np.all(array == wanted))

    def test_foreground_spill_is_soft_local_and_monotone(self):
        d = np.full((96, 160), .2, np.float32)
        d[:, 80:] = .8
        array = lens.photo_weight_arrays(d, 11.52, .2)[0]
        line = array[48]
        self.assertTrue(np.all(np.diff(line.astype(int)) >= 0))
        self.assertEqual(int(line[0]), 0)
        self.assertEqual(int(line[-1]), 255)
        self.assertGreater(len(np.unique(line[:80])), 3)
        sigma = lens.describe(11.52, 160, 96)['levels'][0]['matched_sigma_px']
        self.assertFalse(line[:80-math.ceil(3*sigma)].any())

    def test_far_blur_does_not_spill_into_in_focus_foreground(self):
        d = np.full((96, 160), .2, np.float32)
        d[:, 80:] = .8
        for array in lens.photo_weight_arrays(d, 11.52, .8):
            self.assertFalse(array[:, 80:].any())

    def test_saved_arrays_roundtrip_and_version_is_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            d = np.full((96, 128), .2, np.float32); d[:,64:] = .8
            path = Path(td)/'raw.png'; optics.save_disparity(d, path)
            arrays = lens.photo_weight_arrays(optics.load_disparity(path), 11.52, .2)
            paths = lens.prepare_photo_weights(path, Path(td)/'weight', 11.52, .2)
            self.assertEqual(len(paths), 5)
            for array, p in zip(arrays, paths):
                self.assertTrue(np.array_equal(array, np.array(Image.open(p))))
            old, ref = lens.graph(['30',0],128,96,11.52,paths)
            new, newref = lens.photo_graph(['30',0],128,96,11.52,paths)
            self.assertEqual(ref,newref)
            self.assertEqual(old['round_scale']['_meta']['title'], lens.VERSION+' / area scale')
            self.assertEqual(new['round_scale']['_meta']['title'], lens.PHOTO_VERSION+' / area scale')
            for node in old:
                self.assertEqual(old[node]['inputs'],new[node]['inputs'])
            self.assertEqual(lens.describe_photo(11.52,128,96)['highlight_recovery'], 0)

    def test_invalid_inputs_fail(self):
        for d in [np.array([[np.nan]*2]*2), np.full((2,2), 1.1), np.zeros((1,2)),np.zeros((2,2,2))]:
            with self.assertRaises(ValueError):lens.photo_weight_arrays(d,11.52,.4)
        for blur,focus in [(float('nan'),.4),(51,.4),(11.52,-.1),(11.52,True)]:
            with self.assertRaises(ValueError):lens.photo_weight_arrays(np.zeros((2,2)),blur,focus)


if __name__ == '__main__':unittest.main()
