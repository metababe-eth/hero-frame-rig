"""User-facing focus controls: local, bounded, replayable and opt-in."""
from pathlib import Path
import sys,json,unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ComfyUI-HeroFrameRig'))
import hfr_core as core
from hfr_core import directing,optical
import focus_replay


class FocusDirectingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_presets_set_exact_aperture_and_timing(self):
        self.assertEqual(directing.preset('Cinematic')['aperture'],2.8)
        self.assertEqual(directing.preset('Cinematic')['arrival'],4/9)
        self.assertGreater(directing.preset('Subtle')['aperture'],directing.preset('Dramatic')['aperture'])
        for name in directing.PRESETS:
            p=directing.preset(name);self.assertEqual(directing.preset_name(p['aperture'],p['landing']),name)
        self.assertIsNone(directing.preset_name(3.2,73))

    def test_warning_uses_actual_endpoint_depths(self):
        d=np.broadcast_to(np.linspace(.99,.88,64,dtype=np.float32),(9,32,64)).copy()
        self.assertTrue(directing.depth_warning(d,0,.5,1,.5)['close'])
        d[:,:,-8:]=.2
        self.assertFalse(directing.depth_warning(d,0,.5,1,.5)['close'])

    def test_nudge_only_changes_depth_supported_region(self):
        d=np.full((3,64,96),.82,np.float32);d[:,:,:20]=.99;d[:,:,65:]=.91
        before=d.copy();cfg=dict(x=.45,y=.3,amount=-.2,radius=.5)
        out=directing.nudge_depth(d,cfg)
        np.testing.assert_array_equal(d,before)
        np.testing.assert_array_equal(out[:,:,:20],d[:,:,:20])
        np.testing.assert_array_equal(out[:,:,65:],d[:,:,65:])
        self.assertAlmostEqual(float(out[0,19,43]),.62,places=5)
        self.assertGreater(float(out[0,-1,-1]),.9)
        self.assertTrue(((out-d)>=-.200001).all())

    def test_foreground_crossing_region_is_not_dragged_with_background(self):
        d=np.full((3,64,96),.82,np.float32);d[1,10:40,35:65]=.98
        out=directing.nudge_depth(d,dict(x=.5,y=.3,anchor_frame=0,amount=-.2,radius=.5))
        np.testing.assert_array_equal(out[1,10:40,35:65],d[1,10:40,35:65])

    def test_no_nudge_keeps_part1_processing_identical(self):
        rgb=torch.rand((3,32,48,3),generator=torch.Generator().manual_seed(20))
        d=np.full((3,32,48),.7,np.float32);curve=[.8,.5,.2]
        with patch.object(directing,'nudge_depth',side_effect=AssertionError('unexpected correction')):
            a=core.render(rgb,d,lens_mode='optical',focus_curve=curve)
        b=core.render(rgb,d,lens_mode='optical',focus_curve=curve,depth_nudge=dict(amount=0))
        self.assertTrue(torch.equal(a,b))

    def test_nudge_validation_and_legacy_fail_clearly(self):
        d=np.full((3,32,48),.7,np.float32)
        for cfg in ({'amount':.31},{'radius':0},{'x':float('nan')},{'anchor_frame':3},{'anchor_depth':2}):
            with self.assertRaises(ValueError):directing.nudge_depth(d,cfg)
        with self.assertRaises(ValueError):core.render(torch.zeros((3,32,48,3)),d,focus_curve=[.5]*3,depth_nudge=dict(amount=-.1))

    def test_recipe_serializes_resolved_nudge_with_original_depth(self):
        d=np.full((3,32,48),.82,np.float32)
        nudge=directing.resolve_nudge(d,dict(x=.4,y=.3,amount=-.2,radius=.5))
        plan=dict(width=48,height=32,points=[[10,12],[34,22]],middle='Hold the start',
                  blur=50,pull_lands_at_percent=70,guide=0,fps=24,render_curve=[.9,.7,.5],
                  **optical.settings(),depth_nudge=nudge)
        graph=focus_replay.workflow(json.loads(json.dumps(plan)))
        self.assertEqual(json.loads(graph['4']['inputs']['depth_nudge_json']),nudge)
        self.assertEqual(graph['5']['inputs']['fps'],24)

if __name__=='__main__':unittest.main()
