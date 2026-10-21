"""Small CPU contract checks; no network or models. Run with unittest discovery."""
import sys, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ComfyUI-HeroFrameRig'))
import hfr_core as core


class EdgeCleanupTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.rgb=torch.rand((1,48,64,3),generator=torch.Generator().manual_seed(33))
        self.depth=np.full((1,48,64),.7,dtype=np.float32)

    def test_zero_bypasses_ownership_and_preserves_input(self):
        before=self.rgb.clone();depth=self.depth.copy()
        expected=core.render(self.rgb,self.depth,focus_curve=[.3],edge_cleanup=0)
        with patch.object(core.edges,'ownership',side_effect=AssertionError('new pass called')):
            actual=core.render(self.rgb,self.depth,focus_curve=[.3],edge_cleanup=0)
        self.assertTrue(torch.equal(expected,actual))
        self.assertTrue(torch.equal(self.rgb,before));np.testing.assert_array_equal(self.depth,depth)

    def test_flat_depth_is_untouched(self):
        a=core.render(self.rgb,self.depth,focus_curve=[.3],edge_cleanup=0)
        b=core.render(self.rgb,self.depth,focus_curve=[.3],edge_cleanup=.5)
        self.assertTrue(torch.equal(a,b))

    def test_finite_bounded_strength(self):
        for value in [-.1,1.1,float('nan'),float('inf')]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                core.render(self.rgb,self.depth,focus_curve=[.3],edge_cleanup=value)

    def test_solid_support_and_feather_are_local(self):
        d=torch.full((1,1,48,64),.2);d[:,:,14:34,20:42]=.85
        image=torch.zeros((1,3,48,64));image[:,:,:,43]=1
        owner,confidence=core.edges.ownership(d,image,.5)
        self.assertGreater(float(confidence.max()),0)
        self.assertTrue(bool(((confidence>0)&(confidence<1)).any()))
        self.assertEqual(float(confidence[0,0,0,0]),0)
        self.assertGreater(float(owner[0,0,23,43]),float(d[0,0,23,43]))


if __name__=='__main__':unittest.main()
