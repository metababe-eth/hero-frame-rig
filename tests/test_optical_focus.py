"""Optical quality/compatibility contracts. CPU only, no jobs or models."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ComfyUI-HeroFrameRig'))
import hfr_core as core
from hfr_core import optical
import focus_replay


class OpticalFocusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def render(self,rgb,depth,focus,**options):
        return core.render(rgb,depth,focus_curve=[focus]*len(rgb),lens_mode='optical',
                           depth_smoothing=0,edge_cleanup=0,**options)

    def test_old_mode_remains_default_and_bypasses_optical(self):
        rgb=torch.rand((1,40,64,3),generator=torch.Generator().manual_seed(44))
        depth=np.full((1,40,64),.7,np.float32)
        expected=core._render_legacy(rgb,depth,focus_curve=[.3])
        with patch.object(optical,'render',side_effect=AssertionError('optical invoked')):
            self.assertTrue(torch.equal(expected,core.render(rgb,depth,focus_curve=[.3])))
            self.assertTrue(torch.equal(expected,core.render(rgb,depth,focus_curve=[.3],lens_mode='legacy')))

    def test_focused_pixels_and_zero_strength_keep_source_colour(self):
        rgb=torch.rand((1,32,48,3),generator=torch.Generator().manual_seed(3))
        depth=np.full((1,32,48),.5,np.float32)
        for focus,amount in ((.5,50),(.1,0)):
            result=self.render(rgb,depth,focus,blur_strength=amount)
            self.assertLess(float((result-rgb).abs().max()),2e-6)

    def test_disc_energy_and_radial_shape(self):
        point=torch.zeros((1,1,81,81)); point[0,0,40,40]=1
        image=optical.disc_blur(point,8.)
        self.assertAlmostEqual(float(image.sum()),1.,places=5)
        self.assertAlmostEqual(float(image[0,0,40,40]),float(image[0,0,40,45]),places=6)
        self.assertLess(float(image[0,0,40,50]),1e-6)
        self.assertLess(float((image-image.transpose(-1,-2)).abs().max()),1e-6)

    def test_linear_light_preserves_light_energy(self):
        image=torch.zeros((1,3,81,81)); image[:,:,:,40:]=1
        linear=optical.disc_blur(optical._to_linear(image),6)
        output=optical._to_srgb(linear)
        srgb_blur=optical.disc_blur(image,6)
        self.assertGreater(float(output[0,0,40,39]),float(srgb_blur[0,0,40,39])+.15)

    def test_constant_light_does_not_darken_at_depth_edges(self):
        rgb=torch.full((1,64,96,3),.72)
        depth=np.full((1,64,96),.15,np.float32); depth[:,:,32:64]=.9
        out=self.render(rgb,depth,.2,aperture=1.4,focal_length_mm=85)
        self.assertLess(float((out-rgb).abs().max()),2e-5)

    def test_sharp_foreground_does_not_contaminate_background(self):
        rgb=torch.zeros((1,80,120,3)); rgb[:,:,:,2]=.6; rgb[:,:,40:80]=torch.tensor([1.,0.,0.])
        depth=np.full((1,80,120),.15,np.float32); depth[:,:,40:80]=.9
        out=self.render(rgb,depth,.9,focal_length_mm=85,aperture=1.4)
        self.assertLess(float(out[0,:,20:39,0].max()),1e-4)
        self.assertGreater(float(out[0,20:60,42:78,0].min()),.999)

    def test_blurred_foreground_spills_over_background_without_dark_rim(self):
        rgb=torch.zeros((1,80,120,3)); rgb[:,:,:,2]=.6; rgb[:,:,40:80]=torch.tensor([1.,0.,0.])
        depth=np.full((1,80,120),.15,np.float32); depth[:,:,40:80]=.9
        out=self.render(rgb,depth,.15,focal_length_mm=85,aperture=1.4)
        self.assertGreater(float(out[0,40,39,0]),.1)
        self.assertGreater(float(out[0,40,40,2]),.05)
        self.assertTrue(torch.isfinite(out).all()); self.assertGreaterEqual(float(out.min()),0)
        self.assertLessEqual(float(out.max()),1)

    def test_tiny_far_highlight_cannot_fill_a_hidden_surface(self):
        rgb=torch.full((1,80,120,3),.02); rgb[:,40,60]=1.
        depth=np.broadcast_to(np.linspace(.3,.9,120,dtype=np.float32),(1,80,120)).copy()
        depth[:,40,60]=0.
        out=self.render(rgb,depth,.15,focal_length_mm=85,aperture=1.4)
        # Outside the maximum 5.4px source disc, sparse background support must
        # not create a new light patch. This caught the lantern's white ghost.
        yy,xx=torch.meshgrid(torch.arange(80),torch.arange(120),indexing='ij')
        ring=((xx-60)**2+(yy-40)**2>=7**2)&((xx-60)**2+(yy-40)**2<=9**2)
        self.assertLess(float(out[0][ring].max()),.035)

    def test_aperture_is_monotonic_and_near_plane_is_gentle(self):
        d=torch.tensor([.5,.505,.51,.6,.9])
        radii=[optical.coc_radius(d,.5,720,aperture=n) for n in (1.4,2.8,5.6)]
        self.assertEqual(float(radii[1][0]),0)
        self.assertEqual(float(radii[1][1]),0)
        self.assertTrue(bool((radii[0]>=radii[1]).all()))
        self.assertTrue(bool((radii[1]>=radii[2]).all()))
        self.assertTrue(bool((radii[1][1:]>=radii[1][:-1]).all()))

    def test_depth_filter_reduces_noise_and_resets_on_moving_edges(self):
        rng=np.random.default_rng(19)
        d=(.5+rng.normal(0,.005,(11,12,16))).astype(np.float32)
        rgb=np.full((11,12,16,3),128,np.uint8)
        filtered=optical.smooth_depth(d,rgb,.5)
        self.assertLess(float(filtered.var()),float(d.var())*.8)
        d[5:]=.9; rgb[5:]=240
        filtered=optical.smooth_depth(d,rgb,.5)
        np.testing.assert_array_equal(filtered[5],d[5])
        np.testing.assert_array_equal(optical.smooth_depth(d,rgb,0),d)

    def test_plan_is_longer_eased_pull_and_exact_endpoints(self):
        d=np.broadcast_to(np.linspace(.9,.1,64,dtype=np.float32),(49,48,64)).copy()
        plan=core.make_plan(d,.2,.5,.8,.5,arrival=4/9,lens_mode='optical')
        old=core.make_plan(d,.2,.5,.8,.5,arrival=4/9)
        curve=np.array(plan['render_curve']); before=np.array(old['render_curve'])
        self.assertAlmostEqual(curve[0],core.sample(d,.2,.5,0))
        self.assertAlmostEqual(curve[-1],core.sample(d,.8,.5,48))
        self.assertGreater(np.count_nonzero(np.diff(curve)),np.count_nonzero(np.diff(before))*2)
        self.assertLess(float(np.abs(np.diff(curve)).max()),float(np.abs(np.diff(before)).max()))
        self.assertTrue(np.all(np.diff(curve)<=1e-10))
        self.assertGreaterEqual(len(curve)-np.flatnonzero(np.diff(curve))[-1],5)

    def test_optical_recipe_keeps_exact_values_and_old_recipe_has_no_new_mode(self):
        plan=dict(width=64,height=48,points=[[16,22],[48,26]],middle='Hold the start',
                  blur=50,pull_lands_at_percent=70,guide=0,fps=25,render_curve=[.8,.5,.2])
        old=focus_replay.workflow(plan)['4']['inputs']
        self.assertNotIn('lens_mode',old)
        chosen=optical.settings(aperture=3.1,depth_smoothing=.27,breathing=.012,near_m=.6,far_m=12.)
        plan.update(chosen)
        replay=focus_replay.workflow(json.loads(json.dumps(plan)))['4']['inputs']
        for key,value in chosen.items(): self.assertEqual(replay[key],value)

    def test_invalid_modes_settings_and_curve_fail_clearly(self):
        rgb=torch.zeros((1,16,16,3)); d=np.full((1,16,16),.5,np.float32)
        for options in ({'lens_mode':'unknown'},{'lens_mode':'optical','aperture':0},
                        {'lens_mode':'optical','far_m':.5,'near_m':.5},
                        {'lens_mode':'optical','breathing':.03}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                core.render(rgb,d,focus_curve=[.5],**options)
        with self.assertRaises(ValueError): self.render(rgb,d,float('nan'))

    def test_no_input_mutation_and_breathing_off_is_exact(self):
        rgb=torch.randint(0,256,(1,32,48,3),dtype=torch.uint8,generator=torch.Generator().manual_seed(8))
        depth=np.full((1,32,48),.5,np.float32); original=rgb.clone(); original_d=depth.copy()
        out=core.render(rgb,depth,lens_mode='optical',focus_curve=[.5])
        no_breath=core.render(rgb,depth,lens_mode='optical',focus_curve=[.5],breathing=0)
        self.assertTrue(torch.equal(out,no_breath)); self.assertTrue(torch.equal(rgb,original))
        np.testing.assert_array_equal(depth,original_d)

    def test_explicit_curve_is_used_and_breathing_is_bounded_crop(self):
        ramp=torch.linspace(0,1,96)[None,None,:,None].expand(3,64,96,3).clone()
        depth=np.full((3,64,96),.5,np.float32)
        curve=[.5,.7,.9]
        with patch.object(optical,'make_plan',side_effect=AssertionError('replanned saved curve')):
            a=core.render(ramp,depth,lens_mode='optical',focus_curve=curve,breathing=0)
            b=core.render(ramp,depth,lens_mode='optical',focus_curve=curve,breathing=.02)
        self.assertTrue(torch.equal(a[0],b[0]))
        self.assertGreater(float((a[-1]-b[-1]).abs().max()),.001)
        self.assertLess(float((a[-1]-b[-1]).abs().max()),.02)

    def test_node_exposes_modes_and_runs_saved_recipe(self):
        import types
        spec=importlib.util.spec_from_file_location('optical_test_pack',ROOT/'ComfyUI-HeroFrameRig/__init__.py',
            submodule_search_locations=[str(ROOT/'ComfyUI-HeroFrameRig')])
        pack=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pack;spec.loader.exec_module(pack)
        node=pack.NODE_CLASS_MAPPINGS['HFRFocusPull']()
        self.assertEqual(node.INPUT_TYPES()['optional']['lens_mode'][0],['legacy','optical'])
        self.assertEqual(list(node.INPUT_TYPES()['optional'])[:3],
                         ['edge_cleanup','focus_curve_json','head_targets_json'])
        class Progress:
            def __init__(self,n): self.n=n
            def update_absolute(self,i,n): pass
        rgb=torch.full((3,16,24,3),.6); d=torch.full((3,16,24),.5); curve=[.5,.6,.7]
        chosen=optical.settings(aperture=3.1,depth_smoothing=.27,breathing=.012)
        with patch.dict(sys.modules,{'comfy':types.ModuleType('comfy'),
            'comfy.utils':types.SimpleNamespace(ProgressBar=Progress)}):
            actual=node.execute(rgb,d,focus_curve_json=json.dumps(curve),**chosen)[0]
        expected=core.render(rgb,d,focus_curve=curve,**chosen)
        self.assertTrue(torch.equal(expected,actual))

    def test_shipped_cached_example_executes_optical_settings(self):
        import types
        spec=importlib.util.spec_from_file_location('optical_example_pack',ROOT/'ComfyUI-HeroFrameRig/__init__.py',
            submodule_search_locations=[str(ROOT/'ComfyUI-HeroFrameRig')])
        pack=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pack;spec.loader.exec_module(pack)
        graph=json.loads((ROOT/'ComfyUI-HeroFrameRig/example_workflows/cached_depth_cpu.json').read_text())
        node=pack.NODE_CLASS_MAPPINGS[graph['4']['class_type']]()
        options={k:v for k,v in graph['4']['inputs'].items() if k not in ('images','depth')}
        self.assertEqual(options['lens_mode'],'optical')
        self.assertEqual(options['aperture'],2.8)
        self.assertAlmostEqual(50+45*options['arrival'],70)
        rgb=torch.rand((9,24,32,3),generator=torch.Generator().manual_seed(43))
        d=torch.linspace(.95,.1,32)[None,None,:].expand(9,24,32)
        progress=lambda n:types.SimpleNamespace(update_absolute=lambda *args:None)
        with patch.dict(sys.modules,{'comfy':types.ModuleType('comfy'),
             'comfy.utils':types.SimpleNamespace(ProgressBar=progress)}):
            result=node.execute(rgb,d,**options)[0]
        options.pop('depth_nudge_json')
        self.assertTrue(torch.equal(result,core.render(rgb,d,**options)))
        self.assertEqual(pack.__version__,core.__version__)


if __name__=='__main__': unittest.main()
