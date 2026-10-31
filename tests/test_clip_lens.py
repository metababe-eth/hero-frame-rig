"""Real local recipe round-trip, controls and 24fps cache without backend work."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import clip_focus,clip_lens,clip_flow,focus_replay,backend
from hfr_core import optical


class ClipLensTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='hfr_lens_test_')
        self.root=Path(self.temp.name)
        original=tempfile.mkdtemp
        self.temp_patch=patch.object(clip_focus.tempfile,'mkdtemp',side_effect=lambda **kw:original(dir=self.root,**kw))
        self.temp_patch.start()
        rgb=np.broadcast_to(np.linspace(0,1,96,dtype=np.float32)[None,None,:,None],(9,64,96,3)).copy()
        source=self.root/'source.mp4';focus_replay.encode(rgb,source,24)
        state=clip_focus.initialize(source,50)
        d=np.full((9,64,96),.82,np.float32);d[:,:,:30]=.99
        path=Path(state['dir'])/'source_depth.npz';np.savez_compressed(path,depth=d)
        state.update(points=[[15,30],[60,30]],depth={'format':'relative_npz','raw':str(path),'clip_id':state['id'],'z_near':1.})
        clip_focus.save(state);self.state=state;self.depth=d

    def tearDown(self):
        self.temp_patch.stop();self.temp.cleanup()

    def test_recipe_nudge_replays_exactly_with_original_depth(self):
        self.state['settings']['depth_nudge']=dict(x=.7,y=.3,amount=-.2,radius=.4,depth_band=.035)
        clip_focus.save(self.state)
        with patch.object(clip_focus,'run_job',side_effect=AssertionError('No paid or local backend job')):
            result=clip_focus.render_focus(SimpleNamespace(hfr_backend=backend.CLOUD),self.state,2.8,0,clip_focus.HOLD,70)
        archive=clip_focus.export_recipe(result);restored=clip_focus.import_recipe(archive)
        self.assertEqual(restored['fps'],24)
        self.assertEqual(restored['settings']['depth_nudge'],result['settings']['depth_nudge'])
        d,_=clip_focus.cached_disparity(restored)
        np.testing.assert_array_equal(d,self.depth)
        replay=self.root/'replay.mp4'
        focus_replay.render_python(restored['sharp'],d,restored['plan'],replay)
        np.testing.assert_array_equal(focus_replay.read_video(result['focused'],96,64),focus_replay.read_video(replay,96,64))

    def test_preview_frame_change_keeps_the_render_and_original_anchor(self):
        self.state['settings']['depth_nudge']=dict(x=.7,y=.3,anchor_frame=0,amount=-.2,radius=.4,depth_band=.035)
        self.state['focused']='saved-render.mp4';clip_focus.save(self.state)
        changed=clip_lens.controls(self.state,'End',-20,40,3.5)
        self.assertEqual(changed['focused'],'saved-render.mp4')
        self.assertEqual(changed['settings']['depth_nudge']['anchor_frame'],0)
        panel=clip_lens.panel(changed)
        self.assertEqual(panel[4],'End');self.assertEqual(panel[3].size,(96,64))

    def test_preset_changes_both_controls_without_destroying_depth(self):
        changed=clip_lens.choose_preset(self.state,'Subtle')
        self.assertEqual(changed['settings']['f_stop'],5.6)
        self.assertEqual(changed['settings']['landing'],85)
        self.assertEqual(changed['depth'],self.state['depth'])
        self.assertIsNone(changed['focused'])
        changed=clip_lens.choose_mode(changed,'legacy')
        self.assertEqual(changed['settings']['lens_mode'],'legacy')
        self.assertIsNone(changed['settings']['depth_nudge'])

    def test_preset_apertures_survive_ui_slider_precision(self):
        for aperture in (1.4,2.8,5.6,2.345678):
            slider=round(clip_flow.blur(aperture),1)
            self.assertEqual(clip_flow.aperture(slider,aperture),aperture)
        self.assertNotEqual(clip_flow.aperture(66.8,2.8),2.8)
        self.assertNotEqual(clip_flow.aperture(66.6,2.8),2.8)

    def test_old_recipe_settings_keep_legacy_mode(self):
        self.state['settings']={'middle':clip_focus.HOLD,'landing':85,'f_stop':1.4,'guide':0,'edge_cleanup':0}
        clip_focus.save(self.state)
        restored=clip_focus.import_recipe(clip_focus.export_recipe(self.state))
        self.assertEqual(clip_lens.panel(restored)[0],'legacy')
        self.assertEqual(restored['settings'],self.state['settings'])

if __name__=='__main__':unittest.main()
