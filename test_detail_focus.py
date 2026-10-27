import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import detail_focus as area


class AreaMathTests(unittest.TestCase):
    def test_exact_sharp_core_and_resolve(self):
        frame=np.random.default_rng(7).integers(0,256,(72,128,3),dtype=np.uint8)
        points=[[.2,.5],[.8,.5],[.5,.25]]
        s=copy.deepcopy(area.DEFAULTS)
        masks=area.masks(128,72,points,s['radii'])
        out=area.process_frame(frame,masks,0,192,s)
        core=masks[0]==1
        self.assertGreater(int(core.sum()),100)
        np.testing.assert_array_equal(out[core],frame[core])
        self.assertGreater(np.abs(out.astype(int)-frame).sum(),100)
        np.testing.assert_array_equal(area.process_frame(frame,masks,191,192,s),frame)
        s['blur']=0
        np.testing.assert_array_equal(area.process_frame(frame,masks,71,192,s),frame)

    def test_schedule_has_real_three_holds(self):
        s=copy.deepcopy(area.DEFAULTS)
        np.testing.assert_allclose(area.weights(0,201,s),[1,0,0])
        np.testing.assert_allclose(area.weights(80,201,s),[0,1,0])
        np.testing.assert_allclose(area.weights(150,201,s),[0,0,1])
        for i in range(201):
            self.assertAlmostEqual(sum(area.weights(i,201,s)),1)

    def test_settings_and_point_validation(self):
        for points in [[[0,0],[1,1]],[[0,0],[1,1],[float('nan'),.2]],[[0,0],[1,1],[1.2,.2]]]:
            with self.assertRaises(ValueError):area.validated_points(points)
        with self.assertRaises(ValueError):area.settings(arrival2=15,pull=20)
        with self.assertRaises(ValueError):area.settings(r1=.001)
        with self.assertRaises(ValueError):area.settings(blur=float('inf'))


class AreaRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work=Path(tempfile.mkdtemp(prefix='hfr_area_tests_'))
        clip=cls.work/'fixture.mp4'
        subprocess.run(['ffmpeg','-v','error','-n','-f','lavfi','-i','testsrc2=size=128x72:rate=12:duration=1',
                        '-an','-c:v','libx264','-pix_fmt','yuv420p',str(clip)],check=True)
        state=area.open_clip(clip)
        for i,xy in enumerate([(25,36),(100,36),(64,20)]):state=area.selected(state,i,xy)
        cls.state=area.render(state)
        cls.recipe=Path(area.export_recipe(cls.state))

    def test_real_render_and_roundtrip(self):
        old=self.state;new=area.import_recipe(self.recipe)
        self.assertEqual(old['frames'],12)
        self.assertEqual(old['fps'],12)
        self.assertEqual(new['source_sha256'],old['source_sha256'])
        self.assertEqual(new['result_sha256'],old['result_sha256'])
        self.assertEqual(new['original_sha256'],old['original_sha256'])
        self.assertEqual(new['settings'],old['settings'])
        self.assertEqual(new['points'],old['points'])
        self.assertEqual(new['render_record']['cloud_jobs'],0)
        self.assertFalse(new['render_record']['tracking'])
        second=area.render(new)
        self.assertEqual(second['result_sha256'],old['result_sha256'])

    def test_reject_path_members_without_writing(self):
        bad=self.work/'traversal.zip'
        with zipfile.ZipFile(self.recipe) as source,zipfile.ZipFile(bad,'w') as dest:
            for name in source.namelist():dest.writestr('../outside' if name=='result.mp4' else name,source.read(name))
        with self.assertRaises(ValueError):area.import_recipe(bad)
        self.assertFalse((self.work.parent/'outside').exists())

    def test_reject_changed_media(self):
        bad=self.work/'checksum.zip'
        with zipfile.ZipFile(self.recipe) as source,zipfile.ZipFile(bad,'w') as dest:
            for name in source.namelist():dest.writestr(name,source.read(name)+(b'changed' if name=='result.mp4' else b''))
        with self.assertRaises(ValueError):area.import_recipe(bad)

    def test_reject_missing_result_export(self):
        state=copy.deepcopy(self.state);state['result']=None
        with self.assertRaises(ValueError):area.export_recipe(state)

    def test_failed_render_reaps_encoder_and_preserves_prior_result(self):
        source=area.sha(self.state['source']); original=area.sha(self.state['original'])
        result=area.sha(self.state['result']); before=set(Path(self.state['dir']).glob('area_result_*.mp4'))
        real_popen=subprocess.Popen; processes=[]
        def record_popen(*args,**kwargs):
            process=real_popen(*args,**kwargs);processes.append(process);return process
        with patch.object(area.subprocess,'Popen',side_effect=record_popen), patch.object(area,'process_frame',side_effect=RuntimeError('interrupted frame')):
            with self.assertRaisesRegex(RuntimeError,'interrupted frame'):area.render(self.state)
        self.assertEqual(len(processes),1)
        self.assertIsNotNone(processes[0].returncode)
        self.assertTrue(processes[0].stdin.closed)
        self.assertTrue(processes[0].stderr.closed)
        self.assertEqual(set(Path(self.state['dir']).glob('area_result_*.mp4')),before)
        self.assertEqual(area.sha(self.state['source']),source)
        self.assertEqual(area.sha(self.state['original']),original)
        self.assertEqual(area.sha(self.state['result']),result)

    def test_reject_renamed_playlist_before_following_resources(self):
        playlist=self.work/'playlist.mp4'
        playlist.write_text('#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nfile:///not-an-area-clip.ts\n#EXT-X-ENDLIST\n')
        with self.assertRaises(ValueError) as caught:area.probe(playlist)
        self.assertNotIn('not-an-area-clip.ts',str(caught.exception))

    def test_odd_source_dimensions_do_not_upscale(self):
        clip=self.work/'odd.mp4'
        subprocess.run(['ffmpeg','-v','error','-n','-f','lavfi','-i','testsrc=size=127x73:rate=12:duration=1',
                        '-an','-c:v','libx264','-pix_fmt','yuv444p',str(clip)],check=True)
        prepared=area.open_clip(clip)
        self.assertEqual((prepared['width'],prepared['height']),(126,72))
        self.assertEqual((prepared['frames'],prepared['rate']),(12,'12'))


class AreaUIStateTests(unittest.TestCase):
    def test_private_serialized_events_and_complete_clear(self):
        import asyncio
        loop=asyncio.new_event_loop();asyncio.set_event_loop(loop)
        self.addCleanup(loop.close)
        import gradio as gr
        with gr.Blocks() as demo:area.build_ui()
        functions=list(demo.fns.values())
        self.assertGreater(len(functions),10)
        for fn in functions:
            self.assertEqual(fn.api_visibility,'private')
            self.assertEqual(fn.concurrency_id,'area-focus-ui')
            self.assertEqual(fn.concurrency_limit,1)
        clear=next(fn for fn in functions if fn.fn.__name__=='clear_clip')
        values=clear.fn()
        self.assertEqual(len(values),len(clear.outputs))
        self.assertIsNone(values[0])
        self.assertEqual(values[4]['visible'],False)
        self.assertEqual(values[5]['interactive'],False)
        self.assertIsNone(values[7])
        self.assertIsNone(values[8])
        self.assertEqual(values[9]['visible'],False)
        self.assertIsNone(values[10])
        self.assertEqual(values[11:],[44.,.14,.14,.24,32.,63.,12.,True])


if __name__=='__main__':unittest.main()
