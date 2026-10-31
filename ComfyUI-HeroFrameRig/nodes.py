import json
from pathlib import Path
import numpy as np
import torch
from . import hfr_core as core


class FocusPull:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'images': ('IMAGE',), 'depth': ('*',),
            'start_x': ('FLOAT', {'default':.25,'min':0,'max':1,'step':.001}),
            'start_y': ('FLOAT', {'default':.45,'min':0,'max':1,'step':.001}),
            'end_x': ('FLOAT', {'default':.75,'min':0,'max':1,'step':.001}),
            'end_y': ('FLOAT', {'default':.55,'min':0,'max':1,'step':.001}),
            'blur_strength': ('FLOAT', {'default':50,'min':0,'max':50,'step':.1}),
            'arrival': ('FLOAT', {'default':7/9,'min':0,'max':1,'step':.01}),
            'in_between': (['stay','closest','face'],),
            'spotlight': ('FLOAT', {'default':0,'min':0,'max':100,'step':1}),
            'seed': ('INT', {'default':7,'min':0,'max':4294967295}),
        }, 'optional': {
            # Append new widgets after the old ones: saved UI workflows keep
            # positional widget values even when API recipes use named inputs.
            'edge_cleanup': ('FLOAT', {'default':.5,'min':0,'max':1,'step':.05}),
            'focus_curve_json': ('STRING', {'default':'','multiline':True}),
            'head_targets_json': ('STRING', {'default':'','multiline':True}),
            'lens_mode': (['legacy','optical'],),
            'aperture': ('FLOAT', {'default':2.8,'min':.7,'max':32,'step':.1}),
            'focal_length_mm': ('FLOAT', {'default':50.,'min':8.,'max':200.}),
            'sensor_width_mm': ('FLOAT', {'default':36.,'min':8.,'max':100.}),
            'near_m': ('FLOAT', {'default':.5,'min':.25,'max':100.}),
            'far_m': ('FLOAT', {'default':10.,'min':.5,'max':10000.}),
            'depth_smoothing': ('FLOAT', {'default':.5,'min':0.,'max':1.,'step':.05}),
            'breathing': ('FLOAT', {'default':0.,'min':0.,'max':.02,'step':.001}),
        }}
    RETURN_TYPES=('IMAGE',)
    FUNCTION='execute'
    CATEGORY='Hero Frame Rig'
    DESCRIPTION='Direct existing pixels using bright-near depth. Legacy replays old recipes; Optical uses a virtual f/2.8 lens by default. Relative depth is not measured metres. Face requires head_targets_json.'

    def execute(self, images, depth, focus_curve_json='', head_targets_json='', **settings):
        from comfy.utils import ProgressBar
        p=ProgressBar(len(images))
        out=core.render(images,depth,
            focus_curve=json.loads(focus_curve_json) if focus_curve_json else None,
            head_targets=json.loads(head_targets_json) if head_targets_json else None,
            progress=lambda i,n:p.update_absolute(i,n), **settings)
        return (out,)


class BeatFocus:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'audio': ('AUDIO',), 'images': ('IMAGE',), 'depth': ('*',),
            **{name:('FLOAT',{'default':value,'min':0,'max':1,'step':.001}) for name,value in [('near_x',.25),('near_y',.45),('far_x',.75),('far_y',.55)]},
            'fps': ('FLOAT',{'default':25,'min':1,'max':120}),
            'audio_offset': ('FLOAT',{'default':0,'min':0,'max':36000}),
            'blur_strength': ('FLOAT',{'default':50,'min':0,'max':50}),
            'spotlight': ('FLOAT',{'default':0,'min':0,'max':100}),
            'seed': ('INT',{'default':7,'min':0,'max':4294967295}),
        }, 'optional': {'edge_cleanup':('FLOAT',{'default':.5,'min':0,'max':1,'step':.05})}}
    RETURN_TYPES=('IMAGE','STRING')
    RETURN_NAMES=('images','beat_report_json')
    FUNCTION='execute'
    CATEGORY='Hero Frame Rig'

    def execute(self,audio,images,depth,near_x,near_y,far_x,far_y,fps,audio_offset,blur_strength,spotlight,seed,edge_cleanup=.5):
        depth=core.depth_array(depth,tuple(images.shape[:3]))
        beats=core.detect_beats(audio['waveform'],audio['sample_rate'],fps)
        curve,snaps=core.beat_curve(depth,beats,near_x,near_y,far_x,far_y,fps,audio_offset)
        output=core.render(images,depth,near_x,near_y,far_x,far_y,blur_strength=blur_strength,
                           spotlight=spotlight,seed=seed,focus_curve=curve,edge_cleanup=edge_cleanup)
        report={**beats,'clip_snap_frames':snaps,'audio_offset':audio_offset,'focus_curve':curve}
        return output,json.dumps(report)


class DepthSample:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'depth':('*',),'x':('FLOAT',{'default':.5,'min':0,'max':1}),
            'y':('FLOAT',{'default':.5,'min':0,'max':1}), 'frame':('INT',{'default':0,'min':0})}}
    RETURN_TYPES=('FLOAT',)
    FUNCTION='execute'
    CATEGORY='Hero Frame Rig'
    def execute(self,depth,x,y,frame=0): return (core.sample(depth,x,y,frame),)


class LoadDepthBatch:
    """Small replay utility: exact float depth, no quantization or model required."""
    @classmethod
    def INPUT_TYPES(cls): return {'required': {'file':('STRING',{'default':'hfr_depth.npz'})}}
    RETURN_TYPES=('MASK',)
    FUNCTION='execute'
    CATEGORY='Hero Frame Rig/replay'
    def execute(self,file):
        import folder_paths
        path=Path(folder_paths.get_annotated_filepath(file)).resolve()
        root=Path(folder_paths.get_input_directory()).resolve()
        if not path.is_relative_to(root) or path.suffix.lower()!='.npz': raise ValueError('Choose an NPZ depth batch inside ComfyUI/input.')
        import zipfile
        with zipfile.ZipFile(path) as z:
            if len(z.infolist())!=1 or z.infolist()[0].file_size>800_000_000: raise ValueError('Depth batch is too large or contains unexpected members.')
        with np.load(path,allow_pickle=False) as z:
            d=core.depth_array(z['depth'])
            if d.size>200_000_000: raise ValueError('Depth batch too large.')
            return (torch.from_numpy(np.array(d,dtype=np.float32)),)


NODE_CLASS_MAPPINGS={'HFRFocusPull':FocusPull,'HFRBeatFocus':BeatFocus,'HFRDepthSample':DepthSample,'HFRLoadDepthBatch':LoadDepthBatch}
NODE_DISPLAY_NAME_MAPPINGS={'HFRFocusPull':'HFR Focus Pull','HFRBeatFocus':'HFR Beat Focus','HFRDepthSample':'HFR Depth Sample','HFRLoadDepthBatch':'HFR Load Depth Batch (replay)'}
