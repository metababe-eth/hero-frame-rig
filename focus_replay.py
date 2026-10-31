"""Portable exact-depth workflow and media adapter for the shared node core."""
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
import focus_core


def workflow(plan, sharp='sharp.mp4', depth='hfr_depth.npz'):
    w,h=plan['width'],plan['height']; a,b=plan['points']
    mode={'Hold the start':'stay','Follow what is nearest':'closest',"Follow the subject's head":'face'}.get(plan['middle'],'stay')
    params=dict(start_x=a[0]/(w-1),start_y=a[1]/(h-1),end_x=b[0]/(w-1),end_y=b[1]/(h-1),
        blur_strength=plan['blur'], arrival=(plan['pull_lands_at_percent']-50)/45,
        in_between=mode,spotlight=plan['guide'],seed=7,edge_cleanup=plan.get('edge_cleanup',0),focus_curve_json=json.dumps(plan['render_curve']))
    lens_mode=plan.get('lens_mode','legacy')
    if lens_mode not in ('legacy','optical'): raise ValueError('Unknown recipe lens mode.')
    if lens_mode=='optical':
        from hfr_core.optical import DEFAULTS, settings
        # Complete numeric settings are written into the graph for portable
        # replay. Missing mode on every older recipe still follows Legacy.
        params.update(settings(**{key:plan[key] for key in DEFAULTS if key in plan}))
    return {
        '1': {'class_type':'LoadVideo','inputs':{'file':sharp}},
        '2': {'class_type':'GetVideoComponents','inputs':{'video':['1',0]}},
        '3': {'class_type':'HFRLoadDepthBatch','inputs':{'file':depth}},
        '4': {'class_type':'HFRFocusPull','inputs':{'images':['2',0],'depth':['3',0],**params}},
        '5': {'class_type':'CreateVideo','inputs':{'images':['4',0],'fps':plan['fps']}},
        '6': {'class_type':'SaveVideo','inputs':{'video':['5',0],'filename_prefix':'hero_frame_rig/directed','format':'mp4',
            'format.codec':'h264','format.codec.encoding':'re-encode','format.codec.encoding.crf':18}},
    }


def read_video(path,width,height):
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map','0:v:0','-vsync','0','-f','rawvideo','-pix_fmt','rgb24','-'])
    return np.frombuffer(raw,np.uint8).reshape(-1,height,width,3)


def encode(images,path,fps=25):
    n,h,w,c=images.shape
    p=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}',
        '-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','18','-preset','fast','-pix_fmt','yuv420p','-threads','2',str(path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for im in images:
            if isinstance(im,torch.Tensor): im=im.numpy()
            p.stdin.write(np.round(np.clip(im,0,1)*255).astype(np.uint8).tobytes())
        p.stdin.close();error=p.stderr.read();code=p.wait()
        if code: raise RuntimeError(error.decode(errors='replace')[-1000:])
    finally:
        if p.poll() is None:p.kill();p.wait()


def render_python(sharp,depth,plan,path,progress=None):
    frames=read_video(sharp,plan['width'],plan['height'])
    params=workflow(plan)['4']['inputs'].copy()
    for key in ['images','depth','focus_curve_json']: params.pop(key)
    # UInt8 source avoids retaining an extra full float copy of the clip.
    frames=torch.from_numpy(frames.copy())
    result=focus_core.render(frames,depth,focus_curve=plan['render_curve'],progress=progress,**params)
    encode(result,path,plan['fps'])
    return result
