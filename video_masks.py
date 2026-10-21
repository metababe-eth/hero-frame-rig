"""Lossless masks for the round still kernel applied across video frames."""
from pathlib import Path
import subprocess
import numpy as np
import optics,round_lens

def encode_masks(frames,path,transform,fps=25):
    """Encode RGB without chroma subsampling; verify every decoded mask byte."""
    n,h,w=frames.shape
    command=['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24',
             '-video_size',f'{w}x{h}','-framerate',str(fps),'-i','-',
             '-an','-c:v','libx264rgb','-crf','0','-preset','fast','-threads','2',str(path)]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for frame in frames:
            mask=np.round(np.clip(transform(frame),0,1)*255).astype(np.uint8)
            proc.stdin.write(np.repeat(mask[...,None],3,axis=2).tobytes())
        proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace')[-1000:])
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
    decoder=subprocess.Popen(['ffmpeg','-v','error','-i',str(path),'-map','0:v:0','-vsync','0',
                              '-f','rawvideo','-pix_fmt','rgb24','-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for frame in frames:
            expected=np.round(np.clip(transform(frame),0,1)*255).astype(np.uint8)
            raw=decoder.stdout.read(w*h*3)
            actual=np.frombuffer(raw,np.uint8).reshape(h,w,3)
            if not np.array_equal(actual,np.repeat(expected[...,None],3,axis=2)):
                raise ValueError('Mask video changed values during encoding.')
        if decoder.stdout.read(1):raise ValueError('Mask video contains extra frames.')
        error=decoder.stderr.read();code=decoder.wait()
        if code:raise RuntimeError(error.decode(errors='replace')[-1000:])
    finally:
        if decoder.poll() is None:decoder.kill();decoder.wait()

def prepare(disparities,focus_values,blur,directory,fps=25):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    n,h,w=disparities.shape
    if len(focus_values)!=n:raise ValueError('Focus must cover every video frame.')
    fractions=np.lib.format.open_memmap(directory/'blur_fractions.npy',mode='w+',dtype='float32',shape=(n,h,w))
    for i,(d,focus) in enumerate(zip(disparities,focus_values)):
        # Same saved uint8 disparity boundary and focus rounding as still takes.
        spread=optics.spread_foreground(d,focus,blur)
        spread=np.round(np.clip(spread,0,1)*255)/255
        params=optics.dof_inputs(blur,focus)
        fractions[i]=np.clip((np.abs(spread-params['focus_distance'])-params['focus_range'])/(1-params['focus_range']+1e-6),0,1) if params['blur_amount']>=.1 else 0
    fractions.flush()
    paths=[]
    for level in range(1,6):
        path=directory/f'weight_{level}.mp4'
        encode_masks(fractions,path,lambda f,level=level:np.clip((f-(level-1)/5)*5,0,1),fps)
        paths.append(str(path))
    spec=round_lens.describe(blur,w,h);active=[v['level'] for v in spec['levels'] if not v['identity']]
    offset=(min(active)-1)/5 if active else 1
    maximum=max(0,float(fractions.max())-offset)
    def guide(f):
        t=np.maximum(f-offset,0)/maximum if maximum else np.zeros_like(f)
        return t*t*(3-2*t)
    gp=directory/'guide.mp4';encode_masks(fractions,gp,guide,fps)
    del fractions
    (directory/'blur_fractions.npy').unlink()
    return paths,str(gp),{'kernel':spec,'guide_normalization':'one shared maximum across all frames; no per-frame gain pumping',
                         'maximum_blur_fraction':maximum,'encoding':'lossless libx264rgb crf0; every decoded RGB byte checked',
                         'frames':n,'width':w,'height':h}
