"""HFR shared CPU lens. Only numpy and torch; no models, network or host imports.

The owned HFR schedule/round-kernel calibration is reused here. Linear-light
processing is independently implemented; this pack contains no Radiance code.
"""
import json
import math
from functools import lru_cache
import numpy as np
import torch
import torch.nn.functional as F
from . import schedule, edges
from ._support import describe, patch_level

VERSION = 'hfr-shared-round-v2-edges'


def bounded(value, low, high, name):
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be finite and between {low} and {high}.')
    return value


def depth_array(depth, shape=None):
    a = depth.detach().cpu().numpy() if isinstance(depth, torch.Tensor) else np.asarray(depth)
    if a.ndim == 4: a = a[..., :3].mean(-1)
    if a.ndim == 2: a = a[None]
    if a.ndim != 3 or min(a.shape) < 1 or not np.isfinite(a).all():
        raise ValueError('Depth must be a finite IMAGE or MASK batch, bright = near.')
    if a.min() < 0 or a.max() > 1: raise ValueError('Depth must be normalized to 0–1, bright = near.')
    if shape:
        n, h, w = shape
        if len(a) not in (1, n): raise ValueError('Depth must have one frame or exactly match the video length.')
        if a.shape[1:] != (h, w):
            a = F.interpolate(torch.from_numpy(np.array(a, dtype=np.float32))[:, None], (h, w), mode='bilinear', align_corners=False)[:, 0].numpy()
        if len(a) == 1: a = np.broadcast_to(a, (n, h, w))
    return a


def sample(depth, x, y, frame=0):
    d = depth_array(depth)
    x = round(bounded(x, 0, 1, 'tap x') * (d.shape[2] - 1))
    y = round(bounded(y, 0, 1, 'tap y') * (d.shape[1] - 1))
    frame = int(bounded(frame, 0, len(d) - 1, 'frame'))
    return patch_level(d[frame], x, y)


def make_plan(depth, start_x, start_y, end_x, end_y, blur_strength=50, arrival=7/9, in_between='stay', head_targets=None):
    d = depth_array(depth); n, h, w = d.shape
    if n < 9: raise ValueError('Scheduled focus needs at least nine frames.')
    if in_between not in ('stay', 'closest', 'face'): raise ValueError('Unknown in-between mode.')
    taps = [bounded(x, 0, 1, 'tap') for x in (start_x, start_y, end_x, end_y)]
    points = [[round(taps[i] * (w-1)), round(taps[i+1] * (h-1))] for i in (0, 2)]
    landing = 50 + 45 * bounded(arrival, 0, 1, 'arrival')
    start = max(1, int(n*.12)); end = min(n-1, int(n*landing/100)); pull = max(start+1, end-max(1, int(n*.13)))
    middle = {'start': start, 'end': pull, 'mode': 'fixed', 'point': points[0], 'anchor_frame': 0}
    if in_between == 'closest': middle = dict(start=start, end=pull, mode='nearest', percentile=90, box=[round(w*.3), round(h*.2), round(w*.7), round(h*.8)])
    if in_between == 'face':
        if head_targets is None:
            raise ValueError('Face mode needs optional head_targets_json from subject analysis. Use closest or stay without a model.')
        middle = dict(start=start, end=pull, mode='head')
    phases = [dict(start=0, end=start, mode='fixed', point=points[0], anchor_frame=0), middle,
              dict(start=pull, end=end, mode='pull', point=points[1], anchor_frame=n-1),
              dict(start=end, end=n, mode='fixed', point=points[1], anchor_frame=n-1)]
    return schedule.curve(d, phases, max(.001, bounded(blur_strength, 0, 50, 'blur strength')), head_targets)


def _unit(x): return x.clamp(0, 1)
def _q(x): return torch.round(_unit(x)*255)/255
def _ease(x):
    x = _unit(x); return x*x*(3-2*x)


def _fft_size(value):
    while True:
        test=value
        for prime in (2,3,5):
            while test % prime == 0: test //= prime
        if test == 1: return value
        value += 1


@lru_cache(maxsize=24)
def _ellipse_spectrum(h,w,radius):
    fh,fw=_fft_size(h+2*radius),_fft_size(w+2*radius)
    kernel=torch.zeros((fh,fw),dtype=torch.float32)
    for dy in range(-radius,radius+1):
        dx=round(radius*math.sqrt(max(0,1-(dy/radius)**2)))
        kernel[dy+radius,radius-dx:radius+dx+1]=1
    return torch.fft.rfft2(torch.roll(kernel,(-radius,-radius),(0,1))), (fh,fw)


def _ellipse_dilate(mask, radius):
    """Exact binary ellipse dilation via zero-padded FFT convolution.

    Counts are integers; threshold .5 avoids FFT roundoff creating false edges.
    This is much cheaper than hundreds of full-frame row max-pools on CPU.
    """
    if radius<1:return mask
    h,w=mask.shape[-2:];kernel,size=_ellipse_spectrum(h,w,radius)
    values=torch.fft.irfft2(torch.fft.rfft2(mask,s=size)*kernel,s=size)[...,:h,:w]
    return (values>.5).float()


def _fraction(d, focus, blur):
    spread = d.clone(); near = d-focus; top = float(near.max())
    if top > 0 and blur > 0:
        for t in np.linspace(top/12, top, 12):
            radius = round(.8*blur*float(t))
            if radius >= 1:
                mask = _ellipse_dilate((near >= t).float(), radius)
                spread = torch.maximum(spread, mask*(focus+float(t)))
    return _unit(((_q(spread)-round(float(focus), 4)).abs()-.01)/(.99+1e-6)) if blur*.99 >= .1 else torch.zeros_like(d)


def _gauss(x, radius, sigma):
    coords = torch.linspace(-1, 1, 2*radius+1, dtype=x.dtype)
    kernel = torch.exp(-coords.square()/(2*sigma*sigma)); kernel /= kernel.sum()
    channels = x.shape[1]
    horizontal = kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1)
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode='reflect'), horizontal, groups=channels)
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode='reflect'), vertical, groups=channels)


def _linear(image):
    x = torch.where(image <= .04045, image/12.92, ((image+.055)/1.055).pow(2.4))
    luma = (x*torch.tensor([.2126, .7152, .0722])[None, :, None, None]).sum(1, keepdim=True)
    expanded = luma + 2800 * (luma-.95).clamp_min(0).square()
    return x * expanded/(luma+1e-8)


def _finish(x, seed, frame):
    # Two soft shoulders retain the app's established digital presentation.
    delta = (x-.85).clamp_min(0); x = torch.where(x>.85, .85+delta/(1+.6*delta), x)
    x = torch.where(x<=.0031308, 12.92*x, 1.055*x.clamp_min(0).pow(1/2.4)-.055)
    # Very fine deterministic sensor texture, independently implemented. It is
    # deliberately not a copy of Radiance's GPL film-grain implementation.
    rng = torch.Generator(device='cpu').manual_seed((int(seed)+frame) % (2**63-1))
    x = x + torch.randn(x.shape, generator=rng, dtype=x.dtype)*.00025
    delta = (x-.7).clamp_min(0); x = torch.where(x>.7, .7+delta/(1+1.2*delta), x)
    return _unit(x)


@torch.inference_mode()
def render(images, depth, start_x=.25, start_y=.45, end_x=.75, end_y=.55,
           blur_strength=50, arrival=7/9, in_between='stay', spotlight=0, seed=7,
           focus_curve=None, head_targets=None, progress=None, edge_cleanup=.5):
    """Return NHWC float IMAGE. Identical function used by app and Comfy node.

    Work is framewise on CPU to bound temporary memory. No input is changed.
    blur_strength is the HFR lens scale B (0–50), not a physical aperture.
    """
    if not isinstance(images, torch.Tensor): images = torch.as_tensor(images)
    if images.ndim != 4 or images.shape[-1] != 3: raise ValueError('Images must be NHWC RGB.')
    n,h,w,_ = images.shape
    if min(h,w)<8 or n*h*w>200_000_000: raise ValueError('Batch too large (200 million pixels maximum), or frames too small.')
    blur = bounded(blur_strength, 0, 50, 'blur strength'); guide = bounded(spotlight, 0, 100, 'spotlight')
    for name, value in [('start_x', start_x), ('start_y', start_y), ('end_x', end_x), ('end_y', end_y), ('arrival', arrival)]: bounded(value, 0, 1, name)
    cleanup = bounded(edge_cleanup, 0, 1, 'edge cleanup')
    d = depth_array(depth, (n,h,w))
    if focus_curve is None:
        focus_curve = make_plan(d, start_x, start_y, end_x, end_y, blur, arrival, in_between, head_targets)['render_curve']
    curve = np.asarray(focus_curve, dtype=np.float64)
    if curve.shape != (n,) or not np.isfinite(curve).all() or curve.min()<0 or curve.max()>1: raise ValueError('Focus curve must contain one normalized value per frame.')
    spec = describe(blur,w,h); sw,sh = spec['downsample']
    active = [v for v in spec['levels'] if not v['identity']]
    offset = (min(v['level'] for v in active)-1)/5 if active else 1
    fractions = torch.empty((n,h,w),dtype=torch.float32)
    for i in range(n):
        fractions[i] = _fraction(torch.tensor(d[i])[None,None],curve[i],blur)[0,0]
    maximum = max(0, float(fractions.max())-offset)
    out = torch.empty((n,h,w,3),dtype=torch.float32)
    for i in range(n):
        image = images[i].detach().to('cpu',torch.float32).permute(2,0,1)[None]
        if images.dtype == torch.uint8: image = image / 255
        if not torch.isfinite(image).all() or image.min()<0 or image.max()>1: raise ValueError('Images must be finite RGB values in 0–1.')
        colour = _linear(image); result = colour.clone()
        disparity = torch.tensor(d[i])[None,None]; focus=float(curve[i]); fraction=fractions[i][None,None]
        owner, edge_alpha = edges.ownership(disparity,image,cleanup) if cleanup else (disparity,None)
        owner_fraction = _unit(((owner-focus).abs()-.01)/(.99+1e-6))
        edge_result = colour.clone() if cleanup else None
        margin = max(.025,focus*.15)
        source = _ease((focus-margin-disparity)/margin)
        source = _q((-F.max_pool2d(-source,7,stride=1,padding=3)).clamp_min(1/255))
        target = _q(_ease((focus-.01-disparity)/.04))
        low = F.interpolate(torch.cat((colour,colour*source,source),1),(sh,sw),mode='area')
        for level in active:
            blurred = _gauss(low,level['radius'],level['core_sigma'])
            ordinary = F.interpolate(blurred[:,:3],(h,w),mode='bilinear',align_corners=False)
            gathered = F.interpolate(blurred[:,3:6]/blurred[:,6:7].clamp_min(1e-6),(h,w),mode='bilinear',align_corners=False)
            treatment = ordinary*(1-target)+gathered*target
            weight = _q((fraction-(level['level']-1)/5)*5)
            result = result*(1-weight)+treatment*weight
            if cleanup:
                # A thin edge must not retain an identity-image contribution
                # once its owner is defocused; that contribution is the halo.
                edge_weight = (_ease(owner_fraction*10) if level['level']==1 else
                               _q((owner_fraction-(level['level']-1)/5)*5))
                edge_result = edge_result*(1-edge_weight)+ordinary*edge_weight
        if cleanup:
            # Only a farther-depth structure with a defocused near owner changes.
            alpha = edge_alpha * _ease(owner_fraction*10)
            result = result*(1-alpha)+edge_result*alpha
        if guide and maximum:
            weight = _q(_ease((fraction-offset).clamp_min(0)/maximum))
            gain = 2**(-.7*guide/100); sat=1-.25*guide/100
            graded = result*gain; luma=(graded*torch.tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1,keepdim=True)
            graded = luma+(graded-luma)*sat
            result = result*(1-weight)+graded*weight
        out[i] = _finish(result,seed,i)[0].permute(1,2,0)
        if progress: progress(i+1,n)
    return out


def detect_beats(audio, sample_rate, fps=25):
    """Spectral-flux onset peaks, snapped to the nearest video frame.

    This is transient detection, not a promise to identify every musical beat.
    Analyze the whole supplied song before mapping the clip's time interval.
    """
    wave = audio.detach().cpu().numpy() if isinstance(audio,torch.Tensor) else np.asarray(audio)
    if wave.ndim>1: wave=wave.reshape(-1,wave.shape[-1]).mean(0)
    if not np.isfinite(wave).all() or sample_rate<=0 or fps<=0: raise ValueError('Invalid audio or rate.')
    stride=max(1,round(sample_rate/11025)); wave=wave[::stride]; rate=sample_rate/stride
    size=1024; hop=128
    if len(wave)<size or float(np.max(np.abs(wave)))<1e-8: return {'seconds':[], 'frames':[], 'method':'spectral flux', 'fps':fps}
    windows=np.lib.stride_tricks.sliding_window_view(wave,size)[::hop]
    spec=np.abs(np.fft.rfft(windows*np.hanning(size),axis=1))
    flux=np.maximum(np.diff(spec,axis=0,prepend=spec[:1]),0).sum(1)
    radius=max(1,round(.3*rate/hop)); threshold=np.array([np.median(flux[max(0,i-radius):i+radius+1]) for i in range(len(flux))])*1.4
    candidates=[i for i in range(1,len(flux)-1) if flux[i]>threshold[i]+.02*flux.max() and flux[i]>=flux[i-1] and flux[i]>flux[i+1]]
    selected=[]; gap=max(1,round(.18*rate/hop))
    for i in sorted(candidates,key=lambda j:flux[j],reverse=True):
        if all(abs(i-j)>=gap for j in selected): selected.append(i)
    seconds=[(i*hop+size/2)/rate for i in sorted(selected)]
    return {'seconds':seconds,'frames':[round(t*fps) for t in seconds], 'fps':fps, 'method':'spectral-flux transient peaks; 180 ms minimum spacing'}


def beat_curve(depth, beats, near_x, near_y, far_x, far_y, fps=25, audio_offset=0):
    d=depth_array(depth);n=len(d)
    snaps=sorted(set(round((t-audio_offset)*fps) for t in beats['seconds'] if 0<=round((t-audio_offset)*fps)<n))
    curve=[];far=False;indices=set(snaps)
    for i in range(n):
        if i in indices: far=not far
        curve.append(sample(d,far_x if far else near_x,far_y if far else near_y,i))
    return curve,snaps
