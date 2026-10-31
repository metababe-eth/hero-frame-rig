"""Opt-in CPU optical approximation, independent MIT implementation.

Relative bright-near depth is interpreted as affine inverse distance over a
declared virtual range. Thin-lens CoC drives normalized antialiased discs;
premultiplied depth layers composite far to near. No radiance is invented, no
display grade/noise is added, and the legacy renderer is not changed.
"""
from functools import lru_cache
import math
import numpy as np
import torch
import torch.nn.functional as F
from . import bounded, depth_array, sample, _fft_size

VERSION = 'hfr-optical-v1'
DEFAULTS = dict(lens_mode='optical', aperture=2.8, focal_length_mm=50.,
                sensor_width_mm=36., near_m=.5, far_m=10., depth_smoothing=.5,
                breathing=0.)


def settings(**values):
    """Complete, JSON-safe replay settings; do not drop defaults from recipes."""
    result = {**DEFAULTS, **values}
    if result['lens_mode'] != 'optical': raise ValueError('Expected optical mode.')
    for key, low, high in [('aperture', .7, 32), ('focal_length_mm', 8, 200),
                           ('sensor_width_mm', 8, 100), ('near_m', .25, 100),
                           ('far_m', .5, 10000), ('depth_smoothing', 0, 1),
                           ('breathing', 0, .02)]:
        result[key] = bounded(result[key], low, high, key)
    if result['far_m'] <= result['near_m']: raise ValueError('far_m must be greater than near_m.')
    if result['focal_length_mm']/1000 >= result['near_m']:
        raise ValueError('Virtual nearest distance must exceed focal length.')
    return result


def coc_radius(depth, focus, long_edge, aperture=2.8, focal_length_mm=50.,
               sensor_width_mm=36., near_m=.5, far_m=10., strength=50.):
    """Thin-lens radius in pixels, with a continuous near-focus tolerance.

    c/2 = f^2 * |1/z - 1/z_focus| / (2*N*(1 - f/z_focus)).
    Sensor width is the long edge, including portrait delivery. The 0.55px
    tolerance and soft knee suppress subpixel defocus, not depth itself.
    """
    f = focal_length_mm/1000
    inverse_range = 1/near_m - 1/far_m
    focus_inverse = 1/far_m + float(focus)*inverse_range
    radius = abs(depth-float(focus))*inverse_range*f*f/(2*aperture*(1-f*focus_inverse))
    radius = radius * long_edge/(sensor_width_mm/1000) * (strength/50.)
    radius = (radius-.55).clip(0) if isinstance(radius,np.ndarray) else (radius-.55).clamp_min(0)
    return (radius*radius/(radius+.8)).clip(0, long_edge*.045)


def smooth_depth(depth, rgb, amount=.5):
    """Centered 3-frame filter gated by both depth and RGB agreement.

    Moving silhouettes / cuts reset locally. No optical flow, propagated matte,
    previous-render state, or RGB-frame blending. Source depth is never changed.
    """
    d = np.asarray(depth,dtype=np.float32)
    if not amount or len(d)<2: return d.copy()
    rgb = rgb.detach().cpu().numpy() if isinstance(rgb,torch.Tensor) else np.asarray(rgb)
    scale = 255. if rgb.dtype == np.uint8 else 1.
    out = np.empty_like(d)
    for i in range(len(d)):
        residual = np.zeros_like(d[i]); weight = np.ones_like(d[i])
        current = rgb[i].astype(np.float32)/scale
        for j in (i-1,i+1):
            if not 0<=j<len(d): continue
            colour_delta = np.mean(np.abs(current-rgb[j].astype(np.float32)/scale),axis=-1)
            depth_delta = np.abs(d[i]-d[j])
            gate = np.clip(1-colour_delta/.06,0,1)*np.clip(1-depth_delta/.04,0,1)*amount
            residual += gate*(d[j]-d[i]); weight += gate
        out[i] = d[i]+residual/weight
    return out


def _edge_depth(depth, image, strength):
    """Small RGB-guided depth repair; never dilate blur onto a sharp subject.

    Same-colour neighbours support a depth boundary without pulling depth
    across a strong colour edge. This cannot identify transparent materials.
    """
    if not strength: return depth
    h,w=depth.shape[-2:]
    dp=F.pad(depth,(2,2,2,2),mode='replicate')
    ip=F.pad(image,(2,2,2,2),mode='replicate')
    total=depth.clone(); weight=torch.ones_like(depth)
    for dx,dy in ((-1,0),(1,0),(0,-1),(0,1),(-2,0),(2,0),(0,-2),(0,2)):
        neighbour=ip[:,:,2+dy:2+dy+h,2+dx:2+dx+w]
        delta=(image-neighbour).abs().mean(1,keepdim=True)
        support=torch.exp(-delta/.035)*(.6 if abs(dx)+abs(dy)==2 else 1.)
        total+=support*dp[:,:,2+dy:2+dy+h,2+dx:2+dx+w]; weight+=support
    return depth+(total/weight-depth)*strength


def prepare_depth(depth, images, smoothing=.5, cleanup=.5):
    """Exact preprocessing shared by rendering and recipe/study planning."""
    d=smooth_depth(depth_array(depth,tuple(images.shape[:3])),images,smoothing)
    if not cleanup: return d
    for i in range(len(d)):
        image=torch.as_tensor(images[i]).detach().to('cpu',torch.float32).permute(2,0,1)[None]
        if images.dtype in (torch.uint8,np.dtype('uint8')): image=image/255
        d[i]=_edge_depth(torch.from_numpy(d[i])[None,None],image,cleanup)[0,0].numpy()
    return d


def make_plan(depth, start_x, start_y, end_x, end_y, blur_strength=50,
              arrival=7/9, in_between='stay', head_targets=None):
    d=depth_array(depth); n,h,w=d.shape
    if n<9: raise ValueError('Scheduled focus needs at least nine frames.')
    if in_between not in ('stay','closest','face'): raise ValueError('Unknown in-between mode.')
    bounded(blur_strength,0,50,'blur strength')
    landing=50+45*bounded(arrival,0,1,'arrival')
    first=sample(d,start_x,start_y,0); last=sample(d,end_x,end_y,n-1)
    start=max(1,round((n-1)*.18)); end=min(n-2,max(start+2,round((n-1)*landing/100)))
    settle=max(1,min(3,round(n*.05))); core_end=max(start+1,end-settle)
    values=np.full(n,first,dtype=np.float64)
    if in_between != 'stay':
        if in_between=='face':
            if head_targets is None: raise ValueError('Face mode needs head_targets_json.')
            targets=np.asarray(head_targets,dtype=np.float64)
            if targets.shape!=(n,) or not np.isfinite(targets).all() or targets.min()<0 or targets.max()>1:
                raise ValueError('Head targets must cover every frame with normalized values.')
        else: targets=np.percentile(d[:,round(h*.2):round(h*.8),round(w*.3):round(w*.7)],90,axis=(1,2))
        for i in range(1,start+1): values[i]=values[i-1]+np.clip(targets[i]-values[i-1],-.02,.02)
    initial=values[start]
    # Quintic ease: zero velocity and acceleration at both ends, no overshoot.
    t=np.linspace(0,1,core_end-start+1)
    ease=t*t*t*(10+t*(-15+6*t))
    values[start:core_end+1]=initial+(last-initial)*ease
    values[core_end:]=last
    phases=[dict(start=0,end=start,mode='hold' if in_between=='stay' else in_between),
            dict(start=start,end=core_end,mode='pull'),dict(start=core_end,end=end,mode='settle'),
            dict(start=end,end=n,mode='hold')]
    return dict(version=VERSION,phase_boundary_convention='start inclusive, end exclusive',
                phases=phases,raw_curve=values.tolist(),render_curve=values.tolist(),
                focus_curve=values.tolist(),pull_lands_at_percent=landing,
                timing='quintic pull; zero-velocity settle; final hold',lens_mode='optical')


def _to_linear(image):
    return torch.where(image<=.04045,image/12.92,((image+.055)/1.055).pow(2.4))


def _to_srgb(image):
    return torch.where(image<=.0031308,image*12.92,1.055*image.clamp_min(0).pow(1/2.4)-.055).clamp(0,1)


@lru_cache(maxsize=96)
def _disc_spectrum(h,w,radius_eighths):
    radius=radius_eighths/8.; pad=math.ceil(radius+1)
    shape=(_fft_size(h+2*pad),_fft_size(w+2*pad))
    v=torch.arange(-pad,pad+1,dtype=torch.float32)
    yy,xx=torch.meshgrid(v,v,indexing='ij')
    disc=(radius+.5-torch.sqrt(xx*xx+yy*yy)).clamp(0,1)
    disc/=disc.sum()
    kernel=torch.zeros(shape,dtype=torch.float32)
    kernel[:len(v),:len(v)]=disc
    spectrum=torch.fft.rfft2(torch.roll(kernel,(-pad,-pad),(0,1)))
    return spectrum,shape,pad


def disc_blur(image,radius):
    """Unit-energy circular PSF, antialiased rim, bounded reflect padding."""
    if radius<.5: return image
    h,w=image.shape[-2:]
    radius=min(float(radius),min(h,w)/3)
    spectrum,shape,pad=_disc_spectrum(h,w,max(4,round(radius*8)))
    padded=F.pad(image,(pad,pad,pad,pad),mode='reflect')
    out=torch.fft.irfft2(torch.fft.rfft2(padded,s=shape)*spectrum,s=shape)
    return out[...,pad:pad+h,pad:pad+w].clamp_min(0)


def _frame(colour,depth,focus,params,strength,cleanup):
    h,w=depth.shape[-2:]; long_edge=max(h,w)
    radius=coc_radius(depth,focus,long_edge,**{k:params[k] for k in
        ('aperture','focal_length_mm','sensor_width_mm','near_m','far_m')},strength=strength)
    maximum=float(radius.max())
    if maximum<.5: return colour
    # Far colour under a blurred foreground is absent in the input. A normalized
    # gather from existing farther pixels is a bounded approximation, not a
    # reconstruction. It is only visible where an occluding layer becomes soft.
    far=((focus-.015-depth)/.06).clamp(0,1)
    packed=disc_blur(torch.cat((colour*far,far),1),maximum+2)
    behind=packed[:,:3]/packed[:,3:4].clamp_min(1e-6)
    # Never leak the original sharp foreground through a partially covered
    # blurred layer where no farther sample exists within the gather radius.
    fallback=disc_blur(colour,maximum)
    # A tiny distant highlight is not reliable support for a whole hidden
    # surface. Dividing by its small coverage alone inflates it into a bright
    # patch. Ease toward the gathered colour only with substantial coverage.
    support=(packed[:,3:4]/.15).clamp(0,1)
    support=support*support*(3-2*support)
    base=fallback+(behind-fallback)*support
    # Fixed inverse-depth bins avoid per-frame renormalization and pump. CoC is
    # sampled at each occupied layer's actual mean depth, not an invented plane.
    bins=24
    index=(depth*(bins-1)).round().long()
    for level in range(bins):
        mask=(index==level).to(torch.float32)
        if not bool(mask.any()): continue
        r=float((radius*mask).sum()/mask.sum())
        if r<.5:
            base=base*(1-mask)+colour*mask
            continue
        layer=disc_blur(torch.cat((colour*mask,mask),1),r)
        alpha=layer[:,3:4].clamp(0,1)
        base=base*(1-alpha)+layer[:,:3]
    return base


@torch.inference_mode()
def render(images,depth,start_x=.25,start_y=.45,end_x=.75,end_y=.55,
           blur_strength=50,arrival=7/9,in_between='stay',spotlight=0,seed=7,
           focus_curve=None,head_targets=None,progress=None,edge_cleanup=.5,**optics):
    params=settings(**optics)
    if not isinstance(images,torch.Tensor): images=torch.as_tensor(images)
    if images.ndim!=4 or images.shape[-1]!=3: raise ValueError('Images must be NHWC RGB.')
    n,h,w,_=images.shape
    if n<1 or min(h,w)<8 or n*h*w>200_000_000: raise ValueError('Invalid or oversized image batch.')
    strength=bounded(blur_strength,0,50,'blur strength'); guide=bounded(spotlight,0,100,'spotlight')
    cleanup=bounded(edge_cleanup,0,1,'edge cleanup')
    for name,value in [('start_x',start_x),('start_y',start_y),('end_x',end_x),('end_y',end_y),('arrival',arrival)]:
        bounded(value,0,1,name)
    # Validate before smoothing, including floating-point image range.
    if not torch.isfinite(images).all() or images.min()<0 or images.max()>(255 if images.dtype==torch.uint8 else 1):
        raise ValueError('Images must be finite RGB values in 0–1, or uint8.')
    d=prepare_depth(depth,images,params['depth_smoothing'],cleanup)
    if focus_curve is None:
        focus_curve=make_plan(d,start_x,start_y,end_x,end_y,strength,arrival,in_between,head_targets)['render_curve']
    curve=np.asarray(focus_curve,dtype=np.float64)
    if curve.shape!=(n,) or not np.isfinite(curve).all() or curve.min()<0 or curve.max()>1:
        raise ValueError('Focus curve must contain one normalized value per frame.')
    out=torch.empty((n,h,w,3),dtype=torch.float32)
    for i in range(n):
        image=images[i].detach().to('cpu',torch.float32).permute(2,0,1)[None]
        if images.dtype==torch.uint8: image=image/255
        disparity=torch.from_numpy(d[i])[None,None]
        result=_frame(_to_linear(image),disparity,float(curve[i]),params,strength,cleanup)
        if guide:
            amount=((disparity-float(curve[i])).abs()/.5).clamp(0,1)*(guide/100)
            result=result*(1-amount*(1-2**(-.7)))
        result=_to_srgb(result)
        if params['breathing'] and curve[i] != curve[0]:
            scale=1+params['breathing']*abs(float(curve[i]-curve[0]))
            transform=torch.tensor([[[1/scale,0,0],[0,1/scale,0]]],dtype=torch.float32)
            grid=F.affine_grid(transform,result.shape,align_corners=False)
            result=F.grid_sample(result,grid,mode='bilinear',padding_mode='border',align_corners=False)
        out[i]=result[0].permute(1,2,0)
        if progress: progress(i+1,n)
    return out
