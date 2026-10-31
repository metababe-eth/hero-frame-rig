"""Small, explicit focus controls shared by app, recipes and nodes (MIT)."""
import numpy as np
from . import bounded, depth_array, sample

PRESETS = {
    'Subtle': {'aperture': 5.6, 'landing': 85.},
    'Cinematic': {'aperture': 2.8, 'landing': 70.},
    'Dramatic': {'aperture': 1.4, 'landing': 60.},
}


def preset(name):
    if name not in PRESETS: raise ValueError('Choose Subtle, Cinematic or Dramatic.')
    p=PRESETS[name]
    return dict(lens_mode='optical', aperture=p['aperture'], landing=p['landing'],
                arrival=(p['landing']-50)/45, focus_preset=name)


def preset_name(aperture, landing):
    return next((k for k,p in PRESETS.items() if abs(float(aperture)-p['aperture'])<1e-6
                 and abs(float(landing)-p['landing'])<1e-6), None)


def nudge_settings(value):
    if value is None: return None
    if not isinstance(value,dict): raise ValueError('Depth nudge must be an object.')
    result={k:bounded(value.get(k,default),low,high,k) for k,default,low,high in (
        ('x',.5,0,1),('y',.5,0,1),('amount',0,-.3,.3),
        ('radius',.35,.05,.75),('depth_band',.035,.01,.12))}
    frame=value.get('anchor_frame',0)
    if isinstance(frame,bool) or not isinstance(frame,(int,np.integer)) or frame<0:
        raise ValueError('Depth nudge anchor must be a nonnegative frame index.')
    result['anchor_frame']=int(frame)
    if value.get('anchor_depth') is not None:
        result['anchor_depth']=bounded(value['anchor_depth'],0,1,'anchor depth')
    return result


def resolve_nudge(depth, value):
    cfg=nudge_settings(value)
    if cfg is None: return None
    d=depth_array(depth)
    if cfg['anchor_frame']>=len(d): raise ValueError('Depth nudge anchor is outside the clip.')
    if 'anchor_depth' not in cfg:
        cfg['anchor_depth']=sample(d,cfg['x'],cfg['y'],cfg['anchor_frame'])
    return cfg


def nudge_mask(depth, cfg):
    """Fixed screen region, softly gated by similar depth; not object tracking."""
    d=np.asarray(depth,dtype=np.float32);h,w=d.shape[-2:]
    y,x=np.ogrid[0:h,0:w]
    distance=np.sqrt(((x/max(1,w-1)-cfg['x'])/cfg['radius'])**2+
                     ((y/max(1,h-1)-cfg['y'])/cfg['radius'])**2)
    spatial=np.clip((1-distance)/.3,0,1).astype(np.float32)
    spatial=spatial*spatial*(3-2*spatial)
    band=np.clip((cfg['depth_band']-np.abs(d-cfg['anchor_depth']))/(cfg['depth_band']*.4),0,1)
    return spatial*(band*band*(3-2*band))


def nudge_depth(depth,value):
    """Return adjusted depth. The input cache and RGB are never changed."""
    d=depth_array(depth);cfg=resolve_nudge(d,value)
    if cfg is None or cfg['amount']==0: return d.copy()
    return np.clip(d+cfg['amount']*nudge_mask(d,cfg),0,1).astype(np.float32)


def depth_warning(depth,start_x,start_y,end_x,end_y):
    d=depth_array(depth)
    first=sample(d,start_x,start_y,0);last=sample(d,end_x,end_y,len(d)-1)
    gap=abs(last-first)
    return dict(start=first,end=last,gap=gap,close=gap<.12,
                message=('These focus points have similar estimated depth. The pull may be subtle. '
                         'Choose different points or adjust a depth region.' if gap<.12 else ''))
