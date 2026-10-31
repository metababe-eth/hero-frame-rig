"""Local-only clip lens controls and previews; no backend client or submission."""
import numpy as np
from PIL import Image,ImageDraw
import clip_focus
from hfr_core import directing,optical


def _save_change(state):
    # Old media files remain in the clip directory; don't label a stale result
    # with new recipe settings or offer it as the newly directed clip.
    state.update(focused=None,plan=None)
    clip_focus.save(state)
    return state


def choose_preset(state,name):
    state=clip_focus.current(state);p=directing.preset(name)
    settings=state['settings']
    settings.update(lens_mode='optical',f_stop=p['aperture'],aperture=p['aperture'],
                    landing=p['landing'],focus_preset=name)
    return _save_change(state)


def choose_mode(state,mode):
    if mode not in ('legacy','optical'):raise ValueError('Choose Optical or Legacy.')
    state=clip_focus.current(state)
    if mode=='optical':return choose_preset(state,'Cinematic')
    state['settings'].update(lens_mode='legacy',depth_nudge=None,focus_preset=None,nudge_amount=0)
    return _save_change(state)


def controls(state,frame,amount,radius,band):
    state=clip_focus.current(state)
    if frame not in ('Start','End'):raise ValueError('Choose the start or end preview.')
    cfg=directing.nudge_settings(dict(amount=float(amount)/100,radius=float(radius)/100,depth_band=float(band)/100))
    settings=state['settings'];old=settings.get('depth_nudge')
    old=old or {}
    changed=any(abs(old.get(field,settings.get(k,default)/100)*100-v)>1e-8 for k,field,default,v in (
        ('nudge_amount','amount',0,amount),('nudge_radius','radius',35,radius),('nudge_band','depth_band',3.5,band)))
    settings.update(nudge_frame=frame,nudge_amount=amount,nudge_radius=radius,nudge_band=band)
    if old:
        settings['depth_nudge']={**old,**{k:cfg[k] for k in ('amount','radius','depth_band')}}
    if changed and old:return _save_change(state)
    clip_focus.save(state);return state


def select(state,xy):
    state=clip_focus.current(state);s=state['settings']
    if s.get('lens_mode','legacy')!='optical':raise ValueError('Choose Optical to adjust a depth region.')
    x,y=xy
    s['depth_nudge']=directing.nudge_settings(dict(
        x=clip_focus.focus_core.bounded(float(x)/(state['width']-1),0,1,'tap x'),
        y=clip_focus.focus_core.bounded(float(y)/(state['height']-1),0,1,'tap y'),
        anchor_frame=0 if s.get('nudge_frame','Start')=='Start' else state['frames']-1,
        amount=s.get('nudge_amount',0)/100,radius=s.get('nudge_radius',35)/100,
        depth_band=s.get('nudge_band',3.5)/100))
    return _save_change(state)


def reset(state):
    state=clip_focus.current(state);state['settings'].update(depth_nudge=None,nudge_amount=0)
    return _save_change(state)


def panel(state):
    """Return mode, preset, warning, preview, frame, amount, size, band, readout."""
    if not state:return 'optical','Cinematic','',None,'Start',0,35,3.5,''
    state=clip_focus.current(state);s=state['settings'];mode=s.get('lens_mode','legacy')
    frame=s.get('nudge_frame','Start');index=0 if frame=='Start' else state['frames']-1
    image=Image.open(state['first' if frame=='Start' else 'last']).convert('RGB')
    nudge=s.get('depth_nudge');warning='';readout='Tap a region, then move the nudge. 0 keeps its estimated depth.'
    if state.get('depth'):
        d,_=clip_focus.cached_disparity(state)
        if mode=='optical' and nudge is not None:
            cfg=directing.resolve_nudge(d,nudge)
            mask=directing.nudge_mask(d[index],cfg)
            pixels=np.asarray(image,dtype=np.float32)
            pixels=pixels*(1-mask[...,None]*.4)+np.array([255,185,72])*mask[...,None]*.4
            image=Image.fromarray(np.rint(pixels).clip(0,255).astype(np.uint8))
            draw=ImageDraw.Draw(image);x=round(cfg['x']*(state['width']-1));y=round(cfg['y']*(state['height']-1))
            draw.ellipse((x-6,y-6,x+6,y+6),outline='white',width=2)
            readout='Highlighted area: similar estimated depth inside the selected region. Render to apply.'
        if clip_focus.focus_ready(state):
            # Endpoint-only feedback avoids processing the full clip on a tap.
            ends=np.stack((d[0],d[-1]))
            if mode=='optical' and nudge is not None:
                cfg=directing.resolve_nudge(d,nudge);cfg['anchor_frame']=0
                ends=directing.nudge_depth(ends,cfg)
            a,b=state['points'];w,h=state['width'],state['height']
            warning=directing.depth_warning(ends,a[0]/(w-1),a[1]/(h-1),b[0]/(w-1),b[1]/(h-1))['message']
    else:
        warning='Depth feedback appears after the first analysis. Further changes reuse the saved depth.'
        if nudge:readout='Region selected. Its depth mask appears after analysis.'
    if mode=='legacy':readout='Legacy keeps the original depth. Choose Optical to adjust a region.'
    preset=directing.preset_name(s['f_stop'],s['landing']) if mode=='optical' else None
    nudge=nudge or {}
    return (mode,preset,warning,image,frame,round(nudge.get('amount',s.get('nudge_amount',0)/100)*100,6),
            round(nudge.get('radius',s.get('nudge_radius',35)/100)*100,6),
            round(nudge.get('depth_band',s.get('nudge_band',3.5)/100)*100,6),readout)
