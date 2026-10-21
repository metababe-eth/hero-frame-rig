"""Deterministic scheduled focus, with explicit half-open phase boundaries."""
import numpy as np
from . import _support as optics
from ._support import smooth, smoothstep, patch_level, segments, MAX_SEGMENTS, BLUR_STEP_PX, _anchor

VERSION='scheduled-focus-v2'
def curve(disparities, phases, blur, head_targets=None):
    if not np.isfinite(blur) or blur<=0:raise ValueError('Blur must be positive and finite.')
    if not np.isfinite(disparities).all():raise ValueError('Disparity must be finite.')
    n,h,w=disparities.shape
    raw=np.empty(n,dtype=np.float64);cursor=0
    for phase in phases:
        start,end=phase['start'],phase['end']
        if start!=cursor or not start<end<=n:raise ValueError('Phases must cover every frame once, in order.')
        mode=phase['mode']
        if mode in ('fixed','pull'):
            x,y=_anchor(phase['point'],w,h)
            anchor=phase['anchor_frame']
            if not 0<=anchor<n:raise ValueError('Anchor frame outside clip.')
            target=patch_level(disparities[anchor],x,y)
        if mode=='fixed':raw[start:end]=target
        elif mode=='nearest':
            x0,y0,x1,y1=phase['box']
            if not 0<=x0<x1<=w or not 0<=y0<y1<=h:raise ValueError('Empty or invalid focus box.')
            if not 0<=phase['percentile']<=100:raise ValueError('Invalid percentile.')
            for i in range(start,end):raw[i]=np.percentile(disparities[i,y0:y1,x0:x1],phase['percentile'])
        elif mode=='head':
            if head_targets is None or len(head_targets)!=n or not np.isfinite(head_targets).all():raise ValueError('Head targets must cover every frame.')
            raw[start:end]=head_targets[start:end]
        elif mode=='pull':
            if start==0:raise ValueError('Pull requires a previous phase.')
            initial=raw[start-1]
            for i in range(start,end):raw[i]=initial+(target-initial)*smoothstep((i-start+1)/(end-start))
        else:raise ValueError('Unknown focus mode.')
        cursor=end
    if cursor!=n or not np.isfinite(raw).all():raise ValueError('Incomplete focus schedule.')
    window=9;values=smooth(raw,window)
    if np.max(np.abs(np.diff(values)))>optics.MIN_FOCUS_RANGE:
        window=15;values=smooth(raw,window)
    smoothed=values.copy()
    # Limit estimated automatic motion. Deliberately scheduled pulls have a deadline;
    # limiting them to the tracking slew can leave the chosen endpoint out of focus.
    slew=optics.MIN_FOCUS_RANGE/2
    for phase in phases:
        start,end=phase['start'],phase['end']
        if phase['mode']=='fixed':values[start:end]=raw[start:end]
        elif phase['mode']=='nearest':
            for i in range(max(1,start),end):values[i]=np.clip(smoothed[i],values[i-1]-slew,values[i-1]+slew)
        elif phase['mode']=='head':
            # Acquire the head in eight frames instead of spending the whole
            # approach chasing it from a deliberately distant start target.
            tracked=smooth(raw[start:end],9)
            acquire=min(8,end-start);initial=values[start-1] if start else tracked[0]
            for j in range(end-start):
                i=start+j
                if j<acquire:values[i]=initial+(tracked[j]-initial)*smoothstep((j+1)/acquire)
                else:values[i]=np.clip(tracked[j],values[i-1]-.02,values[i-1]+.02)
        elif phase['mode']=='pull':
            initial=values[start-1];target=raw[end-1]
            for i in range(start,end):values[i]=initial+(target-initial)*smoothstep((i-start+1)/(end-start))
    step=max(BLUR_STEP_PX/blur,.002)
    grouped=segments(values,step)
    while len(grouped)>MAX_SEGMENTS:
        step*=1.25;grouped=segments(values,step)
    rendered=np.empty(n)
    for seg in grouped:rendered[seg['start']:seg['start']+seg['length']]=seg['focus']
    grouping='blur tolerance'
    automatic=np.ones(n-1,dtype=bool)
    for phase in phases:
        if phase['mode'] in ('pull','fixed'):automatic[max(0,phase['start']-1):phase['end']-1]=False
    if automatic.any() and np.max(np.abs(np.diff(rendered))[automatic])>optics.MIN_FOCUS_RANGE+1e-8:
        # Averaging variable length segments can amplify a boundary jump.
        # A one-band grid gives a bounded jump and <= half-band focus error.
        rendered=np.round(values/optics.MIN_FOCUS_RANGE)*optics.MIN_FOCUS_RANGE
        grouped=segments(rendered,optics.MIN_FOCUS_RANGE/2)
        grouping='one focus band grid'
        # This fallback feeds per-frame masks; its descriptive segment count
        # does not expand the render graph or need the old 50-node cap.
    # Quantization may shift a deliberate landing slightly; keep pull/fixed samples
    # exact in the per-frame renderer. Segments are descriptive for legacy exports.
    for phase in phases:
        if phase['mode'] in ('pull','fixed'):rendered[phase['start']:phase['end']]=values[phase['start']:phase['end']]
    grouped=segments(rendered,1e-10)
    return {'version':VERSION,'phase_boundary_convention':'start inclusive, end exclusive',
            'phases':phases,'raw_curve':raw.tolist(),'smoothed_curve':smoothed.tolist(),'focus_curve':values.tolist(),
            'automatic_slew_limit':slew,'max_step_before_slew':float(np.max(np.abs(np.diff(smoothed)))),
            'head_tracking_filter':{'acquisition_frames':8,'tracking_slew_limit':.02,'smoothing_window':9},
            'slew_scope':'automatic tracking; deliberate pulls and holds preserve their scheduled target',
            'render_curve':rendered.tolist(),'smoothing_window':window,
            'max_adjacent_jump':float(np.max(np.abs(np.diff(values)))),
            'focus_band':optics.MIN_FOCUS_RANGE,'max_render_jump':float(np.max(np.abs(np.diff(rendered)))),
            'segments':grouped,'segment_step':step,'grouping':grouping}

def toast_phases(a,b,width,height,frames=121):
    if type(frames) is not int or frames<9:
        raise ValueError('Toast schedule needs at least nine frames.')
    first,approach,settle=(int(frames*p) for p in (.12,.80,.93))
    return [{'start':0,'end':first,'mode':'fixed','point':list(a),'anchor_frame':0},
            {'start':first,'end':approach,'mode':'nearest','percentile':90,
             'box':[round(width*.3),round(height*.2),round(width*.7),round(height*.8)]},
            {'start':approach,'end':settle,'mode':'pull','point':list(b),'anchor_frame':frames-1},
            {'start':settle,'end':frames,'mode':'fixed','point':list(b),'anchor_frame':frames-1}]
