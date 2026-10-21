"""Clip-focus state, local planning and explicit, cached Comfy work.

No client is created by loading, clicking, comparing or exporting a study.
Depth belongs to one immutable clip directory. Failed submissions are recorded
and must be recovered instead of silently resubmitted.
"""
from pathlib import Path
import copy,html,json,math,shutil,subprocess,tempfile,uuid,zipfile
from fractions import Fraction
import numpy as np
from PIL import Image,ImageDraw
import backend,focus_core,focus_replay
import optics,focus_schedule,rack_focus,rig_workflow,round_video,video_masks,head_focus,video_occlusion

FOLLOW='Follow what is nearest'
HEAD="Follow the subject's head"
HOLD='Hold the start'
SCHEMA='hero-clip-focus-v1'

def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')

def probe(path):
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(path)]))
    videos=[s for s in data['streams'] if s['codec_type']=='video']
    if len(videos)!=1:raise ValueError('Choose a clip with one video stream.')
    s=videos[0];n=int(s['nb_read_frames']);fps=float(Fraction(s['avg_frame_rate']))
    if not 9<=n<=193 or fps!=25 or s['width']*s['height']>704*1280:
        raise ValueError('Clip focus supports 9–193 frames at 25 fps, up to 704 × 1280 pixels.')
    return {'frames':n,'fps':fps,'width':s['width'],'height':s['height']}

def save(state):dump(Path(state['dir'])/'clip_state.json',state)

def current(state):
    if not state:raise ValueError('Open or animate a clip first.')
    saved=Path(state['dir'])/'clip_state.json'
    return json.loads(saved.read_text(encoding='utf-8')) if saved.exists() else copy.deepcopy(state)

def initialize(clip,lens=85,strength=1.0):
    if not math.isfinite(float(lens)) or not 1<=float(lens)<=200 or not math.isfinite(float(strength)) or not 0<float(strength)<=4:
        raise ValueError('Invalid clip lens or strength metadata.')
    meta=probe(clip);directory=Path(tempfile.mkdtemp(prefix='rig_clip_'))
    sharp=directory/'sharp.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(clip),'-map','0:v:0','-c:v','copy','-an',str(sharp)],check=True)
    for name,index in [('first',0),('last',meta['frames']-1)]:
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(sharp),'-vf',f'select=eq(n\,{index})','-frames:v','1',str(directory/(name+'.png'))],check=True)
    w,h=meta['width'],meta['height']
    state={'schema':SCHEMA,'id':uuid.uuid4().hex,'dir':str(directory),'sharp':str(sharp),
           'first':str(directory/'first.png'),'last':str(directory/'last.png'),**meta,
           'lens_mm':float(lens),'strength':float(strength),'points':[None,None],
           'depth':None,'focused':None,'plan':None,'jobs':[],
           'settings':{'middle':HEAD,'landing':85,'f_stop':1.4,'guide':0,'edge_cleanup':.5}}
    save(state);return state

def marked(state,which):
    # Rings and labels live in the UI, leaving the displayed source pixels intact.
    return Image.open(state[['first','last'][which]]).convert('RGB')


def point_distance(state,which):
    point=state['points'][which]
    if point is None:return 'Choose a point'
    cache=state.get('depth');plan=state.get('plan') or {}
    if cache:
        # Decode only the two endpoint images, once per clip, never the full array.
        endpoint_file=Path(state['dir'])/'endpoint_depth.npy'
        if not endpoint_file.exists():
            with rack_focus._as_sequence(cache['raw']) as sequence:
                streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v','-count_frames',
                    '-show_entries','stream=index,width,height,nb_read_frames','-of','json',str(sequence)]))['streams']
                matches=[s for s in streams if int(s.get('nb_read_frames',0))==state['frames']]
                if len(matches)!=1:raise ValueError('Depth has no unique matching sequence.')
                s=matches[0];w,h=int(s['width']),int(s['height'])
                raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(sequence),'-map',f"0:{s['index']}",
                    '-vf',f"select=eq(n\\,0)+eq(n\\,{state['frames']-1})",'-vsync','0','-pix_fmt','yuv420p10le','-f','rawvideo','-'])
                fsz=(w*h+2*((w+1)//2)*((h+1)//2))*2
                if len(raw)!=2*fsz or (w,h)!=(state['width'],state['height']):raise ValueError('Invalid endpoint depth.')
                y=np.stack([np.frombuffer(raw[i*fsz:i*fsz+w*h*2],dtype='<u2').reshape(h,w) for i in range(2)])
                metres=float(cache['scale'])*np.clip((y.astype(np.float32)-64)/876,0,1)*optics.metric_scale(state['lens_mm'],w,h)
                np.save(endpoint_file,metres)
        value=optics.sample(np.load(endpoint_file,mmap_mode='r')[which],*point)
        return f'{value:.2f} m (est.)'
    phases_=plan.get('phases',[])
    anchor=phases_[0 if which==0 else -1] if phases_ else {}
    if point==anchor.get('point') and plan.get('z_near') and plan.get('raw_curve'):
        d=plan['raw_curve'][0 if which==0 else -1]
        if d>0:return f"{plan['z_near']/d:.2f} m (est.)"
    return 'Depth not cached'


def ui_metadata(state):
    labels=[]
    for which in (0,1):
        try:labels.append(point_distance(state,which))
        except (ValueError,OSError,subprocess.SubprocessError):labels.append('Depth unavailable')
    data={'id':state['id'],'width':state['width'],'height':state['height'],'points':state['points'],'labels':labels,'has_focused':bool(state.get('focused'))}
    return f'<span id="rig-clip-data" data-clip="{html.escape(json.dumps(data),quote=True)}"></span>'

def select(state,which,xy):
    state=current(state);x,y=xy
    state['points'][which]=[min(state['width']-1,max(0,int(x))),min(state['height']-1,max(0,int(y)))]
    save(state);return state

def job_count(state):
    cached=bool(state and state.get('depth'))
    middle=(state or {}).get('settings',{}).get('middle',FOLLOW)
    return int(backend.config()['kind']==backend.LOCAL)+int(not cached)+int(cached and middle==HEAD and not state.get('head_matte'))

def focus_ready(state):
    return bool(state and len(state.get('points',[]))==2 and all(p is not None for p in state['points']))

def button_label(state):
    points=state.get('points',[None,None]) if state else [None,None]
    if not focus_ready(state):
        if points[0] is None and points[1] is None:return 'Click a point on the first and last frames'
        return 'Click a point on the '+('first' if points[0] is None else 'last')+' frame'
    return f'Direct focus on clip — {job_count(state)} cloud '+('job' if job_count(state)==1 else 'jobs')

def phases(state,middle,landing):
    if middle not in (FOLLOW,HEAD,HOLD):raise ValueError('Choose how focus behaves through the middle.')
    landing=float(landing)
    if not math.isfinite(landing) or not 50<=landing<=95:raise ValueError('Pull landing must be 50–95%.')
    if any(p is None for p in state['points']):raise ValueError('Click a focus point on both the first and last frames.')
    n=state['frames'];start=max(1,int(n*.12));end=min(n-1,int(n*landing/100));pull=max(start+1,end-max(1,int(n*.13)))
    a,b=state['points'];w,h=state['width'],state['height']
    return [{'start':0,'end':start,'mode':'fixed','point':a,'anchor_frame':0},
            ({'start':start,'end':pull,'mode':'head'} if middle==HEAD else
             {'start':start,'end':pull,'mode':'nearest','percentile':90,'box':[round(w*.3),round(h*.2),round(w*.7),round(h*.8)]} if middle==FOLLOW else
             {'start':start,'end':pull,'mode':'fixed','point':a,'anchor_frame':0}),
            {'start':pull,'end':end,'mode':'pull','point':b,'anchor_frame':n-1},
            {'start':end,'end':n,'mode':'fixed','point':b,'anchor_frame':n-1}]

def plot(plan):
    if not plan:return '<p class="clip-plot-empty">Your focus path appears here after you press Direct it.</p>'
    values=plan['render_curve'];n=len(values);w,h=900,170
    # Fit both recorded curves; a constant curve still gets a finite, honest axis.
    extent=[float(v) for key in ['raw_curve','render_curve'] for v in plan[key]]
    low,high=min(extent),max(extent)
    padding=max((high-low)*.12,.0001)
    low-=padding;high+=padding
    lines=[]
    for key,color in [('raw_curve','#66616A'),('render_curve','#EBB67E')]:
        pts=' '.join(f'{40+i/(n-1)*820:.2f},{126-(float(v)-low)/(high-low)*96:.2f}' for i,v in enumerate(plan[key]))
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
    names=[]
    for i,p in enumerate(plan['phases']):
        x=40+p['start']/(n-1)*820
        name={'head':'follow head','nearest':'follow nearest','pull':'pull','fixed':'hold start' if i<len(plan['phases'])-1 else 'hold end'}.get(p['mode'],str(p['mode']))
        names.append(f'<span>{html.escape(name)}</span>')
        lines.append(f'<line x1="{x}" y1="22" x2="{x}" y2="130" stroke="#5D565F"/>')
    duration=(n-1)/25
    return f'<div class="clip-phase-labels">{"".join(names)}</div><div class="clip-scale-label">↑ Nearer · vertical scale fitted to this clip</div><svg viewBox="0 0 {w} {h}" role="img" aria-label="Estimated-depth focus schedule, nearer upward" data-duration="{duration}" data-depth-min="{low}" data-depth-max="{high}" style="width:100%;fill:#B7B0A4"><line x1="40" y1="130" x2="860" y2="130" stroke="#5B555D"/>{"".join(lines)}<line class="clip-playhead" x1="40" x2="40" y1="22" y2="130" stroke="#ECE8E1" stroke-width="2"/><circle class="clip-playhead-dot" cx="40" cy="130" r="4" fill="#ECE8E1"/><text x="40" y="160">0 s</text><text x="685" y="160">{duration:.2f} s · focus depth</text></svg><div class="clip-playback-time" aria-live="off">0.00 s</div>'

def load_study(root):
    root=Path(root);meta=json.loads((root/'study.json').read_text())
    state=initialize(root/'sharp.mp4',meta['lens_mm'],meta['strength'])
    target=Path(state['dir'])/'saved_focused.mp4';shutil.copy2(root/'focused.mp4',target)
    state.update(focused=str(target),plan=meta['plan'],points=meta['points'],study=meta['name'],
                 settings={'middle':FOLLOW,'landing':93,'f_stop':1.4,'guide':60},
                 provenance={'sharp_job':meta['sharp_job'],'focus_job':meta['focus_job'],'kind':'saved placeholder study'})
    # No depth is bundled: this remains two jobs unless a clip recipe is reopened.
    save(state);return state

def export_recipe(state):
    state=current(state);directory=Path(state['dir']);out=directory/f'clip_recipe_{uuid.uuid4().hex[:8]}.zip'
    paths={};meta={k:state[k] for k in ['schema','frames','fps','width','height','lens_mm','strength','points','settings','plan','jobs']}
    for role in ['sharp','focused']:
        if state.get(role):paths[role]=(Path(state[role]),role+'.mp4')
    if state.get('head_matte'):paths['head_matte']=(Path(state['head_matte']),'subject.avif')
    if state.get('depth'):
        meta['depth']={k:v for k,v in state['depth'].items() if k not in ['raw','sky']}
        paths['depth_raw']=(Path(state['depth']['raw']),'depth.avif');paths['depth_sky']=(Path(state['depth']['sky']),'sky.avif')
    replay_depth=directory/'focus/hfr_depth.npz'
    if state.get('focused') and state.get('plan') and replay_depth.exists():
        paths['render_depth']=(replay_depth,'hfr_depth.npz')
    meta['files']={role:name for role,(_,name) in paths.items()}
    meta['provenance']=state.get('provenance',{})
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('clip_recipe.json',json.dumps(meta,indent=2))
        if 'render_depth' in paths:
            z.writestr('workflow.json',json.dumps(focus_replay.workflow(state['plan']),indent=2))
            z.writestr('REPLAY.txt','Install ComfyUI-HeroFrameRig in custom_nodes. Copy sharp.mp4 and hfr_depth.npz into ComfyUI/input. Submit workflow.json (API format) using the SDK or POST /prompt. Uses the exact saved float depth and focus curve; no model download or generation. Seed 7.\n')
        z.writestr('README.txt','Hero Frame Rig clip recipe. Estimated depth, stylized focus, no video subject hold. Reopen this ZIP to reuse its depth. Graphs and lossless masks retain the latest focus render. No graph in an imported ZIP is executed.\n')
        for path,name in paths.values():z.write(path,name)
        for folder in ['depth','focus','head']:
            base=directory/folder
            if base.is_dir():
                for path in base.rglob('*'):
                    if path.is_file() and path.suffix in ('.json','.mp4') and path.name!='focused.mp4':z.write(path,path.relative_to(directory).as_posix())
    return str(out)

def import_recipe(path):
    if not path:raise ValueError('Choose a clip recipe ZIP first.')
    directory=Path(tempfile.mkdtemp(prefix='rig_clip_import_'))
    with zipfile.ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist())>600_000_000 or len(z.namelist())>100:raise ValueError('Clip recipe is too large.')
        names=z.namelist()
        if len(set(names))!=len(names):raise ValueError('Duplicate recipe members.')
        for name in names:
            if '\\' in name or ':' in name or name.startswith('/') or '..' in Path(name).parts:raise ValueError('Unsafe recipe path.')
        meta=json.loads(z.read('clip_recipe.json'))
        if meta.get('schema')!=SCHEMA:raise ValueError('This is not a clip-focus recipe.')
        files=meta['files']
        for role,name in files.items():
            if name not in names or not (directory/name).resolve().is_relative_to(directory.resolve()):raise ValueError('Missing recipe input.')
        z.extractall(directory)
    state=initialize(directory/files['sharp'],meta['lens_mm'],meta['strength'])
    for key in ['frames','fps','width','height']:
        if meta[key]!=state[key]:raise ValueError('Recipe and clip dimensions do not match.')
    state.update(points=meta['points'],settings=meta['settings'],plan=meta.get('plan'),provenance=meta.get('provenance',{}),jobs=meta.get('jobs',[]))
    if focus_ready(state):phases(state,state['settings']['middle'],state['settings']['landing'])
    if 'focused' in files:
        p=Path(state['dir'])/'imported_focused.mp4';shutil.copy2(directory/files['focused'],p);state['focused']=str(p)
    if 'head_matte' in files:
        p=Path(state['dir'])/'subject.avif';shutil.copy2(directory/files['head_matte'],p);state['head_matte']=str(p)
    if meta.get('depth'):
        cached=copy.deepcopy(meta['depth']);scale=float(cached['scale'])
        if not math.isfinite(scale) or not 0<scale<=1024:raise ValueError('Invalid depth packing scale.')
        for role,name in [('raw','depth_raw'),('sky','depth_sky')]:
            p=Path(state['dir'])/('cached_'+role+'.avif');shutil.copy2(directory/files[name],p);cached[role]=str(p)
        cached['clip_id']=state['id'];state['depth']=cached
    if 'render_depth' in files:
        target=Path(state['dir'])/'focus/hfr_depth.npz';target.parent.mkdir(exist_ok=True)
        shutil.copy2(directory/files['render_depth'],target)
    # Keep portable render evidence when a saved recipe is reopened and exported.
    # Imported graphs are retained as data and are never executed.
    for folder in ('depth','focus','head'):
        source=directory/folder
        if source.is_dir():shutil.copytree(source,Path(state['dir'])/folder,dirs_exist_ok=True)
    save(state);return state

def run_job(client,state,kind,graph,bindings,outputs):
    if any(j['status']!='succeeded' for j in state['jobs']):
        raise ValueError('An earlier clip job needs recovery before another submission. Its job ID is saved in the recipe.')
    work=client.workflows.from_json(copy.deepcopy(graph))
    for node,key,path in bindings:work.set_input(node,key,client.assets.from_file(path))
    record={'kind':kind,'status':'reserved'};state['jobs'].append(record);save(state)
    try:
        job=client.submit(work);record.update(id=job.id,status='submitted');save(state)
        print(f'rig clip={state["id"]} stage={kind} submitted_job={job.id}',flush=True)
        result=job.result();directory=Path(state['dir'])/kind;directory.mkdir(exist_ok=True)
        dump(directory/'graph.json',graph)
        for node,name in outputs.items():result.get_outputs(node)[0].to_file(str(directory/name))
        record['status']='succeeded';save(state)
        return directory,job.id
    except Exception:
        record['status']='needs_recovery';save(state);raise

def estimate_depth(client,state,include_head=False):
    graph=head_focus.graph(with_depth=True) if include_head else rig_workflow.clip_depth('sharp.mp4',depth_scale=4)
    outputs={'8':'depth.avif','10':'sky.avif'}
    if include_head:outputs['25']='subject.avif'
    directory,job=run_job(client,state,'depth',graph,[('1','file',state['sharp'])],outputs)
    if include_head:state['head_matte']=str(directory/'subject.avif')
    state['depth']={'raw':str(directory/'depth.avif'),'sky':str(directory/'sky.avif'),'scale':4,'job':job,'clip_id':state['id']}
    save(state)

def render_focus(client,state,f_stop,guide,middle,landing,progress=lambda message:None,edge_cleanup=None):
    state=current(state);requested=phases(state,middle,landing)
    edge_cleanup=focus_core.bounded(state['settings'].get('edge_cleanup',0) if edge_cleanup is None else edge_cleanup,0,1,'edge cleanup')
    f_stop=float(f_stop);guide=float(guide)
    if not math.isfinite(f_stop) or not 1.4<=f_stop<=5.6 or not math.isfinite(guide) or not 0<=guide<=100:raise ValueError('Invalid aperture or guide value.')
    if not state.get('depth'):raise ValueError('Estimate this clip depth first.')
    cache=state['depth']
    if cache['clip_id']!=state['id']:raise ValueError('Depth belongs to a different clip.')
    progress('Reading cached estimated depth…')
    n=state['frames'];z,sky=rack_focus.read_clip_depth(cache['raw'],cache['sky'],depth_scale=cache['scale'],expected_frames=n)
    if z.shape!=(n,state['height'],state['width']):raise ValueError('Depth dimensions do not match the clip.')
    if np.max(z)>=cache['scale']:raise ValueError('Depth clipping detected; inspect the recipe before rendering.')
    z*=optics.metric_scale(state['lens_mm'],state['width'],state['height'])
    near=optics.z_near_for([np.where(sky[i]<.5,z[i],np.inf) for i in range(n)])
    d=np.stack([optics.disparity(z[i],near,sky[i]) for i in range(n)]);del z,sky
    blur=min(optics.blur_scale(state['lens_mm'],f_stop,state['width'],near,state['strength']),optics.MAX_BLUR)
    head=None
    if middle==HEAD:
        progress('Finding the subject…')
        if not state.get('head_matte'):
            progress('Finding the subject for head focus — one backend job…')
            folder,_=run_job(client,state,'head',head_focus.graph(),[('1','file',state['sharp'])],{'25':'subject.avif'})
            state['head_matte']=str(folder/'subject.avif');save(state)
        head=head_focus.targets(state['head_matte'],d)
    a,b=state['points'];w,h=state['width'],state['height']
    mode={HOLD:'stay',FOLLOW:'closest',HEAD:'face'}[middle]
    plan=focus_core.make_plan(d,a[0]/(w-1),a[1]/(h-1),b[0]/(w-1),b[1]/(h-1),blur,(float(landing)-50)/45,mode,head['targets'] if head else None)
    if head:plan['head_tracking']=head
    plan.update(frames=n,fps=25,width=state['width'],height=state['height'],lens_mm=state['lens_mm'],f_stop=f_stop,
                guide=guide,subject_hold=False,look='digital',strength=state['strength'],blur=blur,z_near=near,
                depth_packing_scale=cache['scale'],points=state['points'],middle=middle,pull_lands_at_percent=float(landing),lens_version=focus_core.VERSION,seed=7,edge_cleanup=edge_cleanup)
    directory=Path(state['dir'])/'focus';directory.mkdir(exist_ok=True);dump(directory/'plan.json',plan)
    depth_file=directory/'hfr_depth.npz'
    np.savez_compressed(depth_file,depth=d)
    graph=focus_replay.workflow(plan)
    dump(directory/'workflow.json',graph)
    if getattr(client,'hfr_backend',backend.CLOUD)==backend.LOCAL:
        progress('Rendering focus in local ComfyUI — one local node job…')
        directory,job=run_job(client,state,'focus',graph,
            [('1','file',state['sharp']),('3','file',str(depth_file))],{'6':'focused.mp4'})
        result=directory/f'focused_{job}.mp4';shutil.copy2(directory/'focused.mp4',result)
    else:
        progress('Rendering focus locally in Python — no additional backend job…')
        result=directory/f'focused_{uuid.uuid4().hex[:12]}.mp4'
        focus_replay.render_python(state['sharp'],d,plan,result)
    del d
    if probe(result)!={k:state[k] for k in ['frames','fps','width','height']}:raise ValueError('Rendered clip dimensions differ.')
    state.update(focused=str(result),plan=plan,settings={'middle':middle,'landing':float(landing),'f_stop':f_stop,'guide':guide,'edge_cleanup':edge_cleanup})
    save(state);return state
