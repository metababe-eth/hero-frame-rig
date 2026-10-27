"""Three fixed image-space focus areas; no depth estimation or subject tracking.

Local source pixels supply every output detail. The recipe is a bounded media
archive, never executable code or a graph. Existing depth-focus paths are separate.
"""
from pathlib import Path
from fractions import Fraction
from contextlib import suppress
import copy, hashlib, json, math, shutil, subprocess, tempfile, zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw

SCHEMA = 'hero-area-focus-v1'
MAX_BYTES = 250_000_000
MAX_FRAMES = 288
ANCHOR_FRACTIONS = (.0625, .375, .65625)
DEFAULTS = dict(blur=44., radii=[.14, .14, .24], arrival2=32., arrival3=63., pull=12., resolve=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def _run(args):
    p = subprocess.run([str(x) for x in args], capture_output=True, timeout=120)
    if p.returncode: raise ValueError(p.stderr.decode(errors='replace')[-1000:])
    return p.stdout


def probe(path, prepared=False):
    # A recipe member named .mp4 must actually be a supported media container,
    # never a renamed playlist that can direct the decoder to another resource.
    data = json.loads(_run(['ffprobe', '-v', 'error', '-format_whitelist', 'mov', '-protocol_whitelist', 'file,pipe',
                           '-enable_drefs', '0', '-count_frames', '-show_streams', '-of', 'json', path]))
    streams = [s for s in data['streams'] if s['codec_type'] == 'video' and not s.get('disposition', {}).get('attached_pic')]
    if len(streams) != 1: raise ValueError('Choose a clip containing one video stream.')
    s = streams[0]
    rate = Fraction(s['avg_frame_rate']); fps = float(rate)
    n = int(s['nb_read_frames']); w, h = int(s['width']), int(s['height'])
    if not 9 <= n <= MAX_FRAMES or not 1 <= fps <= 60 or n / fps > 12.001:
        raise ValueError('Area focus supports up to 12 seconds and 288 frames. Trim the clip first.')
    if w * h > 16_777_216 or min(w, h) < 32: raise ValueError('Unsupported source dimensions.')
    if prepared and (max(w, h) > 1280 or w % 2 or h % 2): raise ValueError('Invalid prepared clip size.')
    if abs(float(Fraction(s['r_frame_rate'])) - fps) > .01:
        raise ValueError('Choose a clip with a constant frame rate.')
    return dict(frames=n, fps=fps, rate=str(rate), width=w, height=h, duration=n / fps)


def settings(blur=44, r1=.14, r2=.14, r3=.24, arrival2=32, arrival3=63, pull=12, resolve=True):
    nums = [float(v) for v in (blur, r1, r2, r3, arrival2, arrival3, pull)]
    if not all(math.isfinite(v) for v in nums): raise ValueError('Settings must be finite.')
    blur, r1, r2, r3, a2, a3, pull = nums
    if not 0 <= blur <= 100 or any(not .03 <= r <= .40 for r in (r1, r2, r3)):
        raise ValueError('Choose valid blur and area sizes.')
    if not 15 <= a2 <= 48 or not 50 <= a3 <= 85 or not 2 <= pull <= 20 or a2 < pull or a3 - a2 <= pull:
        raise ValueError('Focus arrivals must leave room for both transitions.')
    if type(resolve) is not bool: raise ValueError('Resolve must be true or false.')
    return dict(blur=blur, radii=[r1, r2, r3], arrival2=a2, arrival3=a3, pull=pull, resolve=resolve)


def validated_settings(s):
    if not isinstance(s, dict) or set(s) != set(DEFAULTS): raise ValueError('Invalid area settings.')
    radii = s['radii']
    if not isinstance(radii, list) or len(radii) != 3: raise ValueError('Three area sizes are required.')
    return settings(s['blur'], *radii, s['arrival2'], s['arrival3'], s['pull'], s['resolve'])


def validated_points(points, complete=True):
    if not isinstance(points, list) or len(points) != 3: raise ValueError('Three focus areas are required.')
    result = []
    for point in points:
        if point is None and not complete: result.append(None); continue
        if not isinstance(point, list) or len(point) != 2: raise ValueError('Select all three details first.')
        values = [float(x) for x in point]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in values): raise ValueError('Invalid focus point.')
        result.append(values)
    return result


def save_state(state):
    (Path(state['dir']) / 'area_state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')


def _snapshots(state):
    cap = cv2.VideoCapture(state['source'])
    state['anchors'] = []
    for i, fraction in enumerate(ANCHOR_FRACTIONS):
        n = min(state['frames'] - 1, round((state['frames'] - 1) * fraction))
        cap.set(cv2.CAP_PROP_POS_FRAMES, n); ok, frame = cap.read()
        if not ok: cap.release(); raise ValueError('Could not read the selected frame.')
        target = Path(state['dir']) / f'detail_{i+1}.png'
        cv2.imwrite(str(target), frame)
        state['anchors'].append(dict(frame=n, seconds=n / state['fps'], image=str(target)))
    cap.release()


def open_clip(path):
    if not path or Path(path).suffix.lower() not in ('.mp4', '.mov', '.m4v'):
        raise ValueError('Upload an MP4 or MOV clip.')
    path = Path(path)
    if path.stat().st_size > MAX_BYTES: raise ValueError('Choose a clip smaller than 250 MB.')
    original = probe(path)
    directory = Path(tempfile.mkdtemp(prefix='hfr_area_'))
    source = directory / 'source.mp4'
    original_file = directory / ('original' + path.suffix.lower())
    shutil.copy2(path, original_file)
    scale = min(1., 1280 / max(original['width'], original['height']))
    w = max(2, math.floor(original['width'] * scale / 2) * 2)
    h = max(2, math.floor(original['height'] * scale / 2) * 2)
    _run(['ffmpeg', '-v', 'error', '-n', '-format_whitelist', 'mov', '-protocol_whitelist', 'file,pipe', '-enable_drefs', '0',
          '-i', path, '-map', '0:v:0', '-vf', f'scale={w}:{h}:flags=lanczos,setsar=1',
          '-fps_mode', 'passthrough', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', source])
    meta = probe(source, prepared=True)
    if meta['frames'] != original['frames'] or meta['rate'] != original['rate']:
        raise ValueError('The prepared clip changed the source timeline.')
    state = dict(schema=SCHEMA, dir=str(directory), source=str(source), original=str(original_file),
                 upload_name=path.name, source_sha256=sha(source), original_sha256=sha(original_file),
                 points=[None, None, None], settings=copy.deepcopy(DEFAULTS), result=None, **meta)
    _snapshots(state); save_state(state)
    return state


def selected(state, which, xy):
    if not state: raise ValueError('Upload a clip first.')
    if type(which) is not int or which not in (0, 1, 2): raise ValueError('Invalid detail index.')
    state = copy.deepcopy(state)
    x, y = (float(v) for v in xy)
    if not all(math.isfinite(v) for v in (x, y)): raise ValueError('Invalid point.')
    state['points'][which] = [min(1., max(0., x / (state['width'] - 1))), min(1., max(0., y / (state['height'] - 1)))]
    state['result'] = None
    save_state(state); return state


def marked(state, which):
    im = Image.open(state['anchors'][which]['image']).convert('RGB')
    point = state['points'][which]
    if point:
        x, y = point[0] * (im.width - 1), point[1] * (im.height - 1)
        d = ImageDraw.Draw(im); r = max(8, im.width / 60)
        d.ellipse((x-r, y-r, x+r, y+r), outline='#f9ecab', width=max(2, im.width // 350))
        d.line((x-r*1.5, y, x+r*1.5, y), fill='#f9ecab', width=2)
        d.line((x, y-r*1.5, x, y+r*1.5), fill='#f9ecab', width=2)
    return im


def smooth(value):
    v = np.clip(value, 0, 1); return v * v * (3 - 2 * v)


def masks(width, height, points, radii):
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    result = []
    for (px, py), radius in zip(validated_points(points), radii):
        # Elliptical area, with a flat sharp centre and continuously feathered edge.
        rx = width * radius; ry = rx * 1.20
        d = np.sqrt(((xx-px*(width-1))/rx)**2 + ((yy-py*(height-1))/ry)**2)
        result.append((1-smooth((d-.52)/.48)).astype(np.float32))
    return result


def weights(index, frames, s):
    percent = index / max(1, frames-1) * 100
    a = float(smooth((percent-(s['arrival2']-s['pull']))/s['pull']))
    b = float(smooth((percent-(s['arrival3']-s['pull']))/s['pull']))
    return [1-a, a*(1-b), b]


def process_frame(frame, regions, index, frames, s):
    if s['blur'] == 0: return frame.copy()
    w = weights(index, frames, s)
    sharp = sum(weight * region for weight, region in zip(w, regions))
    if s['resolve']:
        restore = float(smooth((index / max(1, frames-1) - .89) / .10))
        sharp = sharp + (1-sharp) * restore
    sigma = max(.01, (s['blur']/100) * 18.0 * frame.shape[1]/1280)
    soft = cv2.GaussianBlur(frame, (0, 0), sigma)
    mixed = np.rint(frame.astype(np.float32)*sharp[..., None] + soft.astype(np.float32)*(1-sharp[..., None]))
    out = np.clip(mixed, 0, 255).astype(np.uint8)
    out[sharp >= 1-1e-7] = frame[sharp >= 1-1e-7]
    return out


def render(state, s=None, progress=lambda i, n: None):
    if not state: raise ValueError('Upload a clip first.')
    state = copy.deepcopy(state); points = validated_points(state['points'])
    s = validated_settings(s or state['settings'])
    if sha(state['source']) != state['source_sha256']: raise ValueError('Source changed; reopen the clip.')
    regions = masks(state['width'], state['height'], points, s['radii'])
    out = Path(state['dir']) / f"area_result_{len(list(Path(state['dir']).glob('area_result_*.mp4'))):03d}.mp4"
    encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f"{state['width']}x{state['height']}",
                                '-r', state['rate'], '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)],
                               stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    cap = cv2.VideoCapture(state['source']); count = 0; complete = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            encoder.stdin.write(process_frame(frame, regions, count, state['frames'], s).tobytes())
            count += 1
            if count % 6 == 0: progress(count, state['frames'])
        encoder.stdin.close()
        error = encoder.stderr.read(); code = encoder.wait(timeout=30)
        if code: raise ValueError(error.decode(errors='replace')[-1000:])
        meta = probe(out, prepared=True)
        if count != state['frames'] or any(meta[k] != state[k] for k in ('frames', 'fps', 'width', 'height')):
            raise ValueError('Result does not match the source timeline.')
        state.update(result=str(out), settings=s, result_sha256=sha(out))
        state['render_record'] = dict(engine='local-area-gaussian-v1', tracking=False, depth_estimation=False, cloud_jobs=0,
                                      points=points, anchors=[{k:a[k] for k in ('frame', 'seconds')} for a in state['anchors']], settings=s,
                                      source_sha256=state['source_sha256'], result_sha256=state['result_sha256'])
        save_state(state); progress(count, state['frames']); complete = True
        return state
    finally:
        cap.release()
        with suppress(BrokenPipeError, OSError): encoder.stdin.close()
        if encoder.poll() is None: encoder.kill()
        encoder.wait(timeout=10)
        encoder.stderr.close()
        if not complete:
            # This is only the new versioned render in our own temp directory.
            with suppress(FileNotFoundError): out.unlink()


def export_recipe(state):
    if not state or not state.get('result'): raise ValueError('Render the area sequence before saving.')
    validated_points(state['points']); validated_settings(state['settings'])
    paths = {'source.mp4':Path(state['source']), 'result.mp4':Path(state['result']), 'original'+Path(state['original']).suffix:Path(state['original'])}
    manifest = {k:state[k] for k in ('schema', 'frames', 'fps', 'rate', 'width', 'height', 'duration', 'points', 'settings', 'upload_name', 'render_record')}
    manifest['files'] = {name:dict(bytes=p.stat().st_size, sha256=sha(p)) for name,p in paths.items()}
    if any(f['bytes'] > MAX_BYTES for f in manifest['files'].values()) or sum(f['bytes'] for f in manifest['files'].values()) > 599_936_000:
        raise ValueError('This area recipe exceeds the media archive limits.')
    target = Path(state['dir'])/'area_focus_recipe.zip'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_STORED) as z:
        z.writestr('recipe.json', json.dumps(manifest, indent=2))
        for name,path in paths.items(): z.write(path,name)
    return str(target)


def import_recipe(path):
    if not path or Path(path).stat().st_size > 600_000_000: raise ValueError('Choose an area recipe smaller than 600 MB.')
    with zipfile.ZipFile(path) as z:
        infos=z.infolist(); names=[i.filename for i in infos]
        if len(names)!=len(set(names)) or len(names)!=4 or 'recipe.json' not in names:
            raise ValueError('Invalid area recipe members.')
        if set(names)-{'recipe.json','source.mp4','result.mp4','original.mp4','original.mov','original.m4v'}:
            raise ValueError('Unexpected path or member in area recipe.')
        if not {'source.mp4', 'result.mp4'}.issubset(names) or sum(n.startswith('original.') for n in names) != 1:
            raise ValueError('Area recipes require source, result and exactly one original.')
        if any(i.flag_bits&1 or i.file_size>MAX_BYTES or i.is_dir() for i in infos) or sum(i.file_size for i in infos)>600_000_000:
            raise ValueError('Area recipe exceeds limits.')
        if z.getinfo('recipe.json').file_size>64_000: raise ValueError('Area recipe metadata is too large.')
        m=json.loads(z.read('recipe.json').decode('utf-8'))
        if m.get('schema')!=SCHEMA or set(m.get('files',{})) != set(names)-{'recipe.json'}:
            raise ValueError('This is not an area-focus recipe.')
        points=validated_points(m['points']); s=validated_settings(m['settings'])
        if z.testzip(): raise ValueError('Recipe integrity check failed.')
        payloads={}
        for name,record in m['files'].items():
            data=z.read(name)
            if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
                raise ValueError('Recipe media checksum mismatch.')
            payloads[name]=data
    directory=Path(tempfile.mkdtemp(prefix='hfr_area_recipe_'))
    for name,data in payloads.items(): (directory/name).write_bytes(data)
    source=directory/'source.mp4'; result=directory/'result.mp4'; original=next(directory/n for n in payloads if n.startswith('original.'))
    meta=probe(source,prepared=True); result_meta=probe(result,prepared=True)
    if result_meta!=meta or any(meta[k]!=m[k] for k in meta): raise ValueError('Recipe media metadata mismatch.')
    state=dict(schema=SCHEMA,dir=str(directory),source=str(source),result=str(result),original=str(original),
               source_sha256=sha(source),result_sha256=sha(result),original_sha256=sha(original),upload_name=str(m.get('upload_name','clip')),
               points=points,settings=s,render_record=m.get('render_record',{}),**meta)
    _snapshots(state);save_state(state);return state


def build_ui():
    """Called inside the real app's Use my clip / Area sequence column."""
    import gradio as gr
    state=gr.State(None)
    gr.Markdown('### Pick three details. Direct the order.')
    gr.Markdown('Keep selected areas sharp while the surroundings soften. Areas stay at the positions you choose; they do not track moving subjects.')
    upload=gr.File(label='Your clip for area focus',type='filepath',file_types=['.mp4','.mov','.m4v'],elem_id='area-upload')
    status=gr.Markdown('Upload a short clip, then tap each detail in its frame.',elem_id='area-status')
    with gr.Column(visible=False,elem_id='area-workbench') as workbench:
        with gr.Row():
            images=[gr.Image(label=f'Detail {i+1}',type='pil',interactive=False,format='png',buttons=[],height=245,min_width=250,elem_id=f'area-detail-{i+1}') for i in range(3)]
        with gr.Row():
            blur=gr.Slider(0,100,value=44,step=1,label='Blur',elem_id='area-blur')
            button=gr.Button('Direct these details',variant='primary',interactive=False,elem_id='area-render')
        with gr.Accordion('Area sizes and timing',open=False,elem_id='area-settings'):
            with gr.Row():
                radii=[gr.Slider(.03,.40,value=v,step=.005,label=f'Detail {i+1} area size',elem_id=f'area-size-{i+1}') for i,v in enumerate(DEFAULTS['radii'])]
            with gr.Row():
                a2=gr.Slider(15,48,value=32,step=1,label='Second detail arrives (%)',elem_id='area-arrival-2')
                a3=gr.Slider(50,85,value=63,step=1,label='Third detail arrives (%)',elem_id='area-arrival-3')
                pull=gr.Slider(2,20,value=12,step=1,label='Transition duration (%)',elem_id='area-pull')
            resolve=gr.Checkbox(value=True,label='Bring everything back into focus at the end',elem_id='area-resolve')
        with gr.Column(visible=False,elem_id='area-results') as results:
            with gr.Row():
                before=gr.Video(label='Before · your source',interactive=False,include_audio=False,loop=True,buttons=[],elem_id='area-before')
                after=gr.Video(label='After · area focus',interactive=False,include_audio=False,loop=True,buttons=[],elem_id='area-after')
            with gr.Row():
                save=gr.Button('Save area recipe',elem_id='area-save')
                download=gr.File(label='Area recipe ZIP',interactive=False,elem_id='area-recipe-download')
    with gr.Accordion('Reopen an area recipe',open=False,elem_id='area-reopen'):
        recipe=gr.File(label='Area recipe ZIP',type='filepath',file_types=['.zip'],elem_id='area-recipe-upload')
        reopen=gr.Button('Open area recipe',elem_id='area-recipe-open')

    controls=[blur,*radii,a2,a3,pull,resolve]
    load_outputs=[state,*images,workbench,button,status,before,after,results,download,*controls]
    def loaded(value):
        s=value['settings']; complete=all(p is not None for p in value['points'])
        rendered=bool(value.get('result'))
        pics=[gr.update(value=marked(value,i),label=f"Detail {i+1} · {value['anchors'][i]['seconds']:.2f}s") for i in range(3)]
        return [value,*pics,gr.update(visible=True),gr.update(interactive=complete),
                'Area recipe reopened. No render or backend job.' if rendered else 'Tap a detail in each frame. The order is 1 → 2 → 3.',
                value['source'],value.get('result'),gr.update(visible=rendered),None,s['blur'],*s['radii'],s['arrival2'],s['arrival3'],s['pull'],s['resolve']]
    def load_clip(path):return loaded(open_clip(path))
    def load_recipe(path):return loaded(import_recipe(path))
    def clear_clip():
        s=DEFAULTS
        return [None,*[gr.update(value=None,label=f'Detail {i+1}') for i in range(3)],
                gr.update(visible=False),gr.update(interactive=False),
                'Upload a short clip, then tap each detail in its frame.',None,None,gr.update(visible=False),None,
                s['blur'],*s['radii'],s['arrival2'],s['arrival3'],s['pull'],s['resolve']]
    def pick(which):
        def handler(value,evt:gr.SelectData):
            value=selected(value,which,evt.index);n=sum(p is not None for p in value['points'])
            return value,marked(value,which),gr.update(interactive=n==3),f'{n}/3 details selected. '+('Ready to direct.' if n==3 else 'Choose the next detail.'),None,gr.update(visible=False),None
        return handler
    def direct(value,blur,r1,r2,r3,arrival2,arrival3,pull,resolve,progress=gr.Progress()):
        requested=settings(blur,r1,r2,r3,arrival2,arrival3,pull,resolve)
        value=render(value,requested,lambda i,n:progress((i,n),desc=f'Area focus · {i}/{n} frames'))
        return value,value['source'],value['result'],gr.update(visible=True),'Area focus rendered locally. Compare your source and result.',None
    def saved(value):return export_recipe(value),'Area recipe saved: source, settings and result.'
    def changed(value,blur,r1,r2,r3,arrival2,arrival3,pull,resolve):
        if not value:return None,None,gr.update(visible=False),None,'Upload a clip first.'
        value=copy.deepcopy(value);value['settings']=settings(blur,r1,r2,r3,arrival2,arrival3,pull,resolve);value['result']=None;save_state(value)
        return value,None,gr.update(visible=False),None,'Settings changed. Direct these details to make a new result.'
    # State mutations and renders share a queue, so a completed render cannot
    # overwrite a newer selection/clear operation from a different callback.
    queue=dict(api_visibility='private',concurrency_id='area-focus-ui',concurrency_limit=1)
    upload.upload(load_clip,upload,load_outputs,**queue)
    upload.clear(clear_clip,outputs=load_outputs,**queue)
    for i,im in enumerate(images):
        im.select(pick(i),state,[state,im,button,status,after,results,download],**queue)
    button.click(direct,[state,*controls],[state,before,after,results,status,download],**queue)
    save.click(saved,state,[download,status],**queue)
    reopen.click(load_recipe,recipe,load_outputs,**queue)
    for control in controls:
        control.input(changed,[state,*controls],[state,after,results,download,status],**queue)
    return state
