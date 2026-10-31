"""Stateless file-in/file-out entry points, sharing the existing UI pipelines."""
import copy
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Literal

import gradio as gr
from PIL import Image, PngImagePlugin
import clip_flow
import clip_focus
import clip_share
import clip_upload
import rig_workflow
import backend
import still_engines


DEFAULT_ARRIVAL = (85 - 50) / 45
MIDDLE = {'face': clip_focus.HEAD, 'closest': clip_focus.FOLLOW, 'stay': clip_focus.HOLD}


def bounded(name, value, low, high, integer=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise gr.Error(f'{name} must be a number between {low} and {high}.') from None
    if isinstance(value, bool) or not math.isfinite(number) or not low <= number <= high:
        raise gr.Error(f'{name} must be a finite number between {low} and {high}.')
    if integer and number != int(number):
        raise gr.Error(f'{name} must be an integer.')
    return int(number) if integer else number


def failure(exc, directory):
    message = re.sub(r'comfyui-[A-Za-z0-9_-]+|Bearer\s+\S+', '[redacted]', str(exc))[:600]
    print(f'rig agent request_failed saved_directory={directory} reason={message}', flush=True)
    return gr.Error(f'Request stopped: {message}. No automatic retry was submitted. '
                    'If a cloud job was submitted, check its saved receipt in the Space logs before retrying.')


class AgentAPI:
    def __init__(self, client_factory):
        # Only the factory is shared. Each call owns fresh files, state and jobs.
        self.client_factory = client_factory

    @backend.bind
    def direct_clip(self, video: str, start_x: float, start_y: float,
                    end_x: float, end_y: float,
                    in_between: Literal['face', 'closest', 'stay'] = 'face',
                    blur: float = 1.0, arrival: float = DEFAULT_ARRIVAL,
                    spotlight: float = 60, edge_cleanup: float = .5) -> tuple[str, str, str]:
        """Direct an uploaded MP4/MOV without regenerating its motion: 1 Cloud job per successful call (depth plus optional subject analysis);
        focus renders locally in Python. Local backend adds one HFR node job.
        No UI session or previous call is needed; every call creates fresh analysis.

        start_x/start_y and end_x/end_y are finite 0–1 frame fractions, measured from
        top-left on the first and last PROCESSED frames; 1 reaches the last pixel.
        in_between: face (default, top-of-subject heuristic; not face recognition),
        closest (nearest central depth), or stay (hold starting depth until the pull).
        blur: 0 subtle (f/5.6) to 1 strong (f/1.4), default 1.
        arrival: 0 early (50% of clip) to 1 late (95%); default 7/9 gives 85%.
        New requests use Optical. For Cinematic: stay, blur=2/3, arrival=4/9,
        spotlight=0. The existing API defaults remain independent of UI presets.
        spotlight: 0–100, default 60; darkens out-of-focus areas.
        edge_cleanup: 0–1, default .5; RGB-guided depth cleanup in Optical.
        0 bypasses cleanup within the selected lens mode; it does not select Legacy.

        Matches Direct a clip upload: first 97 frames at 25 fps (up to 3.88 s),
        aspect preserved, at most 1280 px long side / 901120 pixels, no upscaling,
        24 mm estimated lens, strength 1; audio removed. Longer uploads are trimmed.
        Source blur cannot be undone. Returns, in order: directed MP4, muted
        1080x1920 Share MP4 (Directed above Original, played twice, cropped around
        the focus endpoint), recipe ZIP with both clips, settings, graphs and job IDs.
        Share and ZIP cost no extra jobs. Failed calls never automatically retry.
        """
        coords = [bounded(name, value, 0, 1) for name, value in
                  zip(('start_x', 'start_y', 'end_x', 'end_y'), (start_x, start_y, end_x, end_y))]
        if in_between not in MIDDLE:
            raise gr.Error('in_between must be face, closest, or stay.')
        blur = bounded('blur', blur, 0, 1)
        arrival = bounded('arrival', arrival, 0, 1)
        spotlight = bounded('spotlight', spotlight, 0, 100)
        edge_cleanup = bounded('edge_cleanup', edge_cleanup, 0, 1)
        middle = MIDDLE[in_between]
        landing = 50 + 45 * arrival
        directory = None
        try:
            state = clip_upload.prepare(video, start=0, lens=24, max_frames=97)
            directory = state['dir']
            for which in (0, 1):
                x, y = coords[which * 2:which * 2 + 2]
                state = clip_focus.select(state, which, [round(x * (state['width'] - 1)),
                                                        round(y * (state['height'] - 1))])
            state['settings'].update(middle=middle, guide=spotlight, landing=landing,
                                     f_stop=clip_flow.aperture(blur * 100),edge_cleanup=edge_cleanup)
            state['provenance']['agent_request'] = dict(start_x=coords[0], start_y=coords[1],
                end_x=coords[2], end_y=coords[3], in_between=in_between, blur=blur,
                arrival=arrival, spotlight=spotlight,edge_cleanup=edge_cleanup)
            clip_focus.phases(state, middle, landing)
            clip_focus.save(state)
            with self.client_factory() as client:
                clip_focus.estimate_depth(client, state, include_head=middle == clip_focus.HEAD)
                state = clip_focus.render_focus(client, state, clip_flow.aperture(blur * 100),
                                                 spotlight, middle, landing,edge_cleanup=edge_cleanup)
            share = clip_share.export(state)[1]
            recipe = clip_focus.export_recipe(state)
            print(f'rig agent=direct_clip completed clip={state["id"]} jobs='
                  + json.dumps([job['id'] for job in state['jobs']]), flush=True)
            return state['focused'], share, recipe
        except Exception as exc:
            raise failure(exc, directory) from None

    @backend.bind
    def shoot_still(self, prompt: str, seed: int = 1102,
                    width: int = 2768, height: int = 1520, engine: str = 'auto',
                    expand_prompt: bool = True) -> str:
        """Generate one PNG with an open-weight model using exactly 1 Comfy job.
        engine: auto (first available engine, preferring FLUX 3), flux3, moss,
        klein4b. Explicit unavailable engines fail before any job is submitted.
        FLUX 3 expands the prompt with BFL's open-weight caption model plus deep-focus,
        style and distance rules. Set expand_prompt=False for verbatim conditioning.
        Other engines use the prompt verbatim. No lens words or shot preset are added. No depth,
        matte, focus treatment or animation is run. No previous UI session is needed.
        seed defaults to 1102 and must be an integer in 0–4294967295. Dimensions
        default to 2768x1520 (2K area); each must be 256–4096 and a multiple of 16,
        with at most 4300000 total pixels. Explicit smaller dimensions remain supported.
        Returns a PNG with its job ID, original prompt, graph and caption metadata.
        Invalid inputs fail before submission; failures never automatically retry.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise gr.Error('prompt must contain a scene description.')
        seed = bounded('seed', seed, 0, 4294967295, integer=True)
        width = bounded('width', width, 256, 4096, integer=True)
        height = bounded('height', height, 256, 4096, integer=True)
        if width % 16 or height % 16 or width * height > 4300000:
            raise gr.Error('Dimensions must be multiples of 16 with at most 4300000 total pixels.')
        try:
            graph, engine = still_engines.graph(prompt, seed, width, height, engine,
                backend.object_info(), backend.config()['kind'], expand_prompt=expand_prompt)
        except ValueError as exc:
            raise gr.Error(str(exc)) from None
        directory = Path(tempfile.mkdtemp(prefix='rig_agent_still_'))
        receipt = {'kind': 'shoot_still', 'status': 'reserved', 'prompt': prompt,
                   'seed': seed, 'width': width, 'height': height, 'engine': engine,
                   'expand_prompt': bool(engine == 'flux3' and expand_prompt),
                   'prompt_system_source': still_engines.PUP_SOURCE if engine == 'flux3' and expand_prompt else None}
        def save():
            (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        save()
        (directory / 'graph.json').write_text(json.dumps(graph, indent=2), encoding='utf-8')
        try:
            with self.client_factory() as client:
                job = client.submit(client.workflows.from_json(copy.deepcopy(graph)))
                receipt.update(id=job.id, status='submitted'); save()
                print(f'rig agent=shoot_still submitted_job={job.id} directory={directory}', flush=True)
                result = job.result()
                target = directory / f'still_{job.id}.png'
                result.get_outputs(rig_workflow.STILL_NODE)[0].to_file(str(target))
                receipt['status'] = 'succeeded'
                receipt['expanded_prompt'] = backend.expanded_caption(job.id) if '102' in graph else None
                save()
                # Portable still provenance without changing PNG pixels or API output shape.
                with Image.open(target) as im:
                    metadata = PngImagePlugin.PngInfo()
                    for key, value in im.info.items():
                        if isinstance(value, str): metadata.add_text(key, value)
                    metadata.add_text('hero_frame_rig', json.dumps({'receipt': receipt, 'graph': graph}))
                    annotated = directory / 'annotated.png'
                    im.save(annotated, pnginfo=metadata)
                annotated.replace(target)
            return str(target)
        except Exception as exc:
            receipt['status'] = 'needs_recovery'; save()
            raise failure(exc, directory) from None


def register(client_factory):
    """API-only controls use normal Gradio file validation and authenticated serving."""
    endpoints = AgentAPI(client_factory)
    queue = dict(concurrency_id='rig', concurrency_limit=1)
    with gr.Column(visible=False):
        video = gr.File(type='filepath', file_types=['.mp4', '.mov'], label='video')
        coords = [gr.Number(label=name) for name in ('start_x', 'start_y', 'end_x', 'end_y')]
        middle = gr.Dropdown(['face', 'closest', 'stay'], value='face', label='in_between')
        blur = gr.Number(value=1.0, label='blur')
        arrival = gr.Number(value=DEFAULT_ARRIVAL, label='arrival')
        spotlight = gr.Number(value=60, label='spotlight')
        edge_cleanup = gr.Number(value=.5,label='edge_cleanup')
        result = [gr.File(label=name) for name in ('directed_mp4', 'share_mp4', 'recipe_zip')]
        gr.Button('API direct').click(endpoints.direct_clip,
            [video, *coords, middle, blur, arrival, spotlight, edge_cleanup], result, api_name='direct_clip', **queue)
        prompt = gr.Textbox(label='prompt')
        seed = gr.Number(value=1102, label='seed')
        width = gr.Number(value=2768, label='width')
        height = gr.Number(value=1520, label='height')
        engine = gr.Textbox(value='auto', label='engine')
        expand_prompt = gr.Checkbox(value=True, label='expand_prompt')
        png = gr.File(label='png')
        gr.Button('API shoot').click(endpoints.shoot_still, [prompt, seed, width, height, engine, expand_prompt],
                                    png, api_name='shoot_still', **queue)
