"""Small open-weight API graphs, selected against the active loader inventory."""
import math
from functools import lru_cache
import httpx
import rig_workflow as rig
import engine_templates

ENGINES = engine_templates.snapshot()

PUP_FILE = ENGINES['flux3']['pup']
PUP_SOURCE = 'https://huggingface.co/black-forest-labs/flux-3-image-klein-pup/resolve/main/system_prompts.json'
RIG_RULE = ('Hero Frame Rig constraint: keep deep focus from foreground to background. '
            'Preserve the stated visual style exactly. Keep the stated near object and '
            'far subject at their specified distances and planes; never move the distant '
            'subject into the foreground. No text, logos or brands.')


@lru_cache(maxsize=1)
def prompt_system():
    """Fetch BFL's Apache-2.0 caption instructions at runtime, not from our repo."""
    r = httpx.get(PUP_SOURCE, follow_redirects=True, timeout=20)
    r.raise_for_status()
    text = r.json()['t2i']
    if not isinstance(text, str) or not text.strip():
        raise ValueError('The upstream caption instructions are unavailable.')
    return text.strip() + '\n\n' + RIG_RULE


def canvas(resolution='2K', aspect=1280 / 704):
    side = 2048 if resolution == '2K' else 1024
    return (round(side * math.sqrt(aspect) / 16) * 16,
            round(side / math.sqrt(aspect) / 16) * 16)


def choices(info, node, field, cloud=True):
    data = info.get(node, {}).get('input', {}).get('required', {}).get(field, [[]])
    values = list(data[0]) if isinstance(data[0], list) else list(data[1].get('options', [])) if len(data)>1 else []
    tag = {('UNETLoader','unet_name'): 'diffusion_models',
           ('CLIPLoader','clip_name'): 'text_encoders', ('VAELoader','vae_name'): 'vae'}.get((node, field))
    for asset in info.get('_hfr_cloud_catalogue', []) if cloud else []:
        if tag and tag in asset.get('tags', []) and asset.get('path'):
            values.append(asset['path'])
    return list(dict.fromkeys(values))


def availability(info, kind):
    result = {}
    specs = info.get('_hfr_engine_specs', ENGINES)
    for key in ENGINES:
        if key not in specs:
            result[key] = dict(available=False, reason=info.get('_hfr_template_errors', {}).get(key, 'Official template unavailable; reconnect to refresh.'), files={})
            continue
        spec = specs[key]
        files, missing = {}, []
        for role, node, field in [('unet', 'UNETLoader', 'unet_name'), ('clip', 'CLIPLoader', 'clip_name'), ('vae', 'VAELoader', 'vae_name')]:
            installed = choices(info, node, field, cloud=kind == 'Comfy Cloud')
            files[role] = next((v for preferred in spec[role] for v in installed if v.replace('\\', '/').split('/')[-1] == preferred), None)
            if files[role] is None: missing.append(spec[role][0])
        if spec['type'] not in choices(info, 'CLIPLoader', 'type', cloud=False):
            missing.append('CLIPLoader type ' + spec['type'])
        required = ['CLIPTextEncode', 'VAEDecode', 'EmptyFlux2LatentImage', 'SaveImage']
        required += ['ConditioningZeroOut', 'CFGGuider', 'RandomNoise', 'KSamplerSelect', 'Flux2Scheduler', 'SamplerCustomAdvanced'] if key == 'klein4b' else ['KSampler']
        missing += [name for name in required if name not in info]
        reason = ''
        if missing:
            reason = ('Not found in this Cloud inventory; execution availability is unconfirmed.' if kind == 'Comfy Cloud'
                      else 'Install the required models or update ComfyUI.') + ' Missing: ' + ', '.join(missing)
        result[key] = dict(available=not missing, reason=reason, files=files)
    return result


def describe(status):
    return '\n\n'.join('**' + spec['label'] + '** — ' + ('Ready' if status[k]['available'] else status[k]['reason']) for k, spec in ENGINES.items())


def resolve(engine, info, kind):
    status = availability(info, kind)
    if engine in (None, '', 'auto'):
        engine = next((k for k in ENGINES if status[k]['available']), 'flux3')
    if engine not in ENGINES:
        raise ValueError('engine must be auto, flux3, moss or klein4b.')
    if not status[engine]['available']:
        raise ValueError(ENGINES[engine]['label'] + ': ' + status[engine]['reason'])
    return engine, status[engine]['files']


def graph(prompt, seed, width, height, engine, info, kind, analysis=False, expand_prompt=True):
    engine, files = resolve(engine, info, kind)
    spec = info.get('_hfr_engine_specs', ENGINES)[engine]
    if engine == 'klein4b':
        nodes = {k: v for k, v in rig._shoot_nodes(prompt, 50, seed, width, height, prompt=prompt).items() if int(k) <= 13}
    else:
        # Preserve the accepted Brief25 FLUX 3 sampling behavior. This pass
        # refreshes its filenames; only MOSS takes its sampler from the template.
        sampling = spec['sampling'] if engine == 'moss' else dict(steps=16, cfg=1.0, sampler_name='euler', scheduler='simple', denoise=1.0)
        # Official template's ordinary sampling path, with MOSS caption expansion
        # and SLG switches off. Keep IDs stable for the shared analysis/recipe path.
        nodes = {
            '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': files['unet'], 'weight_dtype': 'default'}},
            '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': files['clip'], 'type': spec['type'], 'device': 'default'}},
            '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': files['vae']}},
            '4': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 0], 'text': prompt}},
            '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 0], 'text': 'nsfw, worst quality, bad anatomy, watermark, jpeg artifact' if engine == 'moss' else ''}},
            '10': {'class_type': 'EmptyFlux2LatentImage', 'inputs': {'width': width, 'height': height, 'batch_size': 1}},
            '11': {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'positive': ['4', 0], 'negative': ['5', 0],
                    'latent_image': ['10', 0], 'seed': seed, **sampling}},
            '12': {'class_type': 'VAEDecode', 'inputs': {'samples': ['11', 0], 'vae': ['3', 0]}},
            '13': {'class_type': 'SaveImage', 'inputs': {'images': ['12', 0], 'filename_prefix': 'rig/still'}},
        }
    if engine == 'flux3' and expand_prompt:
        pup = next((p for p in choices(info, 'CLIPLoader', 'clip_name', cloud=kind == 'Comfy Cloud')
                    if p.replace('\\', '/').split('/')[-1] == spec.get('pup')), None)
        if not pup or 'TextGenerate' not in info or 'PreviewAny' not in info:
            raise ValueError('FLUX 3 caption model is unavailable. Turn off Expand prompt to use your caption verbatim.')
        nodes.update({
            '101': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': pup, 'type': 'flux3', 'device': 'default'}},
            '102': {'class_type': 'TextGenerate', 'inputs': {
                'clip': ['101', 0], 'prompt': prompt, 'system_prompt': prompt_system(),
                'max_length': 3584, 'sampling_mode': 'on', 'thinking': False,
                'use_default_template': True, 'mtp': 'off', 'sampling_mode.temperature': 1,
                'sampling_mode.top_k': 1, 'sampling_mode.top_p': .95, 'sampling_mode.min_p': 0,
                'sampling_mode.repetition_penalty': 1.05, 'sampling_mode.seed': 0,
                'sampling_mode.presence_penalty': 1.5}},
            '103': {'class_type': 'PreviewAny', 'inputs': {'source': ['102', 0]}},
        })
        nodes['4']['inputs']['text'] = ['103', 0]
    for node, role, field in [('1', 'unet', 'unet_name'), ('2', 'clip', 'clip_name'), ('3', 'vae', 'vae_name')]:
        nodes[node]['inputs'][field] = files[role]
    if analysis:
        nodes.update(rig._depth_nodes(['12', 0])); nodes.update(rig._matte_nodes(['12', 0]))
    return nodes, engine
