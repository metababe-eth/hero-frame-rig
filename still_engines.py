"""Small open-weight API graphs, selected against the active loader inventory."""
import os
import rig_workflow as rig

ENGINES = {
    'flux3': {'label': 'FLUX 3 Image Klein', 'type': 'flux3',
              'unet': ['flux3-image-klein-int8-convrot.safetensors', 'flux3-image-klein-bf16.safetensors'],
              'clip': ['qwen3vl_4b_int8_convrot.safetensors'], 'vae': ['flux2-vae.safetensors']},
    'moss': {'label': 'MOSS-Image', 'type': 'moss_image',
             'unet': ['moss_image_sft_int8_convrot.safetensors', 'moss_image_sft_bf16.safetensors'],
             'clip': ['qwen3.5_9b_w4a8_gptq.safetensors'], 'vae': ['flux2-vae_small.safetensors']},
    'klein4b': {'label': 'FLUX.2 Klein 4B', 'type': 'flux2',
                'unet': ['flux-2-klein-4b.safetensors'], 'clip': ['qwen_3_4b.safetensors'],
                'vae': ['flux2-vae.safetensors']},
}


def choices(info, node, field):
    data = info.get(node, {}).get('input', {}).get('required', {}).get(field, [[]])
    if isinstance(data[0],list): return data[0]
    return data[1].get('options',[]) if len(data)>1 else []


def availability(info, kind):
    result = {}
    for key, spec in ENGINES.items():
        files, missing = {}, []
        for role, node, field in [('unet', 'UNETLoader', 'unet_name'), ('clip', 'CLIPLoader', 'clip_name'), ('vae', 'VAELoader', 'vae_name')]:
            installed = choices(info, node, field)
            files[role] = next((v for preferred in spec[role] for v in installed if v.replace('\\', '/').split('/')[-1] == preferred), None)
            if files[role] is None: missing.append(spec[role][0])
        if spec['type'] not in choices(info, 'CLIPLoader', 'type'):
            missing.append('CLIPLoader type ' + spec['type'])
        required = ['CLIPTextEncode', 'VAEDecode', 'EmptyFlux2LatentImage', 'SaveImage']
        required += ['ConditioningZeroOut', 'CFGGuider', 'RandomNoise', 'KSamplerSelect', 'Flux2Scheduler', 'SamplerCustomAdvanced'] if key == 'klein4b' else ['KSampler']
        missing += [name for name in required if name not in info]
        reason = ''
        if missing:
            reason = ('Not on Comfy Cloud yet. Works on local ComfyUI with the required models.' if kind == 'Comfy Cloud'
                      else 'Install the required models or update ComfyUI.') + ' Missing: ' + ', '.join(missing)
        result[key] = dict(available=not missing, reason=reason, files=files)
    return result


def describe(status):
    return '\n\n'.join('**' + spec['label'] + '** — ' + ('Ready' if status[k]['available'] else status[k]['reason']) for k, spec in ENGINES.items())


def resolve(engine, info, kind):
    status = availability(info, kind)
    if engine in (None, '', 'auto'):
        engine = 'klein4b' if kind == 'Comfy Cloud' else next((k for k in ENGINES if status[k]['available']), 'flux3')
    if engine not in ENGINES:
        raise ValueError('engine must be auto, flux3, moss or klein4b.')
    if not status[engine]['available']:
        raise ValueError(ENGINES[engine]['label'] + ': ' + status[engine]['reason'])
    return engine, status[engine]['files']


def graph(prompt, seed, width, height, engine, info, kind, analysis=False):
    engine, files = resolve(engine, info, kind)
    if engine == 'klein4b':
        nodes = {k: v for k, v in rig._shoot_nodes(prompt, 50, seed, width, height, prompt=prompt).items() if int(k) <= 13}
    else:
        # Official Comfy-Org templates, flattened. Prompt enhancement and optional
        # MOSS SLG are off: the caller's prompt is used verbatim, as in the API.
        nodes = {
            '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': files['unet'], 'weight_dtype': 'default'}},
            '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': files['clip'], 'type': ENGINES[engine]['type'], 'device': 'default'}},
            '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': files['vae']}},
            '4': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 0], 'text': prompt}},
            '5': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 0], 'text': 'nsfw, worst quality, bad anatomy, watermark, jpeg artifact' if engine == 'moss' else ''}},
            '10': {'class_type': 'EmptyFlux2LatentImage', 'inputs': {'width': width, 'height': height, 'batch_size': 1}},
            '11': {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'positive': ['4', 0], 'negative': ['5', 0],
                    'latent_image': ['10', 0], 'seed': seed, 'steps': 25 if engine == 'moss' else 16,
                    'cfg': 2.5 if engine == 'moss' else 1.0, 'sampler_name': 'er_sde' if engine == 'moss' else 'euler',
                    'scheduler': 'simple', 'denoise': 1.0}},
            '12': {'class_type': 'VAEDecode', 'inputs': {'samples': ['11', 0], 'vae': ['3', 0]}},
            '13': {'class_type': 'SaveImage', 'inputs': {'images': ['12', 0], 'filename_prefix': 'rig/still'}},
        }
    for node, role, field in [('1', 'unet', 'unet_name'), ('2', 'clip', 'clip_name'), ('3', 'vae', 'vae_name')]:
        nodes[node]['inputs'][field] = files[role]
    if analysis:
        nodes.update(rig._depth_nodes(['12', 0])); nodes.update(rig._matte_nodes(['12', 0]))
    return nodes, engine
