"""Discover model filenames from official templates, with an offline local snapshot.

Cloud refreshes upstream once per hour. Failed refreshes fail closed for that
engine. Local operation never needs a network request: its server inventory is
matched against the small, dated snapshot shipped with this app.
"""
import copy
import json
import time
from functools import lru_cache
from pathlib import Path
import httpx

TEMPLATES = {
    'flux3': ('FLUX 3 Image Klein', 'image_flux3_klein_distilled_t2i'),
    'moss': ('MOSS-Image', 'image_moss_image_t2i'),
    'klein4b': ('FLUX.2 Klein 4B', 'image_flux2_klein_text_to_image'),
}
BASE = 'https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/'
SNAPSHOT = Path(__file__).with_name('engine_templates.json')


def extract(key, document):
    """Read loader defaults and sampler settings; never execute upstream code."""
    nodes = document['nodes'] + [n for s in document.get('definitions', {}).get('subgraphs', []) for n in s['nodes']]
    loaders = {role: [n for n in nodes if n['type'] == cls]
               for role, cls in [('unet', 'UNETLoader'), ('clip', 'CLIPLoader'), ('vae', 'VAELoader')]}
    pups = [n for n in loaders['clip'] if 'prompt_upsampler' in str(n.get('widgets_values', [''])[0])]
    loaders['clip'] = [n for n in loaders['clip'] if n not in pups]
    spec = dict(label=TEMPLATES[key][0], template=TEMPLATES[key][1], source=BASE + TEMPLATES[key][1] + '.json')
    for role, matches in loaders.items():
        names = list(dict.fromkeys(n['widgets_values'][0] for n in matches))
        if not names or any(not isinstance(n, str) or not n.endswith('.safetensors') for n in names):
            raise ValueError('Official template has no recognized ' + role + ' loader.')
        spec[role] = names
    if key == 'klein4b':
        # This gallery document includes an alternate base-model branch. The
        # rig's existing four-step sampler uses only its distilled branch.
        spec['unet'] = [name for name in spec['unet'] if '-base-' not in name]
        if not spec['unet']:
            raise ValueError('The official template has no distilled 4B model.')
    types = list(dict.fromkeys(n['widgets_values'][1] for n in loaders['clip']))
    if len(types) != 1:
        raise ValueError('Official template has ambiguous CLIP types.')
    spec['type'] = types[0]
    if pups:
        spec['pup'] = pups[0]['widgets_values'][0]
    samplers = [n for n in nodes if n['type'] == 'KSampler']
    if samplers:
        values = samplers[0]['widgets_values']
        spec['sampling'] = dict(steps=values[2], cfg=values[3], sampler_name=values[4], scheduler=values[5], denoise=values[6])
    return spec


def snapshot():
    return json.loads(SNAPSHOT.read_text(encoding='utf-8'))['engines']


@lru_cache(maxsize=12)
def _fetch(key, hour):
    response = httpx.get(BASE + TEMPLATES[key][1] + '.json', follow_redirects=True, timeout=20)
    response.raise_for_status()
    return extract(key, response.json())


def discover(cloud=False):
    if not cloud:
        return snapshot(), {}
    specs, errors = {}, {}
    for key in TEMPLATES:
        try:
            specs[key] = copy.deepcopy(_fetch(key, int(time.time() // 3600)))
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
            errors[key] = 'Could not read the current official template. Reconnect to refresh engine discovery.'
    return specs, errors
