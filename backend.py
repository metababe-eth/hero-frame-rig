"""Per-session backend selection. Credentials never enter recipes or workflow JSON."""
import atexit
import contextvars
import functools
import inspect
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from contextlib import nullcontext
from types import SimpleNamespace
from urllib.parse import urlsplit

import gradio as gr
import httpx
from comfy_sdk import Comfy

LOCAL, CLOUD = 'Local ComfyUI', 'Comfy Cloud'
_active = contextvars.ContextVar('hfr_backend', default=None)
_sessions = {}
_lock = threading.RLock()
_proxies = {}


def defaults():
    hosted = bool(os.getenv('SPACE_ID'))
    return dict(kind=CLOUD if os.getenv('HFR_BACKEND', 'cloud' if hosted else 'local').lower() == 'cloud' else LOCAL,
                url=os.getenv('COMFY_LOCAL_URL', 'http://127.0.0.1:8188'), key='')


def config():
    return dict(_active.get() or defaults())


def _private_host():
    """Fail closed if the private Space is made public or its status cannot be checked."""
    space = os.getenv('SPACE_ID', '')
    if not space or os.getenv('HFR_PRIVATE_HOSTED') != '1':
        return False
    try:
        # An unauthenticated Hub read hides private Spaces. Public copies return 200.
        r = httpx.get('https://huggingface.co/api/spaces/' + space, timeout=8)
        return r.status_code in (401, 404)
    except httpx.HTTPError:
        return False


def cloud_key(cfg):
    key = cfg.get('key', '').strip()
    if not key and (_private_host() or cfg.get('loopback', False)):
        with _lock:
            key = os.getenv('COMFY_API_KEY', '').strip()
    if not key.startswith('comfyui-'):
        raise ValueError('Comfy Cloud needs your own Comfy API key. Add it in Backend settings.')
    return key


def local_url(cfg):
    url = cfg['url'].strip().rstrip('/')
    p = urlsplit(url)
    if p.scheme not in ('http', 'https') or not p.hostname or p.username or p.password or p.query or p.fragment:
        raise ValueError('Use a ComfyUI http(s) URL without credentials, a query or a fragment.')
    if os.getenv('SPACE_ID') and p.hostname in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Run this app on your computer to reach your local ComfyUI.')
    return url


def object_info(cfg=None):
    cfg = cfg or config()
    if cfg['kind'] == CLOUD:
        url, headers = 'https://cloud.comfy.org/api/object_info', {'X-API-Key': cloud_key(cfg)}
    else:
        url, headers = local_url(cfg) + '/object_info', {}
    r = httpx.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict) or 'UNETLoader' not in data:
        raise ValueError('Backend did not return ComfyUI node and model information.')
    return data


def _proxy(url):
    """SDK 0.5 uses v2; a loopback-only official adapter fronts standard ComfyUI."""
    with _lock:
        old = _proxies.get(url)
        if old and old[0].poll() is None:
            return old[1]
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        p = subprocess.Popen([sys.executable, '-m', 'comfy_api_proxy.cli', '--comfyui', url,
                              '--host', '127.0.0.1', '--port', str(port)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        _proxies[url] = p, base
        for _ in range(60):
            if p.poll() is not None:
                raise ValueError('Local SDK adapter could not start. Install comfy-api-proxy from requirements.txt.')
            try:
                httpx.get(base + '/api/v2/jobs', timeout=.5)
                return base
            except httpx.HTTPError:
                time.sleep(.1)
        p.terminate()
        raise ValueError('Local SDK adapter did not become ready.')


@atexit.register
def _close_proxies():
    for p, _ in _proxies.values():
        if p.poll() is None:
            p.terminate()


def client(cfg=None):
    cfg = cfg or config()
    if cfg['kind'] == CLOUD:
        base, key = 'https://cloud.comfy.org', cloud_key(cfg)
    else:
        url = local_url(cfg)
        try:
            httpx.get(url + '/system_stats', timeout=5).raise_for_status()
        except httpx.HTTPError:
            raise ValueError('Local ComfyUI cannot be reached at ' + url) from None
        base, key = _proxy(url), ''
    # The pinned SDK reads its base URL only during construction. Never retain an
    # environment change across calls, and never forward a Cloud key to local.
    with _lock:
        saved = os.environ.get('COMFY_BASE_URL')
        saved_key = os.environ.get('COMFY_API_KEY')
        os.environ['COMFY_BASE_URL'] = base
        if cfg['kind']==LOCAL: os.environ.pop('COMFY_API_KEY',None)
        try:
            result = Comfy(api_key=key, timeout=120)
        finally:
            if saved is None: os.environ.pop('COMFY_BASE_URL', None)
            else: os.environ['COMFY_BASE_URL'] = saved
            if saved_key is not None: os.environ['COMFY_API_KEY'] = saved_key
    result.hfr_backend = cfg['kind']
    return result


def clip_client(state, needs_head=False):
    if config()['kind']==CLOUD and state.get('depth') and (not needs_head or state.get('head_matte')):
        return nullcontext(SimpleNamespace(hfr_backend=CLOUD))
    return client()


def _request_config(request):
    with _lock:
        cfg = dict(_sessions.get(getattr(request, 'session_hash', None), (defaults(), 0))[0])
    if request:
        key = request.headers.get('x-comfy-api-key', '')
        if key: cfg['key'] = key
        cfg['loopback'] = _loopback_request(request)
    return cfg


def _loopback_request(request):
    """A tunnel arriving on loopback is still a public link: require local origin."""
    if not request or os.getenv('SPACE_ID'): return False
    if not request.client or request.client.host not in ('127.0.0.1','::1'): return False
    headers=request.headers
    if any(headers.get(k) for k in ('forwarded','x-forwarded-host','x-forwarded-for')): return False
    host=headers.get('host','')
    if urlsplit('http://'+host).hostname not in ('127.0.0.1','localhost','::1'): return False
    origin=headers.get('origin')
    return not origin or urlsplit(origin).hostname in ('127.0.0.1','localhost','::1')


@contextmanager
def use(cfg):
    token = _active.set(cfg)
    try: yield
    finally: _active.reset(token)


def bind(fn):
    """Inject Gradio's request without adding a user-visible API parameter."""
    sig = inspect.signature(fn)
    request_index = sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in sig.parameters.values())
    if inspect.isgeneratorfunction(fn):
        @functools.wraps(fn)
        def wrapped(*args, request=None, **kwargs):
            if len(args) > request_index and (args[request_index] is None or isinstance(args[request_index], gr.Request)):
                request = args[request_index]
                args = args[:request_index] + args[request_index+1:]
            generator = fn(*args, **kwargs)
            while True:
                with use(_request_config(request)):
                    try: value = next(generator)
                    except StopIteration: return
                yield value
    else:
        @functools.wraps(fn)
        def wrapped(*args, request=None, **kwargs):
            if len(args) > request_index and (args[request_index] is None or isinstance(args[request_index], gr.Request)):
                request = args[request_index]
                args = args[:request_index] + args[request_index+1:]
            with use(_request_config(request)):
                return fn(*args, **kwargs)
    # Gradio injects Request only among positional parameters and reads actual
    # annotations as well as the signature. A keyword-only synthetic parameter
    # silently lost both browser session selection and API BYOK headers.
    wrapped.__annotations__ = {**getattr(fn, '__annotations__', {}), 'request': gr.Request}
    parameters = list(sig.parameters.values())
    parameters.insert(request_index, inspect.Parameter('request', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=gr.Request, default=None))
    wrapped.__signature__ = sig.replace(parameters=parameters)
    return wrapped


def apply(kind, url, key, request: gr.Request):
    cfg = dict(kind=kind, url=url, key=key or '')
    cfg['loopback'] = _loopback_request(request)
    with _lock:
        now = time.monotonic()
        for session, (_, touched) in list(_sessions.items()):
            if now - touched > 86400: del _sessions[session]
        if request and request.session_hash: _sessions[request.session_hash] = (cfg, now)
    import still_engines
    try:
        info = object_info(cfg)
        availability = still_engines.availability(info, kind)
        status = f'Active: {kind} · reachable. ' + ('Focus renders in this app.' if kind == CLOUD else 'Focus renders in your ComfyUI.')
    except Exception as exc:
        availability = {k: {'available': False, 'reason': 'Connect the backend to check its models.'} for k in still_engines.ENGINES}
        # Do not expose a URL containing a key or an HTTP exception's headers.
        status = f'Active: {kind} · cannot connect. ' + (str(exc) if isinstance(exc, ValueError) else 'Check the URL or your Comfy key.')
    selected = next((k for k in still_engines.ENGINES if availability[k]['available']), None)
    return status, selected, *[gr.update(interactive=availability[k]['available']) for k in still_engines.ENGINES], still_engines.describe(availability)
