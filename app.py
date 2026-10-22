"""Hero Frame Rig: hero shot builder (Gradio front end).

Flow: pick a shot (Portrait, Medium hero, Full-body hero, Product) and describe the scene ->
Shoot (render still + metric depth + subject matte) -> the subject is in focus and held
sharp -> click to move focus, change aperture, depth of field and look without
re-rendering -> compare takes side by side -> save the recipe (every input, setting and
ComfyUI graph needed to reproduce the frame).

Generation and depth analysis use the selected backend through comfy-sdk.
Local clip focus runs the HFR node; Cloud clip focus imports the same core here.
Lens physics (depth in metres, disparity, blur size, subject layer, clean background plate,
foreground spill) runs locally in optics.py.
"""

import copy
import base64
import html
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
import queue as thread_queue
import threading
import contextvars
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

import optics
import rig_workflow as rig
from frame_input import prepare_upload
import provenance
import ui_state
import clip_focus
import clip_upload
import clip_flow
import camera_control
import agent_api
import backend
import still_engines
from review_state import load_study, select_take, selected_take

TRUE, CINE = "True to lens (1x)", "Cinematic (stylized, up to 4x)"
MODES = [TRUE, CINE]
ANY_LENS = "Any lens"
SHOT_CHOICES = list(rig.SHOTS) + [ANY_LENS]
F_STOPS = [("f/1.4", 1.4), ("f/2", 2.0), ("f/2.8", 2.8), ("f/4", 4.0), ("f/5.6", 5.6)]
COMPARE = {"Apertures": "f/1.4 · f/2.8 · f/5.6", "Looks": "digital · film · cinestill",
           "Depth of field": "true to lens · cinematic"}
MAX_UPLOAD_SIDE = 1600
EXAMPLES = [  # scene, shot, seed (the Oct 12 hero test frames)
    ["A street musician playing a gleaming tenor saxophone on a rainy French Quarter street "
     "at night, gas lamps and wet cobblestones far behind him.", "Medium hero", 1102],
    ["A second line dancer in a feathered costume twirling a decorated parasol on a festival "
     "street at dusk, a brass band and crowd far behind.", "Full-body hero", 1103],
    ["A cup of cafe au lait and a plate of powdered-sugar beignets on a marble cafe table, "
     "soft morning light, a busy cafe far behind.", "Product", 1104],
    ["A jazz singer in a sequined dress outside a club on Frenchmen Street at night, warm rim "
     "light from the doorway, neon signs and string lights glowing far behind her.",
     "Portrait", 1201],
]
EXAMPLES = [row + [rig.SHOTS[row[1]]["lens"], rig.SHOTS[row[1]]["f_stop"]]
            for row in EXAMPLES]


# ---------------- execution ----------------

def _client():
    return backend.client()


def _save(result, node, directory, name):
    path = os.path.join(directory, name)
    result.get_outputs(node)[0].to_file(path)
    return path


def _run_many(client, workflows, stage):
    """Submit all, then collect: compare takes render in parallel. Logs job IDs and
    latency, never credentials."""
    start = time.monotonic()
    jobs = []
    for wf in workflows:
        job = client.submit(wf)
        jobs.append(job)
        print(f"rig stage={stage} submitted_job={job.id}", flush=True)
    results = []
    for job in jobs:
        try:
            results.append(job.result())
        except Exception as exc:
            error = getattr(exc, "error", None)
            code = getattr(error, "code", None) or getattr(exc, "code", None) or type(exc).__name__
            message = getattr(error, "message", None) or str(exc)
            message = re.sub(r"comfyui-[A-Za-z0-9_-]+|Bearer\s+\S+", "[redacted]", str(message))[:600]
            print(f"rig stage={stage} job={job.id} failed code={code} reason={message}", flush=True)
            raise gr.Error(f"backend job {job.id} did not finish successfully ({code}): {message}. "
                           "Your saved takes are unchanged. No retry was submitted.") from exc
    elapsed = time.monotonic() - start
    for r in results:
        print(f"rig stage={stage} job={r.id} cloud_job_seconds={elapsed:.2f}", flush=True)
    return results


def _run(client, workflow, stage):
    return _run_many(client, [workflow], stage)[0]


# ---------------- frame ----------------

def _fit(arr, shape):
    """Resize a 2-D map to (h, w) if the model returned another size."""
    import cv2
    if arr.shape[:2] == tuple(shape):
        return arr
    return cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _frame_state(directory, still, exr, sky, matte, lens_mm, meta):
    """Depth -> disparity; default focus on the main subject (else the nearest central
    thing); the Cinematic multiplier for this frame."""
    with Image.open(still) as im:
        w, h = im.size
    disp, z_near, d = optics.still_disparity(exr, sky, int(lens_mm),
                                             os.path.join(directory, "disparity.png"))
    d = _fit(optics.load_disparity(disp), (h, w))
    optics.save_disparity(d, disp)
    m = _fit(optics.read_mask(matte), (h, w))
    Image.fromarray(np.round(m * 255).astype(np.uint8)).save(matte)
    labels, ids, label = optics.default_subject(m)
    focus = optics.subject_focus(d, labels, label) if label else optics.auto_focus(d)
    cine = optics.cinematic_strength(int(lens_mm), w, z_near, optics.background_gap(d, focus))
    rgb = np.asarray(Image.open(still).convert("RGB"), dtype=np.float32)
    view = rgb * (0.3 + 0.7 * m[..., None])
    Image.fromarray(np.clip(view, 0, 255).astype(np.uint8)).save(
        os.path.join(directory, "subject_view.png"))
    return {"dir": directory, "still": still, "depth_exr": exr, "sky": sky, "matte": matte,
            "disparity": disp, "z_near": z_near, "lens": int(lens_mm), "width": w,
            "height": h, "focus": focus, "label": int(label), "cine": cine, "takes": [],
            "frame_provenance": provenance.capture(),
            **meta}


def _blur(state, f_stop, mode):
    """(blur B in px, effective multiplier, capped?) for this lens, aperture and mode."""
    b1 = optics.blur_scale(state["lens"], float(f_stop), state["width"], state["z_near"], 1.0)
    b = b1 * (state["cine"] if mode == CINE else 1.0)
    return min(b, optics.MAX_BLUR), min(b, optics.MAX_BLUR) / b1, b > optics.MAX_BLUR


def _maps(state, blur, hold, tag):
    """Maps for one take. With the subject layer on and matte subjects at or in front of
    the focus plane: the subject layer, its flat disparity, its matte and the clean
    background plate (layered develop). Otherwise one disparity map with foreground spill.
    Returns (files, subjects held sharp, subjects in the layer)."""
    d = optics.load_disparity(state["disparity"])
    focus = state["focus"]
    path = lambda name: os.path.join(state["dir"], f"{tag}_{name}.png")
    if hold:
        matte = optics.read_mask(state["matte"])
        labels, ids = optics.subject_components(matte)
        layer_ids = optics.layer_subjects(d, labels, ids, focus, blur)
        if layer_ids:
            rgb = np.asarray(Image.open(state["still"]).convert("RGB"))
            fg, fg_d, alpha, plate, plate_d, held = optics.subject_layers(
                rgb, d, matte, labels, layer_ids, focus, blur)
            files = {"layer": path("subject_layer"), "plate": path("plate"),
                     "subject": path("subject_matte")}
            Image.fromarray(fg).save(files["layer"])
            Image.fromarray(plate).save(files["plate"])
            Image.fromarray(np.round(alpha * 255).astype(np.uint8)).save(files["subject"])
            files["layer_disparity"] = optics.save_disparity(fg_d, path("subject_layer_disparity"))
            files["plate_disparity"] = optics.save_disparity(
                optics.spread_foreground(plate_d, focus, blur), path("plate_disparity"))
            return files, held, len(layer_ids)
    d = optics.spread_foreground(d, focus, blur)
    return {"disparity": optics.save_disparity(d, path("disparity"))}, 0, 0


def _caption(n, state, s):
    mode = ui_state.cinematic_label(s) if s["mode"] == CINE else "True to lens"
    guide = f" · guide {s['guide']:g}" if "guide" in s else ""
    return f"#{n} · f/{s['f_stop']:g} · {mode} · {s['focus_m']:.1f} m · {s['look']}{guide}"


def _develop(client, state, treatments, hold, guide=0):
    """Develop one or more takes of the same frame (no re-render) in parallel."""
    guide = rig.guide_eye.amount(guide)
    still_asset = None
    mask_assets = {}
    graphs, wfs, pending = [], [], []
    for i, (f_stop, stock, mode) in enumerate(treatments):
        n = len(state["takes"]) + i + 1
        tag = f"take{n:02d}"
        blur, strength, capped = _blur(state, f_stop, mode)
        files, held, layered = _maps(state, blur, hold, tag)
        round_masks = {}
        roles = (("subject", "layer_disparity"), ("plate", "plate_disparity")) if layered else (("main", "disparity"),)
        for role, disparity_key in roles:
            prefix = os.path.join(state["dir"], f"{tag}_round_{role}_weight")
            masks = rig.round_lens.prepare_weights(files[disparity_key], prefix, blur, state["focus"])
            for level, mask_path in enumerate(masks, 1):
                files[f"round_{role}_weight_{level}"] = mask_path
            round_masks[role] = [os.path.basename(mask_path) for mask_path in masks]
        guide_masks, guide_spec = None, None
        if guide:
            paths, guide_spec = rig.guide_eye.prepare_masks(
                {role: files[key] for role, key in roles},
                os.path.join(state["dir"], f"{tag}_guide"), blur, state["focus"])
            files.update({f"guide_{role}": path for role, path in paths.items()})
            guide_masks = {role: os.path.basename(path) for role, path in paths.items()}
        names = {k: os.path.basename(v) for k, v in files.items()}
        if layered:
            graph = rig.develop(names["layer"], names["layer_disparity"], blur, state["focus"], stock,
                                layers=(names["plate"], names["plate_disparity"], names["subject"]),
                                round_masks=round_masks, width=state["width"], height=state["height"],
                                guide=guide, guide_masks=guide_masks)
            inputs = {"20": files["layer"], "21": files["layer_disparity"], "22": files["plate"],
                      "23": files["plate_disparity"], "24": files["subject"]}
        else:
            graph = rig.develop("still.png", names["disparity"], blur, state["focus"], stock,
                                round_masks=round_masks, width=state["width"], height=state["height"],
                                guide=guide, guide_masks=guide_masks)
            inputs = {"21": files["disparity"]}
        wf = client.workflows.from_json(copy.deepcopy(graph))  # set_input writes into it
        for node, p in inputs.items():
            wf.set_input(node, "image", client.assets.from_file(p))
        weight_paths = {os.path.basename(p): p for key, p in files.items()
                        if key.startswith(("round_", "guide_"))}
        for node, entry in graph.items():
            if entry["class_type"] == "LoadImage" and entry["inputs"]["image"] in weight_paths:
                path = weight_paths[entry["inputs"]["image"]]
                if path not in mask_assets:
                    mask_assets[path] = client.assets.from_file(path)
                wf.set_input(node, "image", mask_assets[path])
        if not layered:
            still_asset = still_asset or client.assets.from_file(state["still"])
            wf.set_input("20", "image", still_asset)
        settings = {"f_stop": float(f_stop), "mode": mode, "strength": round(strength, 2),
                    "blur_px": round(blur, 2), "blur_capped": capped, "look": stock,
                    "focus_disparity": float(state["focus"]),
                    "focus_m": round(optics.focus_metres(state["focus"], state["z_near"]), 2),
                    "subject_layer_enabled": bool(hold),
                    "subject_label": state["label"], "held": held, "layered": layered,
                    "grain_seed": 7, "lens_graph_version": rig.round_lens.VERSION,
                    "lens_kernel": rig.round_lens.describe(blur, state["width"], state["height"])}
        settings["guide"] = guide
        if guide_spec:
            settings["guide_render"] = guide_spec
        graphs.append(graph)
        wfs.append(wf)
        pending.append((n, tag, files, settings))
    results = _run_many(client, wfs, "develop" if len(wfs) == 1 else "compare")
    takes = list(state["takes"])
    for (n, tag, files, settings), graph, result in zip(pending, graphs, results):
        hero = _save(result, rig.HERO_NODE, state["dir"], f"{tag}_hero.png")
        takes.append({"n": n, "hero": hero, "files": files, "settings": settings,
                      "graph": graph, "job": result.id, "caption": _caption(n, state, settings),
                      "provenance": provenance.capture(), "focus_point": state.get("focus_point")})
    return {**state, "takes": takes, "selected_take": takes[-1]["n"]}


def _status(state):
    t = selected_take(state)
    s = t["settings"]
    what = "subject" if state["label"] and s["held"] else "point"
    mode = (ui_state.cinematic_label(s) if s["mode"] == CINE
            else "True to lens")
    cap = " · blur at the node's 50 px limit" if s["blur_capped"] else ""
    held = (" · subject held sharp" if s["held"] else
            " · subject blurred as a layer" if s.get("layered") else "")
    return (f"**Selected take #{t['n']}** · Focus on the {what} at about "
            f"{s['focus_m']:.1f} m (estimated depth) · "
            f"{state['lens']}mm f/{s['f_stop']:g} full-frame · {mode}{held}{cap}\n\n"
            "The comparison uses the same working source. Detail already blurred in that "
            "source cannot be recovered by moving focus.")


def _outputs(state):
    gallery = [(_take_preview(state, t), t["caption"]) for t in reversed(state["takes"])]
    t = selected_take(state)
    settings = t["settings"]
    choices = [(take["caption"], take["n"]) for take in reversed(state["takes"])]
    a, b = ui_state.default_pair(state)
    pair = ui_state.pair_paths(state, a, b)
    return (t["hero"], gallery, state["disparity"],
            os.path.join(state["dir"], "subject_view.png"), state, ui_state.take_readout(state),
            pair, gr.update(value=t["hero"], interactive=True),
            gr.update(choices=[(f"#{take['n']} · f/{take['settings']['f_stop']:g} · "
                                f"{take['settings']['focus_m']:.1f} m (est.) · {take['settings']['look']}", take['n'])
                               for take in reversed(state["takes"])], value=t["n"]),
            settings["f_stop"], settings["look"], settings["mode"],
            settings.get("subject_layer_enabled", bool(settings.get("layered"))), gr.update(value=None, visible=False),
            state["lens"], state.get("shot") or ANY_LENS,
            gr.update(visible=False), gr.update(visible=True), _frame_summary(state),
            "Frame", _stage_metadata(state, "Frame"),
            gr.update(choices=choices, value=a), gr.update(choices=choices, value=b),
            pair[1], _pair_readout(state, a, b), gr.update(value=None, visible=False), None, None,
            gr.update(visible=False), settings.get("guide", 0))


def _take_preview(state, take):
    preview = Path(__file__).parent / "examples" / "nola-previews" / f"take{take['n']:02d}.jpg"
    return str(preview) if state.get("study", {}).get("id") == "nola" and preview.is_file() else take["hero"]


def _frame_summary(state):
    name = state.get("original_name") or ("Saved NOLA study" if state.get("study") else state.get("shot") or "Working frame")
    return f"**1 · Frame** &nbsp; {html.escape(name)} · {state['width']} × {state['height']} · {len(state['takes'])} takes"


def _stage_metadata(state, view):
    data = {**ui_state.orientation_metadata(state["width"], state["height"]),
            "view": view, "point": ui_state.focus_point(state),
            "point_label": ui_state.focus_point_status(state), "loaded": True}
    return '<span id="rig-stage-data" data-stage="' + html.escape(json.dumps(data), quote=True) + '"></span>'


def _pair_readout(state, a, b):
    return f"A: {ui_state.take_readout(state, a)}\n\nB: {ui_state.take_readout(state, b)}"


def show_stage_view(state, view):
    """Review the retained frame/depth/subject image without selecting or rendering a take."""
    if not state:
        return None, ""
    return ui_state.stage_path(state, view), _stage_metadata(state, view)


def compare_saved(state, a, b):
    """Compare existing takes only; the selected hero and its controls stay unchanged."""
    if not state:
        raise gr.Error("Open a frame first.")
    try:
        paths = ui_state.pair_paths(state, int(a), int(b))
        return (paths, paths[1], _pair_readout(state, int(a), int(b)),
                gr.update(value=None, visible=False), None, None, gr.update(visible=False))
    except (ValueError, TypeError) as exc:
        raise gr.Error("Choose A and B from the saved takes.") from exc


def flip_saved(state, a, b):
    """Swap A and B and show the new B at full frame; no selection or Cloud call."""
    if not state:
        raise gr.Error("Open a frame first.")
    a, b = ui_state.swap_pair(state, int(a), int(b))
    return a, b, *compare_saved(state, a, b), gr.update(value="Full frame")


def export_saved_pair(state, a, b):
    """Local PIL/ffmpeg pair export; uses saved take pixels and metadata only."""
    from pair_export import export_pair
    if not state:
        raise gr.Error("Open a frame first.")
    result = export_pair(state, int(a), int(b))
    return (gr.update(value=[result["png"], result["mp4"], result["manifest"], result["bundle"]], visible=True),
            result["mp4"], result["png"], gr.update(visible=True))


def save_recipe_for_ui(state):
    """Reveal the existing recipe download only after an archive is available."""
    return gr.update(value=save_recipe(state), visible=True)


def show_pair_view(view):
    """Toggle saved full-frame/slider views without changing takes or running a job."""
    return gr.update(visible=view == "Full frame"), gr.update(visible=view == "A/B slider")


# ---------------- handlers ----------------

def open_study():
    """Open the bundled NOLA study and its five saved takes without a backend job."""
    directory = tempfile.mkdtemp(prefix="rig_study_")
    state = load_study(Path(__file__).parent / "examples" / "nola_study.zip", directory)
    # These clicks exist in the executed acceptance record; legacy auto-focus takes
    # deliberately have no invented marker. UI metadata stays outside the recipe.
    points = json.loads((Path(__file__).parent / "examples/nola-previews/focus-points.json").read_text(encoding="utf-8"))
    for take in state["takes"]:
        point = points["takes"].get(str(take["n"]))
        if point and hashlib.sha256(Path(take["hero"]).read_bytes()).hexdigest() == point["hero_sha256"]:
            take["focus_point"] = ui_state.normalized_click(*point["pixel_xy"], state["width"], state["height"])
            take["focus_point_source"] = points["source_record"]
    state["focus_point"] = ui_state.focus_point(state)
    return _outputs(state)


def choose_take(state, take_number):
    """Select a saved take and restore its controls, without rendering or a backend job."""
    if not state or not state.get("takes"):
        raise gr.Error("Shoot, load a frame, or open the saved study first.")
    try:
        state = select_take(state, int(take_number))
        state = {**state, "focus_point": ui_state.focus_point(state)}
    except (ValueError, TypeError, KeyError) as exc:
        raise gr.Error("Choose one of this frame's saved takes.") from exc
    return _outputs(state)


def gallery_select(state, evt: gr.SelectData):
    """The gallery is newest first; selecting an image restores that saved take."""
    if not state or not state.get("takes"):
        raise gr.Error("Open a frame first.")
    index = int(evt.index)
    takes = list(reversed(state["takes"]))
    if index < 0 or index >= len(takes):
        raise gr.Error("Choose one of the saved takes.")
    return choose_take(state, takes[index]["n"])

def shoot(scene, shot, lens, f_stop, stock, mode, hold, seed, guide=0, engine="auto", resolution="2K", expand_prompt=True):
    guide = rig.guide_eye.amount(guide)
    if not scene.strip():
        raise gr.Error("Describe the scene first.")
    if shot in rig.SHOTS:
        lens = rig.SHOTS[shot]["lens"]
        prompt = rig.shot_prompt(scene, shot)
    else:
        prompt = rig.build_prompt(scene, int(lens))
    try:
        info, kind = backend.object_info(), backend.config()['kind']
        selected_engine, _ = still_engines.resolve(engine, info, kind)
        width, height = still_engines.canvas(resolution) if selected_engine in ('flux3', 'moss') else (1344, 768)
        graph, selected_engine = still_engines.graph(prompt, int(seed), width, height, selected_engine,
            info, kind, analysis=True, expand_prompt=expand_prompt)
    except ValueError as exc:
        raise gr.Error(str(exc)) from None
    with _client() as client:
        directory = tempfile.mkdtemp(prefix="rig_frame_")
        result = _run(client, client.workflows.from_json(copy.deepcopy(graph)), "shoot")
        files = [_save(result, node, directory, name) for node, name in (
            (rig.STILL_NODE, "still.png"), (rig.DEPTH_NODE, "depth.exr"),
            (rig.SKY_NODE, "sky.png"), (rig.MATTE_NODE, "subject_matte.png"))]
        meta = {"source": "generated", "scene": scene.strip(), "shot": shot,
                "seed": int(seed), "engine": selected_engine, "prompt": prompt,
                "expanded_prompt": backend.expanded_caption(result.id) if '102' in graph else None,
                "prompt_system_source": still_engines.PUP_SOURCE if '102' in graph else None,
                "resolution": resolution, "width": width, "height": height,
                "frame_graph": graph, "frame_job": result.id}
        state = _frame_state(directory, *files, lens, meta)
        state = _develop(client, state, [(f_stop, stock, mode)], hold, guide)
    return _outputs(state)


def bring_frame(path, lens, f_stop, stock, mode, hold, guide=0):
    """Use an existing image: depth + subject matte, no generation. `lens` = what it was
    shot on (full-frame equivalent); it sets the depth scale and the blur size."""
    guide = rig.guide_eye.amount(guide)
    if not path:
        raise gr.Error("Upload a frame first.")
    with _client() as client:
        directory = tempfile.mkdtemp(prefix="rig_frame_")
        still, original_file, original = prepare_upload(path, directory, MAX_UPLOAD_SIDE)
        graph = rig.scan("still.png")
        wf = client.workflows.from_json(copy.deepcopy(graph))
        wf.set_input("20", "image", client.assets.from_file(still))
        result = _run(client, wf, "scan")
        files = [_save(result, node, directory, name) for node, name in (
            (rig.DEPTH_NODE, "depth.exr"), (rig.SKY_NODE, "sky.png"),
            (rig.MATTE_NODE, "subject_matte.png"))]
        meta = {"source": "uploaded", "original_name": os.path.basename(path),
                "original_file": original_file, "original": original,
                "frame_graph": graph, "frame_job": result.id}
        state = _frame_state(directory, still, *files, lens, meta)
        state = _develop(client, state, [(f_stop, stock, mode)], hold, guide)
    return _outputs(state)


def focus_click(state, f_stop, stock, mode, hold, evt: gr.SelectData, guide=0):
    """Click a subject to focus on it (held sharp with the subject layer on); click anywhere
    else to focus on that point."""
    if not state:
        raise gr.Error("Shoot or load a frame first.")
    x = min(max(int(evt.index[0]), 0), state["width"] - 1)
    y = min(max(int(evt.index[1]), 0), state["height"] - 1)
    d = optics.load_disparity(state["disparity"])
    labels, ids = optics.subject_components(optics.read_mask(state["matte"]))
    label = optics.subject_at(labels, x, y)
    label = label if label in ids else 0
    focus = optics.subject_focus(d, labels, label) if (label and hold) else optics.sample(d, x, y)
    return develop({**state, "focus": focus, "label": label,
                    "focus_point": ui_state.normalized_click(x, y, state["width"], state["height"])},
                   f_stop, stock, mode, hold, guide)


def focus_stage_click(state, f_stop, stock, mode, hold, view, evt: gr.SelectData, guide=0):
    """Only the Frame view is interactive; inspecting saved maps never submits a job."""
    if view != "Frame":
        return tuple(gr.skip() for _ in outs)
    return focus_click(state, f_stop, stock, mode, hold, evt, guide)


def develop(state, f_stop, stock, mode, hold, guide=0):
    if not state:
        raise gr.Error("Shoot or load a frame first.")
    with _client() as client:
        state = _develop(client, state, [(f_stop, stock, mode)], hold, guide)
    return _outputs(state)


def compare(state, what, f_stop, stock, mode, hold, guide=0):
    """Several treatments of the same frame side by side, rendered in parallel."""
    if not state:
        raise gr.Error("Shoot or load a frame first.")
    if what == "Apertures":
        treatments = [(n, stock, mode) for n in (1.4, 2.8, 5.6)]
    elif what == "Looks":
        treatments = [(f_stop, s, mode) for s in rig.STOCKS]
    else:
        treatments = [(f_stop, stock, m) for m in MODES]
    with _client() as client:
        state = _develop(client, state, treatments, hold, guide)
    return _outputs(state)


def save_recipe(state):
    """Zip: still, depth EXR, sky, subject matte, every take (hero + the exact maps and
    ComfyUI graph that made it) and recipe.json. Put the PNGs in ComfyUI/input and queue a
    take's graph to reproduce it."""
    if not state or not state["takes"]:
        raise gr.Error("Shoot or load a frame first.")
    out = os.path.join(state["dir"], f"hero_frame_recipe_{uuid.uuid4().hex[:12]}.zip")
    takes = [{"take": t["n"], "hero": f"take{t['n']:02d}_hero.png", "cloud_job": t["job"],
              "settings": t["settings"], "inputs": {k: os.path.basename(v) for k, v in t["files"].items()},
              "provenance": t.get("provenance", {"producer_revision": state.get("producer_revision"),
                                                "source_recipe_sha256": state.get("source_recipe_sha256")}),
              "graph": f"take{t['n']:02d}_develop_graph.json"} for t in state["takes"]]
    recipe = {
        "app": "Hero Frame Rig", "schema_version": 2,
        "made": datetime.now(timezone.utc).isoformat(),
        "selected_take": selected_take(state)["n"],
        "provenance": {**provenance.capture(), "source_study": state.get("study")},
        "frame": {"source": state["source"], "shot": state.get("shot"),
                  "scene": state.get("scene"), "prompt": state.get("prompt"),
                  "engine": state.get("engine"), "expanded_prompt": state.get("expanded_prompt"),
                  "prompt_system_source": state.get("prompt_system_source"),
                  "seed": state.get("seed"), "lens_mm_full_frame": state["lens"],
                  "size": [state["width"], state["height"]], "cloud_job": state["frame_job"],
                  "provenance": state.get("frame_provenance", {
                      "producer_revision": state.get("producer_revision"),
                      "source_recipe_sha256": state.get("source_recipe_sha256")}),
                  "graph": "frame_graph.json"},
        "original_upload": state.get("original"),
        "optics": {"depth": "Depth Anything 3 Metric (estimated, metres for the chosen lens)",
                   "z_near_m": state["z_near"], "cinematic_multiplier": round(state["cine"], 2),
                   "disparity": "D = z_near / z (near is bright)",
                   "blur": "CoC radius px = B * |D - D_focus|, B = f^2 W / (2 N 36mm z_near)"},
        "models": {"image": state.get("frame_graph", {}).get("1", {}).get("inputs", {}).get("unet_name"),
                   "text_encoder": state.get("frame_graph", {}).get("2", {}).get("inputs", {}).get("clip_name"),
                   "prompt_upsampler": state.get("frame_graph", {}).get("101", {}).get("inputs", {}).get("clip_name"),
                   "vae": state.get("frame_graph", {}).get("3", {}).get("inputs", {}).get("vae_name"),
                   "depth": rig.METRIC_MODEL, "subject_matte": "birefnet.safetensors (MIT)"},
        "takes": takes,
    }
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", "Hero Frame Rig recipe\n\n"
                   "still.png is the RGB working frame used for every take. Uploaded originals are\n"
                   "retained unchanged under source/ when present; recipe.json records conversion.\n"
                   "Legacy saved studies may contain only the working image.\n"
                   "Keep that original even if you later choose another working resolution.\n\n"
                   "Replay a developed take: install ComfyUI and its required Radiance nodes,\n"
                   "copy the root-level PNG inputs into ComfyUI/input, then load/submit that take's\n"
                   "*_develop_graph.json (API format). The saved PNG inputs already contain the\n"
                   "prepared depth and subject layers; generation/depth models are not needed\n"
                   "to replay a develop graph. frame_graph.json is a separate generation/scan\n"
                   "graph and does require the models/nodes named in that graph.\n\n"
                   "Repository validate_recipe.py checks the archive offline and can prepare a\n"
                   "namespaced input bundle. Validation is not proof of desktop replay or identical\n"
                   "pixels across node versions. See DEPENDENCIES.md for known versions and gaps.\n\n"
                   "recipe.json selected_take identifies the chosen hero. integrity.json lists\n"
                   "SHA-256 checksums for all other files, to detect corruption or missing inputs.\n"
                   "Checksums establish package consistency, not authenticity or authorship.\n")
        z.writestr("recipe.json", json.dumps(recipe, indent=2))
        z.writestr("frame_graph.json", json.dumps(state["frame_graph"], indent=2))
        if state.get("original_file"):
            z.write(state["original_file"], state["original"]["archive_path"])
        for path, name in ((state["still"], "still.png"), (state["depth_exr"], "depth.exr"),
                           (state["sky"], "sky.png"), (state["matte"], "subject_matte.png"),
                           (state["disparity"], "disparity.png")):
            z.write(path, name)
        for t in state["takes"]:
            z.write(t["hero"], f"take{t['n']:02d}_hero.png")
            for path in t["files"].values():
                z.write(path, os.path.basename(path))
            z.writestr(f"take{t['n']:02d}_develop_graph.json", json.dumps(t["graph"], indent=2))
        dependencies = Path(__file__).parent / "DEPENDENCIES.md"
        if dependencies.is_file():
            z.write(dependencies, "DEPENDENCIES.md")
        dependency_sources = Path(__file__).parent / "dependency_provenance.json"
        if dependency_sources.is_file():
            z.write(dependency_sources, "dependency_provenance.json")
    with zipfile.ZipFile(out, "a", zipfile.ZIP_DEFLATED) as z:
        integrity = {"algorithm": "sha256", "files": {
            name: hashlib.sha256(z.read(name)).hexdigest() for name in z.namelist()}}
        z.writestr("integrity.json", json.dumps(integrity, indent=2))
    return out


def shot_changed(shot):
    if shot in rig.SHOTS:
        return gr.update(value=rig.SHOTS[shot]["lens"]), gr.update(value=rig.SHOTS[shot]["f_stop"])
    return gr.update(), gr.update()


ANIMATION_SOURCES = ["Working source", "Selected take", "Uploaded frame"]
import motion_length


def animate_frame(state, source_mode, uploaded_frame, motion_prompt, animation_seed, end_frame=None, middle_frame=None,
                  camera='None', pin_end=True, camera_amount=25, length=motion_length.STANDARD):
    """Generate one short LTX study; Long requires a middle guide."""
    from animation_input import prepare_motion_input
    from animation_workflow import build_animation_workflow
    frames=motion_length.frames(length,middle_frame)
    if source_mode not in ANIMATION_SOURCES:
        raise gr.Error("Choose the working source, selected take, or an uploaded frame.")
    if not str(motion_prompt).strip():
        raise gr.Error("Describe the small movement to test.")
    if middle_frame and not end_frame and not (pin_end and camera in ('Push in','Pull out')):
        raise gr.Error("Add an end frame to use a middle-frame guide.")
    if source_mode == "Uploaded frame":
        if not uploaded_frame:
            raise gr.Error("Upload an animation frame first.")
        source = uploaded_frame
    else:
        if not state or not state.get("takes"):
            raise gr.Error("Open a frame first, or choose Uploaded frame.")
        source = state["still"] if source_mode == "Working source" else selected_take(state)["hero"]
    directory = Path(tempfile.mkdtemp(prefix="rig_animation_"))
    try:
        camera_meta={'pinned':False,'crop':None,'explicit_end_frame':bool(end_frame)}
        if pin_end and camera in ('Push in','Pull out') and not end_frame:
            original=state.get('original_file',source) if state and source_mode=='Working source' else source
            matte=state.get('matte') if state and source_mode=='Working source' else None
            cropped,crop_meta=camera_control.crop_guide(original,directory/'camera',camera_amount,matte)
            source,end_frame=(cropped,original) if camera=='Pull out' else (original,cropped)
            camera_meta.update(pinned=True,crop=crop_meta)
        prepared, transform = prepare_motion_input(source, directory)
        graph = build_animation_workflow("animation_input.png", str(motion_prompt).strip(),
                                         int(animation_seed),frames=min(frames,121))
        end_prepared, end_transform = None, None
        middle_prepared, middle_transform = None, None
        if end_frame:
            from endframe_workflow import build_endframe_workflow
            end_prepared, end_transform = prepare_motion_input(end_frame, directory / 'end')
            graph = build_endframe_workflow('animation_input.png', 'end/animation_input.png',
                                            int(animation_seed), str(motion_prompt).strip(),frames=min(frames,121))
        if middle_frame:
            from keyframe_workflow import build
            middle_prepared, middle_transform = prepare_motion_input(middle_frame, directory / 'middle')
            graph = build('animation_input.png', 'middle/animation_input.png', 'end/animation_input.png',
                          str(motion_prompt).strip(), int(animation_seed), frames)
        camera_meta.update(camera_control.apply(graph,camera))
    except (ValueError, OSError) as exc:
        raise gr.Error(str(exc)) from exc
    with _client() as client:
        workflow = client.workflows.from_json(copy.deepcopy(graph))
        workflow.set_input("269", "image", client.assets.from_file(prepared))
        if end_prepared:
            from endframe_workflow import LAST_IMAGE_NODE
            workflow.set_input(LAST_IMAGE_NODE, 'image', client.assets.from_file(end_prepared))
        if middle_prepared:
            from keyframe_workflow import MIDDLE_IMAGE_NODE
            workflow.set_input(MIDDLE_IMAGE_NODE, 'image', client.assets.from_file(middle_prepared))
        result = _run(client, workflow, "animation")
        video = _save(result, "75", str(directory), "animation.mp4")
    (directory / "animation_graph.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
    metadata = {
        "app": "Hero Frame Rig", "kind": "experimental animation", "schema_version": 1,
        "cloud_job": result.id, "source_mode": source_mode, "motion_prompt": str(motion_prompt).strip(),
        "seed_first_pass": int(animation_seed), "seed_second_pass": 42,
        "requested": {"frames": frames, "fps": 25, "width": 704, "height": 1280},
        "requested_seconds": frames/25, "length_choice": length, "actual_media_metadata": "Inspect the returned video; request settings are not a media probe.",
        "input_transform": transform, "end_frame_transform": end_transform, "middle_frame_transform": middle_transform, "provenance": provenance.capture(),
        "camera":camera_meta,
        "limits": "Generative motion can alter identity, geometry and focus. Source blur remains baked in. This clip has not received per-frame focus processing or subject-matte protection.",
        "model": "ltx-2.3-22b-dev-fp8.safetensors", "workflow_source_job": "50e5ea56-949b-49c8-aefa-ec6f247f1c52",
        "selected_still_take": selected_take(state)["n"] if state and source_mode != "Uploaded frame" else None,
    }
    (directory / "animation_recipe.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if state and source_mode != "Uploaded frame":
        shutil.copyfile(save_recipe(state), directory / "still_recipe.zip")
    package = directory / "animation_recipe.zip"
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob('*')):
            if path != package and path.is_file():
                archive.write(path, path.relative_to(directory).as_posix())
    # Reopen before reading entries: Windows ZipInfo.orig_filename can retain
    # backslashes during the write pass although ZIP headers normalize them.
    with zipfile.ZipFile(package, "a", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("integrity.json", json.dumps({"algorithm": "sha256", "files": {
            n: hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()}}, indent=2))
    return video, str(package), (f"Experimental motion study complete · job `{result.id}`. "
                                f"Requested {metadata['requested']['frames']} frames at 25 fps, portrait 704 × 1280. "
                                "Review small details and focus before using it in the demo.")


def clip_outputs(value, message, viewing='Focus-directed'):
    settings=value['settings'];has_result=bool(value.get('focused'))
    if not has_result:viewing='Sharp'
    return (value,value['sharp'],value.get('focused') or value['sharp'],
            gr.update(value=viewing,interactive=has_result),clip_focus.marked(value,0),clip_focus.marked(value,1),
            settings['middle'],settings['landing'],settings['f_stop'],settings['guide'],
            gr.update(value=clip_focus.button_label(value),interactive=clip_focus.focus_ready(value)),clip_focus.plot(value.get('plan')),
            message,gr.update(visible=True),gr.update(value=None,visible=False),settings['f_stop'],settings['guide'],clip_focus.ui_metadata(value),settings.get('edge_cleanup',0))


def open_clip_study():
    """Open the two bundled placeholder clips and their plot without any Cloud client."""
    value=clip_focus.load_study(ROOT/'examples/clip-study')
    return clip_outputs(value,'Saved bar study · no backend job. Placeholder art. Depth is not bundled; reopen a clip recipe to reuse it.')


def open_clip_recipe(path):
    """Reopen a clip recipe locally, retaining its cached estimated depth."""
    try:value=clip_focus.import_recipe(path)
    except (ValueError,OSError,KeyError,zipfile.BadZipFile) as exc:raise gr.Error(str(exc)) from exc
    return clip_outputs(value,f'Clip recipe reopened · no backend job. {clip_focus.job_count(value)} backend job(s) for the selected focus mode.')


def animate_for_ui(state, source_mode, uploaded, prompt, seed, end, middle, lens,camera='None',pin_end=True,camera_amount=25,length=motion_length.STANDARD):
    video,package,message=animate_frame(state,source_mode,uploaded,prompt,seed,end,middle,camera,pin_end,camera_amount,length)
    uses_still=bool(state and source_mode!='Uploaded frame')
    strength=_blur(state,1.4,CINE)[1] if uses_still else 1.0
    value=clip_focus.initialize(video,state['lens'] if uses_still else lens,strength)
    value['provenance']=json.loads((Path(video).parent/'animation_recipe.json').read_text())
    clip_focus.save(value)
    return (*clip_outputs(value,message),package)


def inspect_clip_upload(path,length=motion_length.STANDARD):
    if not path:return gr.update(visible=False,value=0),''
    try:meta=clip_upload.inspect_upload(path,motion_length.frames(length,True))
    except (ValueError,OSError,KeyError) as exc:raise gr.Error(str(exc)) from exc
    return (gr.update(visible=meta['show_trim'],value=0,minimum=0,maximum=max(.04,meta['max_start_seconds'])),
            f"{meta['duration_seconds']:.2f} s · full-frame fit to {meta['canvas_size'][0]} × {meta['canvas_size'][1]} · up to {motion_length.frames(length,True)} frames at 25 fps · audio dropped.")


def use_clip_for_ui(path,start,lens,length=motion_length.STANDARD):
    try:value=clip_upload.prepare(path,start,lens,motion_length.frames(length,True))
    except (ValueError,OSError,KeyError) as exc:raise gr.Error(str(exc)) from exc
    return clip_outputs(value,'Your clip is ready · no backend job. Choose both focus points, then render.')


def camera_options(choice,end):
    eligible=choice in ('Push in','Pull out') and not end
    return gr.update(value=eligible,interactive=eligible),gr.update(visible=eligible)


def length_options(middle,current):
    choices=motion_length.choices(middle)
    return gr.update(choices=choices,value=current if current in choices else motion_length.STANDARD)


def clip_start_click(value,evt:gr.SelectData):
    value=clip_focus.select(value,0,evt.index)
    return value,clip_focus.marked(value,0),'Start point selected · no backend job. Render to apply changed points.',clip_focus.ui_metadata(value),gr.update(value=clip_focus.button_label(value),interactive=clip_focus.focus_ready(value))


def clip_end_click(value,evt:gr.SelectData):
    value=clip_focus.select(value,1,evt.index)
    return value,clip_focus.marked(value,1),'End point selected · no backend job. Render to apply changed points.',clip_focus.ui_metadata(value),gr.update(value=clip_focus.button_label(value),interactive=clip_focus.focus_ready(value))


def compare_clip(value,view):
    """Flip the comparison without changing the clip or submitting work."""
    if not value:return None
    return value.get('focused') if view=='Focus-directed' and value.get('focused') else value['sharp']


def direct_clip(value,aperture,guide,middle,landing,progress=gr.Progress(),edge_cleanup=None):
    """Reuse this clip's depth when present, then render estimated-depth focus once."""
    try:
        value=clip_focus.current(value)
        if edge_cleanup is not None:
            value['settings']['edge_cleanup']=clip_focus.focus_core.bounded(edge_cleanup,0,1,'edge cleanup')
        clip_focus.phases(value,middle,landing)  # Validate before creating any client/job.
        value['settings']['middle']=middle;clip_focus.save(value)
        yield (value,gr.update(interactive=False),gr.skip(),gr.skip(),gr.skip(),gr.skip(),'Preparing clip focus…',gr.update(visible=False))
        with backend.clip_client(value, middle==clip_focus.HEAD) as client:
            if not value.get('depth'):
                progress(0.05,desc=('Estimating depth and finding the subject' if middle==clip_focus.HEAD else 'Estimating clip depth')+' — one backend job…')
                clip_focus.estimate_depth(client,value,include_head=middle==clip_focus.HEAD)
                yield (value,gr.update(value=clip_focus.button_label(value),interactive=False),gr.skip(),gr.skip(),gr.skip(),gr.skip(),'Analysis saved. Preparing the focus render…',gr.skip())
            value=clip_focus.render_focus(client,value,aperture,guide,middle,landing,
                lambda message:progress(0.4,desc=message),edge_cleanup=edge_cleanup)
        message='Focus complete · rendered locally · estimated depth · Digital · no subject hold in video.'
        if value['plan'].get('head_tracking'):
            found=value['plan']['head_tracking']['subject_found']
            message+=f" Head target found in {sum(found)}/{len(found)} frames; nearest used where absent."
        yield (value,gr.update(value=clip_focus.button_label(value),interactive=True),value['focused'],
               gr.update(value='Focus-directed',interactive=True),clip_focus.plot(value['plan']),
               gr.skip(),message,gr.update(visible=False))
    except Exception as exc:
        recovered=clip_focus.current(value) if value else value
        message=str(exc) if isinstance(exc,(ValueError,gr.Error)) else f'Clip focus stopped ({type(exc).__name__}). Saved job records prevent automatic resubmission.'
        yield (recovered,gr.update(value=clip_focus.button_label(recovered),interactive=clip_focus.focus_ready(recovered)),gr.skip(),gr.skip(),gr.skip(),gr.skip(),message,gr.skip())


def save_clip_recipe(value):
    """Export clip, cached depth, current treatment and exact schedule without Cloud work."""
    return gr.update(value=clip_focus.export_recipe(value),visible=True)


def clip_middle_changed(value,middle):
    if not value:return gr.skip(),gr.skip()
    value=clip_focus.current(value);value['settings']['middle']=middle;clip_focus.save(value)
    return value,gr.update(value=clip_focus.button_label(value),interactive=clip_focus.focus_ready(value))


def share_clip(value):
    import clip_share
    try:return gr.update(value=clip_share.export(clip_focus.current(value)),visible=True)
    except (ValueError,OSError,subprocess.SubprocessError) as exc:raise gr.Error(str(exc)) from exc


def direct_clip_for_ui(*args,progress=gr.Progress()):
    for row in direct_clip(*args,progress=progress):
        yield (*row,clip_focus.ui_metadata(row[0]) if row[0] else gr.skip())



def simple_outputs(value,message='Choose a point in each frame.'):
    row=list(clip_outputs(value,message))
    row[8]=clip_flow.blur(value['settings']['f_stop'])
    row[10]=gr.update(value='Direct it',interactive=clip_focus.focus_ready(value))
    row[11]=clip_flow.plot(value.get('plan'))
    row[17]=clip_flow.metadata(value)
    return tuple(row)

def try_our_clip():
    return simple_outputs(clip_flow.fresh_study(ROOT/'examples/arrival/arrival_recipe.zip'))

def upload_for_direct(path):
    if not path:return tuple(gr.skip() for _ in clip_outs)
    try:value=clip_upload.prepare(path,0,24,97)
    except (ValueError,OSError,KeyError) as exc:raise gr.Error(str(exc)) from exc
    value['settings']['guide']=60;clip_focus.save(value)
    return simple_outputs(value)

def open_simple_recipe(path):
    try:value=clip_focus.import_recipe(path)
    except (ValueError,OSError,KeyError,zipfile.BadZipFile) as exc:raise gr.Error(str(exc)) from exc
    return simple_outputs(value,'Saved recipe opened. Your settings and focus points are ready.')

def simple_point(value,which,xy):
    value=clip_focus.select(value,which,xy)
    message='Ready. Press Direct it.' if clip_focus.focus_ready(value) else "Now tap what's sharp in the other frame."
    return value,clip_focus.marked(value,which),message,clip_flow.metadata(value),gr.update(value='Direct it',interactive=clip_focus.focus_ready(value))

def simple_start(value,evt:gr.SelectData):return simple_point(value,0,evt.index)
def simple_end(value,evt:gr.SelectData):return simple_point(value,1,evt.index)

def simple_middle(value,middle):
    if not value:return gr.skip(),gr.skip()
    value=clip_focus.current(value);value['settings']['middle']=middle;clip_focus.save(value)
    return value,gr.update(value='Direct it',interactive=clip_focus.focus_ready(value))

def direct_simple(value,blur,spotlight,middle,landing,edge_cleanup=.5):
    # Forward real stage callbacks as streamed HTML. The Gradio progress box is
    # disabled for this event, so there is one visible status throughout.
    messages=thread_queue.Queue()
    def stage(value,desc=None):messages.put(('stage',desc or 'Preparing focus…'))
    def run():
        try:
            for row in direct_clip(value,clip_flow.aperture(blur),spotlight,middle,landing,stage,edge_cleanup=edge_cleanup):messages.put(('row',row))
        except Exception as exc:messages.put(('error',exc))
        finally:messages.put(('end',None))
    request_context=contextvars.copy_context()
    threading.Thread(target=lambda:request_context.run(run),daemon=True).start()
    while True:
        kind,row=messages.get()
        if kind=='end':break
        if kind=='error':raise row
        if kind=='stage':
            updates=[gr.skip() for _ in range(9)];updates[6]=clip_flow.progress_html(row)
            yield tuple(updates);continue
        row=list(row);update=row[1]
        if isinstance(update,dict):row[1]={**update,'value':'Direct it'}
        if str(row[6]).startswith('Focus complete'):row[6]=clip_flow.progress_html('Your clip is ready.',done=True)
        elif str(row[6]).startswith(('Analysis saved','Preparing clip focus')):row[6]=clip_flow.progress_html('Preparing focus…')
        else:row[6]=html.escape(str(row[6]))
        if isinstance(row[4],str):row[4]=clip_flow.plot(row[0].get('plan'))
        yield (*row,clip_flow.metadata(row[0]) if row[0] else gr.skip())

def animate_simple(*args):
    row=list(animate_for_ui(*args));value=row[0]
    value['settings']['guide']=60;clip_focus.save(value)
    return (*simple_outputs(value,'Your clip is ready to direct.'),row[-1])

def simple_state(value):
    ready=bool(value and value.get('focused'))
    return gr.update(visible=ready),gr.update(value=None,visible=False),clip_flow.cost(value)


# ---------------- UI ----------------

ROOT = Path(__file__).resolve().parent
THEME = gr.themes.Base(
    primary_hue="gray", secondary_hue="gray", neutral_hue="zinc",
    font=[gr.themes.Font(name) for name in ("system-ui", "Arial", "sans-serif")],
    font_mono=[gr.themes.Font(name) for name in ("ui-monospace", "Cascadia Mono", "Consolas", "monospace")],
).set(
    body_background_fill="#0E0E10", body_background_fill_dark="#0E0E10",
    body_text_color="#ECE8E1", body_text_color_dark="#ECE8E1",
    background_fill_primary="#17171A", background_fill_primary_dark="#17171A",
    background_fill_secondary="#17171A", background_fill_secondary_dark="#17171A",
    block_background_fill="#17171A", block_background_fill_dark="#17171A",
    block_border_color="#2A2A2F", block_border_color_dark="#2A2A2F",
    block_label_text_color="#9A958C", block_label_text_color_dark="#9A958C",
    input_background_fill="#17171A", input_background_fill_dark="#17171A",
    input_border_color="#2A2A2F", input_border_color_dark="#2A2A2F",
    button_secondary_background_fill="#17171A", button_secondary_background_fill_dark="#17171A",
    button_secondary_text_color="#ECE8E1", button_secondary_text_color_dark="#ECE8E1",
    button_secondary_border_color="#2A2A2F", button_secondary_border_color_dark="#2A2A2F",
    button_primary_background_fill="#C8361F", button_primary_background_fill_dark="#C8361F",
    button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#FFFFFF",
)
preview_path = ROOT / "examples/nola-previews/take04.jpg"
preview_url = "data:image/jpeg;base64," + base64.b64encode(preview_path.read_bytes()).decode("ascii") if preview_path.is_file() else ""
UI_JS = (ROOT / "ui.js").read_text(encoding="utf-8").replace("__NOLA_PREVIEW__", json.dumps(preview_url)).replace(
    "__DOWNLOAD_HELPER__", (ROOT / "downloads.js").read_text(encoding="utf-8"))
LAUNCH_OPTIONS = {"theme": THEME, "css": (ROOT / "ui.css").read_text(encoding="utf-8"), "js": UI_JS,
                  "allowed_paths": [str(ROOT / "examples/nola-previews")]}

for _name in ('shoot', 'bring_frame', 'develop', 'focus_stage_click', 'compare', 'animate_simple', 'direct_simple', 'simple_state'):
    globals()[_name] = backend.bind(globals()[_name])

with gr.Blocks(title="Hero Frame Rig") as demo:
    state = gr.State(None)
    clip_state = gr.State(None)
    with gr.Column(elem_id="hero-rig", elem_classes=["rig-empty", "rig-portrait"]):
        with gr.Row(elem_id="rig-topbar"):
            gr.HTML('<div class="brand">Hero Frame Rig <span>Hold the moment. Direct the focus.</span></div>')
            study_btn = gr.Button("Explore saved study", variant="secondary", size="sm", elem_id="study-button", scale=0,interactive=(ROOT/'examples/nola_study.zip').is_file())

        with gr.Accordion("Backend · local first, Cloud optional", open=True, elem_id="backend-settings"):
            with gr.Row():
                backend_kind = gr.Radio([backend.LOCAL, backend.CLOUD], value=backend.defaults()['kind'], label="Run jobs on")
                backend_url = gr.Textbox(value=backend.defaults()['url'], label="Local ComfyUI URL")
                backend_key = gr.Textbox(type="password", label="Your Comfy Cloud key", placeholder="Session only · never saved in recipes")
            backend_apply = gr.Button("Connect / refresh models")
            backend_status = gr.Markdown("Checking the active backend…")

        with gr.Tabs(selected="direct",elem_id="workflow-tabs") as workflow_tabs:
            with gr.Tab("Direct a clip",id="direct",elem_id="motion-tab") as motion_tab:
                gr.Markdown('**1 · Your clip**',elem_id='clip-source-heading')
                with gr.Row(elem_id='clip-source-fields'):
                    phone_clip=gr.File(label='Drop a clip (MP4 or MOV)',type='filepath',file_types=['.mp4','.mov'],height=125,min_width=0,elem_id='phone-clip-upload',elem_classes=['rig-input-file'],scale=3)
                    clip_study_button=gr.Button('No clip? Try ours',elem_id='clip-study-button',scale=1,min_width=140,interactive=(ROOT/'examples/arrival/arrival_recipe.zip').is_file())
                change_clip_button=gr.Button('Change clip',elem_id='change-clip-button',size='sm')
                with gr.Column(visible=False,elem_id='clip-workbench') as clip_workbench:
                    with gr.Column(elem_id='clip-layout'):
                        gr.Markdown("**2 · Tap what's sharp**",elem_id='clip-aim-heading')
                        with gr.Row(elem_id='clip-endpoints'):
                            clip_start=gr.Image(label="Start: tap what's sharp",type='pil',interactive=False,format='png',buttons=[],min_width=0,elem_id='clip-start')
                            clip_end=gr.Image(label="End: tap what's sharp",type='pil',interactive=False,format='png',buttons=[],min_width=0,elem_id='clip-end')
                        with gr.Column(min_width=0,elem_id='clip-controls'):
                            gr.Markdown('**3 · Direct it**')
                            clip_direct_button=gr.Button('Direct it',interactive=False,variant='primary',elem_id='clip-direct-button')
                            clip_cost=gr.Markdown(elem_id='clip-cost')
                            clip_status=gr.HTML(elem_id='clip-status')
                    with gr.Accordion('More controls',open=False,elem_id='clip-more-controls'):
                        clip_aperture=gr.Slider(0,100,value=100,step=1,label='Blur',elem_id='clip-aperture')
                        gr.HTML('<div class="slider-ends"><span>Subtle</span><span>Strong</span></div>')
                        clip_middle=gr.Radio(clip_flow.MIDDLE,value=clip_focus.HEAD,label='In between',elem_id='clip-middle')
                        gr.Markdown('Face follow aims at the top of the subject. It can miss small or hidden faces.',elem_id='clip-follow-help')
                        clip_landing=gr.Slider(50,95,value=85,step=1,label='When focus arrives',elem_id='clip-landing')
                        gr.HTML('<div class="slider-ends"><span>Early</span><span>Late</span></div>')
                        clip_guide=gr.Slider(0,100,value=60,step=1,label='Spotlight',info="Darkens what's out of focus",elem_id='clip-guide')
                        clip_edge_cleanup=gr.Slider(0,1,value=.5,step=.05,label='Edge cleanup',info='Feathers thin edges toward nearby solid depth. 0 keeps the previous render.',elem_id='clip-edge-cleanup')
                        with gr.Column(min_width=0,elem_id='clip-timeline'):
                            gr.Markdown('**Focus over time**')
                            clip_plot=gr.HTML(clip_focus.plot(None),elem_id='clip-plot')
                        with gr.Accordion('Open a saved recipe',open=False):
                            clip_recipe_upload=gr.File(label='Saved recipe ZIP',type='filepath',file_types=['.zip'],elem_id='clip-recipe-upload',elem_classes=['rig-input-file'])
                            clip_import_button=gr.Button('Open recipe',elem_id='clip-import-button')
                    with gr.Column(visible=False,elem_id='clip-result') as clip_result:
                        gr.Markdown('**Your directed clip**')
                        with gr.Column(min_width=0,elem_id='clip-viewer'):
                            clip_toggle=gr.Radio([('Before','Sharp'),('After','Focus-directed')],value='Sharp',show_label=False,interactive=False,elem_id='clip-toggle')
                            with gr.Row(elem_id='clip-videos'):
                                animation_video=gr.Video(show_label=False,interactive=False,include_audio=False,loop=True,elem_id='clip-sharp',min_width=0,buttons=[])
                                clip_compare=gr.Video(show_label=False,interactive=False,include_audio=False,loop=True,elem_id='clip-comparison',min_width=0,buttons=[])
                            with gr.Row(elem_id='clip-result-actions',elem_classes=['rig-download-source']):
                                clip_share_button=gr.Button('Share',variant='primary',elem_id='clip-share-button')
                                gr.HTML('<div class="rig-download-buttons"><button type="button" data-rig-video-download="clip-comparison" disabled>Download</button></div>',elem_id='clip-download-actions')
                        with gr.Accordion('More downloads',open=False,elem_id='clip-more-downloads'):
                            gr.HTML('<div class="rig-download-source rig-download-buttons"><button type="button" data-rig-video-download="clip-sharp" disabled>Download original clip</button></div>')
                            clip_save_button=gr.Button('Save recipe (.zip)',elem_id='clip-save-button')
                            clip_recipe_download=gr.File(label='Saved recipe',interactive=False,visible=False,elem_id='clip-recipe-download',elem_classes=['rig-download-source'])
                            clip_share_download=gr.File(label='Share files',file_count='multiple',interactive=False,visible=False,elem_id='clip-share-download',elem_classes=['rig-download-source'])
                    clip_meta=gr.HTML(elem_id='clip-metadata')
                with gr.Accordion('Open a saved recipe',open=False,elem_id='initial-open-recipe'):
                    initial_recipe=gr.File(label='Saved recipe ZIP',type='filepath',file_types=['.zip'],elem_classes=['rig-input-file'])
                    initial_recipe_open=gr.Button('Open recipe')
            with gr.Tab("Make a clip",id="make",elem_id="make-tab"):
                with gr.Column(elem_id='motion-generation') as motion_generation:
                    animation_source = gr.Radio(ANIMATION_SOURCES,value='Working source',label='Animation input')
                    with gr.Row(elem_id='motion-frame-row'):
                        animation_upload = gr.File(label='Start frame (choose Uploaded frame above)',type='filepath',file_types=['image'],height=112,min_width=0,elem_id='motion-start-frame',elem_classes=['rig-input-file'])
                        animation_middle = gr.File(label='Middle frame (optional)',type='filepath',file_types=['image'],height=112,min_width=0,elem_classes=['rig-input-file'])
                        animation_end = gr.File(label='End frame (optional)',type='filepath',file_types=['image'],height=112,min_width=0,elem_classes=['rig-input-file'])
                    gr.Markdown("**2 · Create motion**",elem_id="motion-create-heading")
                    animation_length=gr.Radio(motion_length.choices(),value=motion_length.STANDARD,label='Length',elem_id='motion-length')
                    motion_prompt = gr.Textbox(label='Small movement to test',lines=2,value='A restrained continuous portrait shot. Very slight breathing and subtle movement in the surroundings. Keep the subject appearance and framing consistent. No dialogue.')
                    camera_choice=gr.Radio([c for c in camera_control.CHOICES if '(untested)' not in c],value='None',label='Camera',elem_id='camera-choice')
                    camera_pin=gr.Checkbox(value=False,interactive=False,label='Pin the end of the move',elem_id='camera-pin')
                    camera_amount=gr.Slider(10,35,value=25,step=1,label='Amount (%)',visible=False,elem_id='camera-amount')
                    with gr.Row():
                        animation_seed = gr.Number(value=20261013,precision=0,label='Animation seed')
                        animate_button = gr.Button('Animate one study — 1 backend job',elem_id='animate-button')
                    gr.Markdown('Add a middle frame for a longer clip. Motion can change identity and details.')
                    animation_recipe = gr.File(label='Animation recipe',interactive=False,elem_id='animation-recipe',elem_classes=['rig-download-source'])
            with gr.Tab("Stills",id="stills",elem_id="stills-tab") as stills_tab:
                with gr.Accordion("1 · Frame", open=True, elem_id="frame-builder") as frame_builder:
                    with gr.Row():
                        shot = gr.Dropdown(SHOT_CHOICES, value="Medium hero", label="Shot preset")
                        lens = gr.Radio([24, 35, 50, 85], value=50, label="Lens (mm, full-frame equivalent)")
                    with gr.Tab("Generate"):
                        engine = gr.Dropdown(['auto', *still_engines.ENGINES], value='auto', visible=False)
                        engine_selected = gr.Markdown("**Still engine:** checking availability")
                        with gr.Row(elem_id="engine-picker"):
                            engine_buttons = [gr.Button(v['label'], interactive=False) for v in still_engines.ENGINES.values()]
                        engine_status = gr.Markdown("Connect a backend to check available models.")
                        with gr.Row():
                            still_resolution = gr.Radio(['2K', '1K'], value='2K', label='FLUX 3 size')
                            expand_prompt = gr.Checkbox(value=True, label='Expand prompt for FLUX 3', info='Keeps your style and depth instructions. Turn off to use your own full caption.')
                        scene = gr.Textbox(label="Scene", lines=2, placeholder="A street musician under the French Quarter gas lamps…")
                        with gr.Row():
                            seed = gr.Number(value=1102, precision=0, label="Seed", scale=1)
                            shoot_btn = gr.Button("Shoot", variant="primary", elem_id="shoot-button", scale=1)
                    with gr.Tab("Use my frame"):
                        upload = gr.File(label="Original image", type="filepath", file_types=["image"],elem_classes=['rig-input-file'])
                        gr.Markdown("Choose an estimated lens above. Your original file is kept in the recipe; processing uses an oriented RGB copy up to 1600 px.")
                        upload_preview = gr.Image(label="Upload preview", interactive=False, height=160,buttons=[])
                        bring_btn = gr.Button("Load frame", elem_id="bring-button")
                    examples_panel = gr.Accordion("Scene starting points", open=False)

                with gr.Row(visible=False, elem_id="frame-summary-row") as frame_summary_row:
                    frame_summary = gr.Markdown()
                    change_frame_btn = gr.Button("Change frame", size="sm", scale=0, elem_id="change-frame-button")

                with gr.Row(elem_id="direct-heading"):
                    stage_view = gr.Radio(["Frame", "Depth", "Subject"], value="Frame", show_label=False,
                                          container=False, elem_id="stage-view", scale=0)
                    gr.Markdown("**2 · Direct**")
                with gr.Row(elem_id="direct-layout"):
                    with gr.Column(scale=0, min_width=0, elem_id="stage-column"):
                        hero = gr.Image(label="Frame (click to focus)", show_label=False, interactive=False,
                                        format="png", container=False, elem_id="stage-image", buttons=[])
                        with gr.Column(elem_id="stage-empty-overlay", min_width=0):
                            stage_study_btn = gr.Button("Open the saved study — no backend job", variant="secondary", elem_id="stage-study-button")
                            stage_motion_study_btn = gr.Button("or see it move: open the saved clip study",variant="secondary",elem_id="stage-motion-study-button",interactive=(ROOT/"examples/arrival/arrival_recipe.zip").is_file())
                            gr.Markdown("or shoot a frame, or bring your own, above")
                        status = gr.Markdown("Bring a frame or open the saved study.", elem_id="frame-readout")
                        stage_meta = gr.HTML("", elem_id="stage-metadata", visible=True)
                    with gr.Column(scale=1, min_width=260, elem_id="direct-controls"):
                        f_stop = gr.Radio(F_STOPS, value=1.4, label="Aperture", elem_id="aperture-control")
                        mode = gr.Radio(MODES, value=CINE, label="Depth of field", elem_id="dof-control")
                        guide = gr.Slider(0, 100, value=0, step=1, label="Guide the eye (stylized)",
                                          elem_id="guide-control")
                        hold = gr.Checkbox(value=True, label="Subject layer (subject matte)",
                                           info="Holds the focused subject sharp. Off: estimated-depth blur only.", elem_id="subject-control")
                        stock = gr.Radio(list(rig.STOCKS), value="film", label="Look", elem_id="look-control")
                        develop_btn = gr.Button("Re-develop", elem_id="redevelop-button")
                        gr.Markdown("Click the Frame to refocus. Focus and looks re-develop the same source. Detail already blurred in the source cannot be recovered.", elem_id="direct-help")
                        with gr.Accordion("Render several treatments", open=False):
                            what = gr.Dropdown(list(COMPARE), value="Apertures", label="Treatments")
                            compare_btn = gr.Button("Render comparison takes")
                            gr.Markdown("Each treatment runs a backend job.")

                gr.Markdown("**3 · Compare & keep**", elem_id="keep-heading")
                takes = gr.Gallery(label="Saved takes", show_label=False, columns=5, rows=1,
                                   height=154, object_fit="contain", elem_id="take-gallery", allow_preview=False,buttons=[])
                with gr.Row(elem_id="keep-actions"):
                    take_picker = gr.Dropdown(choices=[], value=None, label="Selected take", elem_id="selected-take", scale=3)
                    hero_download = gr.DownloadButton("Download PNG", value=None, interactive=False,elem_id='hero-download',elem_classes=['rig-download-source'])
                    recipe_btn = gr.Button("Save recipe (.zip)")
                recipe_file = gr.File(label="Recipe", interactive=False, visible=False, elem_id="recipe-file",elem_classes=['rig-download-source'])
                with gr.Row(elem_id="pair-layout"):
                    with gr.Column(scale=0, min_width=0, elem_id="pair-stage-column"):
                        pair_full = gr.Image(label="Viewing B (selected take unchanged)", show_label=False, interactive=False,
                                             format="png", elem_id="pair-full", buttons=[], container=False)
                        source_compare = gr.ImageSlider(label="A (left) / B (right)", show_label=False, interactive=False,
                                                        elem_id="pair-slider", visible=False, container=False,buttons=['fullscreen'])
                    with gr.Column(min_width=0, elem_id="pair-controls"):
                        pair_a = gr.Dropdown(choices=[], value=None, label="A", elem_id="pair-a")
                        pair_b = gr.Dropdown(choices=[], value=None, label="B", elem_id="pair-b")
                        pair_readout = gr.Markdown("Choose two saved takes. Comparing and exporting run no backend jobs.", elem_id="pair-readout")
                        flip_btn = gr.Button("Flip A ↔ B", elem_id="flip-button")
                        pair_export_btn = gr.Button("Export pair", elem_id="export-pair-button")
                        pair_view = gr.Radio(["Full frame", "A/B slider"], value="Full frame", label="Compare view", elem_id="pair-view")
                pair_files = gr.File(label="Pair exports and SHA-256 manifest", file_count="multiple", interactive=False,
                                     visible=False, elem_id="pair-files",elem_classes=['rig-download-source'])
                with gr.Accordion("Pair export previews", open=False, visible=False, elem_id="pair-export-previews") as pair_previews:
                    pair_png = gr.Image(label="Labeled pair", interactive=False,buttons=['fullscreen'])
                    pair_video = gr.Video(label="A → B → A · 4 seconds", loop=True, autoplay=False,buttons=[])
                gr.Markdown("Saved-take selection, view switches, Flip, downloads and pair exports run locally. Shoot, Load frame, Re-develop and clicking Frame run on the selected backend.", elem_id="cost-note")


        depth = gr.Textbox(visible=False)
        subject = gr.Textbox(visible=False)

    with examples_panel:
        gr.Examples(EXAMPLES, [scene, shot, seed, lens, f_stop], label="Hero shots")

    outs = [hero, takes, depth, subject, state, status, source_compare, hero_download,
            take_picker, f_stop, stock, mode, hold, recipe_file, lens, shot,
            frame_builder, frame_summary_row, frame_summary, stage_view, stage_meta,
            pair_a, pair_b, pair_full, pair_readout, pair_files, pair_video, pair_png, pair_previews, guide]
    pair_outs = [source_compare, pair_full, pair_readout, pair_files, pair_video, pair_png, pair_previews]
    queue = {"concurrency_id": "rig", "concurrency_limit": 1, "show_progress": "minimal", "api_visibility": "private"}
    hero_download.change(None,hero_download,None,
        js="(file) => { const button=document.getElementById('hero-download'); if (button) button.dataset.rigDownloadUrl=file?.url || ''; return []; }",
        queue=False,api_visibility="private")
    shot.input(shot_changed, shot, [lens, f_stop], api_visibility="private")
    lens.input(lambda: ANY_LENS, None, shot, api_visibility="private")
    upload.change(lambda path: path, upload, upload_preview, queue=False, api_visibility="private")
    shoot_btn.click(shoot, [scene, shot, lens, f_stop, stock, mode, hold, seed, guide, engine, still_resolution, expand_prompt], outs, **queue)
    bring_btn.click(bring_frame, [upload, lens, f_stop, stock, mode, hold, guide], outs, **queue)
    develop_btn.click(develop, [state, f_stop, stock, mode, hold, guide], outs, **queue)
    hero.select(focus_stage_click, [state, f_stop, stock, mode, hold, stage_view, guide], outs, **queue)
    compare_btn.click(compare, [state, what, f_stop, stock, mode, hold, guide], outs, **queue)
    recipe_btn.click(save_recipe_for_ui, state, recipe_file, **queue)
    study_btn.click(open_study, None, outs, js="() => { window.heroRig?.showWorkflow('Stills'); window.heroRig?.startStudy(); return []; }", **queue)
    stage_study_btn.click(open_study, None, outs, js="() => { window.heroRig?.showWorkflow('Stills'); window.heroRig?.startStudy(); return []; }", **queue)
    change_frame_btn.click(lambda: (gr.update(visible=True, open=True), gr.update(visible=False)),
                           None, [frame_builder, frame_summary_row], queue=False, api_visibility="private")
    take_picker.input(choose_take, [state, take_picker], outs, **queue)
    takes.select(gallery_select, state, outs, **queue)
    stage_view.input(show_stage_view, [state, stage_view], [hero, stage_meta], **queue)
    pair_a.input(compare_saved, [state, pair_a, pair_b], pair_outs, **queue)
    pair_b.input(compare_saved, [state, pair_a, pair_b], pair_outs, **queue)
    flip_btn.click(flip_saved, [state, pair_a, pair_b], [pair_a, pair_b, *pair_outs, pair_view], **queue)
    pair_view.change(show_pair_view, pair_view, [pair_full, source_compare], queue=False,api_visibility="private")
    pair_export_btn.click(export_saved_pair, [state, pair_a, pair_b], [pair_files, pair_video, pair_png, pair_previews], **queue)
    clip_outs=[clip_state,animation_video,clip_compare,clip_toggle,clip_start,clip_end,clip_middle,clip_landing,
               clip_aperture,clip_guide,clip_direct_button,clip_plot,clip_status,clip_workbench,clip_recipe_download,f_stop,guide,clip_meta,clip_edge_cleanup]
    clip_study_button.click(try_our_clip,None,clip_outs,**queue)
    stage_motion_study_btn.click(try_our_clip,None,clip_outs,js="() => { window.heroRig?.showWorkflow('Direct a clip'); return []; }",**queue)
    clip_import_button.click(open_simple_recipe,clip_recipe_upload,clip_outs,**queue)
    initial_recipe_open.click(open_simple_recipe,initial_recipe,clip_outs,**queue)
    phone_clip.upload(upload_for_direct,phone_clip,clip_outs,**queue)
    change_clip_button.click(None,None,None,js="() => { window.heroRig?.changeClip(); }",queue=False,api_visibility="private")
    animation_middle.change(length_options,[animation_middle,animation_length],animation_length,queue=False,api_visibility="private")
    animation_upload.change(lambda path:gr.update(value='Uploaded frame') if path else gr.skip(),animation_upload,animation_source,queue=False,api_visibility="private")
    camera_choice.change(camera_options,[camera_choice,animation_end],[camera_pin,camera_amount],queue=False,api_visibility="private")
    animation_end.change(camera_options,[camera_choice,animation_end],[camera_pin,camera_amount],queue=False,api_visibility="private")
    animate_button.click(animate_simple,[state,animation_source,animation_upload,motion_prompt,animation_seed,animation_end,animation_middle,lens,camera_choice,camera_pin,camera_amount,animation_length],
                         clip_outs+[animation_recipe],**queue).then(None,None,None,js="() => { window.heroRig?.showWorkflow('Direct a clip'); }",queue=False,api_visibility="private")
    clip_start.select(simple_start,clip_state,[clip_state,clip_start,clip_status,clip_meta,clip_direct_button],**queue)
    clip_end.select(simple_end,clip_state,[clip_state,clip_end,clip_status,clip_meta,clip_direct_button],**queue)
    clip_toggle.change(None,clip_toggle,None,js="(view) => { window.heroRig?.flipClip(view); return []; }",queue=False,api_visibility="private")
    clip_direct_button.click(direct_simple,[clip_state,clip_aperture,clip_guide,clip_middle,clip_landing,clip_edge_cleanup],
        [clip_state,clip_direct_button,clip_compare,clip_toggle,clip_plot,animation_video,clip_status,clip_recipe_download,clip_meta],**{**queue,'show_progress':'hidden'})
    clip_save_button.click(save_clip_recipe,clip_state,clip_recipe_download,**queue)
    clip_middle.input(simple_middle,[clip_state,clip_middle],[clip_state,clip_direct_button],**queue)
    clip_state.change(simple_state,clip_state,[clip_result,clip_share_download,clip_cost],queue=False,api_visibility="private")
    clip_share_button.click(share_clip,clip_state,clip_share_download,**queue).then(
        None,clip_share_download,None,js="(files) => { window.heroRig?.saveShare(files); return []; }",queue=False,api_visibility="private")

    backend_outputs = [backend_status, engine, *engine_buttons, engine_status]
    backend_apply.click(backend.apply, [backend_kind, backend_url, backend_key], backend_outputs, api_visibility='private')
    backend_kind.change(backend.apply, [backend_kind, backend_url, backend_key], backend_outputs, api_visibility='private')
    demo.load(backend.apply, [backend_kind, backend_url, backend_key], backend_outputs, api_visibility='private')
    for key, button in zip(still_engines.ENGINES, engine_buttons):
        button.click(lambda key=key: key, None, engine, api_visibility='private')
    engine.change(lambda key: '**Still engine:** ' + still_engines.ENGINES.get(key, {}).get('label', 'none available'),
                  engine, engine_selected, api_visibility='private')

    agent_api.register(_client)

if __name__ == "__main__":
    demo.launch(**LAUNCH_OPTIONS)
