"""Hero Frame Rig: turns camera-rig settings into ComfyUI API-format workflows.

Named like a real camera pipeline:
  shoot()   -> renders the still with FLUX.2 [klein] 4B and estimates METRIC depth
  scan()    -> metric depth for a frame you bring (no generation)
  develop() -> lens blur (focus + aperture) and the sensor look on a saved still
  clip_depth() / rack_focus_video() / develop_video() -> the same lens on animated clips

Changing focus, aperture or look never re-renders the image.

Lens v2 (Oct 12, after the first rack-focus render looked fake):
  * Depth Anything 3 Metric -> metres (for the chosen lens) -> disparity map D = z_near/z,
    built locally by optics.py. Blur on a disparity map is physically scaled (circle of
    confusion is linear in 1/distance); the old rank-equalized depth exaggerated
    separation and made subjects look cut out.
  * Blur runs in LINEAR light with clipped lights expanded (HDR Expand), so out-of-focus
    bright highlights retain energy during blur.
  * Legacy graphs retain the "Octagon" setting for compatibility. Executed desktop replay
    with Registry Radiance 2.3.3 matches the saved Cloud take within 1/255; that package
    uses a Gaussian approximation for every shape. Earlier true-disc claims were wrong.
    See DESKTOP_REPLAY.md. New still graphs use round_lens.py with saved blend weights,
    a soft round Gaussian matched to the legacy kernel variance. round_video.py
    applies that versioned kernel to scheduled video; old video helpers stay compatible.
  * Grain recalibrated: film profiles at 0.35 were coarse colour clumps at 1280 px.

Hero shot builder (Oct 12, late): SHOTS presets frame the prompt subject-first; shoot()
and scan() also save a BiRefNet subject matte; develop(layers=...) blurs the subject as its
own layer over a clean background plate: a held subject stays crisp without a glow, and an
out-of-focus foreground subject melts over a sharp background.

All models are open weights: FLUX.2 [klein] 4B (Apache 2.0), Depth Anything 3 and
BiRefNet (MIT).
"""

import json
import sys

import optics
import round_lens
import guide_eye

# Lens: framing language goes FIRST in the prompt. Klein weights early tokens
# most, and a lens phrase tacked on at the end was ignored in testing.
LENSES = {
    24: ("Wide-angle 24mm lens shot. Very wide field of view with deep, stretched "
         "perspective; the subject is small in the frame with lots of environment around them."),
    35: ("35mm lens shot. Natural documentary field of view; the subject is framed full "
         "body with the surrounding street clearly visible."),
    50: ("50mm lens shot. Normal perspective like the human eye; the subject is framed "
         "from the thighs up and fills about half the frame height."),
    85: ("Tight 85mm telephoto portrait shot. The subject fills most of the frame height, "
         "framed from the knees up and centered. Compressed perspective: the background is "
         "pulled close behind the subject and only a narrow slice of the street is visible."),
}

# Sensor looks (Radiance Film Grain). Film at 0.35 / size 1 read as coarse colour noise
# at 1280 px and its default halation warmed the frame; these were picked on 100% crops.
# halation/gate_weave: 0 means "profile default", so tiny values switch them off.
STOCKS = {
    "digital": {"profile": "ARRI Alexa 35", "intensity": 0.25, "size": 1.0,
                "halation": 0.0, "gate_weave": 0.0},
    "film": {"profile": "Kodak Vision3 500T 5219", "intensity": 0.12, "size": 0.6,
             "halation": 0.01, "gate_weave": 1e-5},
    "cinestill": {"profile": "CineStill 800T", "intensity": 0.2, "size": 0.6,
                  "halation": 0.0, "gate_weave": 1e-5},
}

# sRGB -> linear, with near-clipped lights (linear luma > 0.95) expanded toward 8x so
# that blurring them makes bright bokeh. Midtones are an exact sRGB decode (round trip
# is not an identity over all colours: saved own-frame no-blur checks before the look
# measured mean 0.46-0.51/255 and maxima 21-28/255 on near-white coloured highlights.
# The new round lens alone is identity at zero blur; expansion/finish are unchanged.
EXPAND = {"source_gamma": 2.2, "highlight_recovery": 1.0, "black_point": 0.0,
          "target_stops": 11.0, "highlight_rolloff": 1.0}

# Linear-light shoulder: rolls expanded highlights back under 1.0 (values below 0.85 are
# untouched), then the exact linear -> sRGB encode.
LINEAR_SHOULDER = {"preset": "None (Custom)", "operator": "exposure_only", "exposure": 0,
                   "gamma": 1, "white_point": 1, "contrast": 1, "saturation": 1,
                   "highlight_compression": 0.3, "shadow_lift": 0, "use_gpu": True}

# Display shoulder after grain: Radiance Film Grain does not clamp, so bright pixels go
# above 1.0 and SaveImage clips them. exposure_only + compression only rolls those off
# (0 clipped px, midtone shift 0.12/255).
SHOULDER = {"preset": "None (Custom)", "operator": "exposure_only", "exposure": 0,
            "gamma": 1, "white_point": 1.5, "contrast": 1, "saturation": 1,
            "highlight_compression": 0.6, "shadow_lift": 0, "use_gpu": True}

DEPTH_RESOLUTION = 1008   # DA3 inference size; optics.metric_scale assumes the same
METRIC_MODEL = "depth_anything_3_metric_large.safetensors"


def build_prompt(scene: str, lens_mm: int) -> str:
    return f"{LENSES[lens_mm]} {scene.strip()}"


# Hero shot presets: subject first, and framed so the subject stands well in front of a
# deep background (that separation is what makes focus and blur read on screen; the
# first street test put a small musician mid-street, which is a weak hero frame).
# Each sets the lens (prompt framing + blur physics) and the default aperture.
SHOTS = {
    "Portrait": {"lens": 85, "f_stop": 1.4, "framing": (
        "Tight 85mm portrait. Head and shoulders, the face fills the upper third of the frame "
        "and the eyes are in sharp detail; the background is far behind the subject.")},
    "Medium hero": {"lens": 50, "f_stop": 1.4, "framing": (
        "50mm medium shot from the waist up. The subject stands a few steps from the camera "
        "in the foreground; the scene stretches far behind them.")},
    "Full-body hero": {"lens": 35, "f_stop": 2.0, "framing": (
        "35mm full-body hero shot from a low camera angle. The subject stands in the "
        "foreground, framed head to toe; deep space and distant lights behind them.")},
    "Product": {"lens": 85, "f_stop": 2.0, "framing": (
        "85mm product hero shot. The object sits on a surface close to the camera in the "
        "centre of the frame, in crisp detail; the background is far behind.")},
}


def shot_prompt(scene: str, shot: str) -> str:
    return f"{SHOTS[shot]['framing']} {scene.strip()}"


# ---------------- graph pieces ----------------

def _depth_nodes(image_ref, start=14, prefix="rig/depth"):
    """Metric depth (raw DA3 units, EXR) + sky mask for one image."""
    m, i, d, s, k, ks = (str(start + n) for n in range(6))
    return {
        m: {"class_type": "LoadDA3Model", "_meta": {"title": "Depth model (metric)"},
            "inputs": {"model_name": METRIC_MODEL, "weight_dtype": "default"}},
        i: {"class_type": "DA3Inference", "_meta": {"title": "Estimate depth"},
            "inputs": {"da3_model": [m, 0], "image": image_ref, "resolution": DEPTH_RESOLUTION,
                       "resize_method": "upper_bound_resize", "mode": "mono"}},
        d: {"class_type": "DA3Render", "_meta": {"title": "Metric depth (raw units)"},
            "inputs": {"da3_geometry": [i, 0], "output": "depth",
                       "output.normalization": "raw", "output.apply_sky_clip": False}},
        s: {"class_type": "SaveImageAdvanced", "_meta": {"title": "Save depth (EXR)"},
            "inputs": {"images": [d, 0], "filename_prefix": prefix, "format": "exr",
                       "format.bit_depth": "32-bit float", "format.input_color_space": "linear"}},
        k: {"class_type": "DA3Render", "_meta": {"title": "Sky mask"},
            "inputs": {"da3_geometry": [i, 0], "output": "sky_mask", "output.colored": False}},
        ks: {"class_type": "SaveImage", "_meta": {"title": "Save sky mask"},
             "inputs": {"images": [k, 0], "filename_prefix": prefix + "_sky"}},
    }


def _matte_nodes(image_ref, start=22, prefix="rig/subject"):
    """Subject matte (BiRefNet, MIT, core node): foreground = white. optics uses it to keep
    the whole subject, including thin things it holds (a sax), on one focus plane."""
    l, r, c, s = (str(start + n) for n in range(4))
    return {
        l: {"class_type": "LoadBackgroundRemovalModel", "_meta": {"title": "Subject matte model"},
            "inputs": {"bg_removal_name": "birefnet.safetensors"}},
        r: {"class_type": "RemoveBackground", "_meta": {"title": "Subject matte"},
            "inputs": {"bg_removal_model": [l, 0], "image": image_ref}},
        c: {"class_type": "MaskToImage", "_meta": {"title": "Matte as image"}, "inputs": {"mask": [r, 0]}},
        s: {"class_type": "SaveImage", "_meta": {"title": "Save subject matte"},
            "inputs": {"images": [c, 0], "filename_prefix": prefix}},
    }


def _to_linear(nid, ref):
    return {nid: {"class_type": "HDRExpandDynamicRange", "_meta": {"title": "Linear light + bright lights"},
                  "inputs": {"image": ref, **EXPAND}}}


def _lens(nid, image_ref, disp_ref, blur, focus_d, title=None):
    return {nid: {"class_type": "RadianceDepthOfField",
                  "_meta": {"title": title or f"Lens blur, focus D={round(focus_d, 3)}"},
                  "inputs": {"image": image_ref, "depth_map": disp_ref,
                             **optics.dof_inputs(blur, focus_d), "bokeh_shape": "Octagon",
                             "highlight_boost": 1.0, "foreground_blur": True, "use_gpu": True}}}


def _finish(ids, ref, stock, seed=7, grain_use_gpu=True):
    """Linear shoulder -> sRGB -> grain -> display shoulder. ids = 4 node ids."""
    a, b, c, d = ids
    s = STOCKS[stock]
    return {
        a: {"class_type": "HDRToneMap", "_meta": {"title": "Linear shoulder"},
            "inputs": {"image": ref, **LINEAR_SHOULDER}},
        b: {"class_type": "ColorSpaceConvert", "_meta": {"title": "Linear -> sRGB"},
            "inputs": {"image": [a, 0], "source_space": "Linear_sRGB", "target_space": "sRGB",
                       "exposure": 0, "gamma_adjust": 1, "chromatic_adaptation": "None", "use_gpu": True}},
        c: {"class_type": "RadianceFilmGrain", "_meta": {"title": f"Look: {s['profile']}"},
            "inputs": {"image": [b, 0], "profile": s["profile"], "intensity": s["intensity"],
                       "seed": seed, "size": s["size"], "iso": 0, "halation": s["halation"],
                       "blend_mode": "Additive", "gate_weave": s["gate_weave"], "use_gpu": grain_use_gpu}},
        d: {"class_type": "HDRToneMap", "_meta": {"title": "Highlight shoulder"},
            "inputs": {"image": [c, 0], **SHOULDER}},
    }


# ---------------- stills ----------------

def _shoot_nodes(scene, lens_mm, seed, width, height, prompt=None):
    wf = {
        "1": {"class_type": "UNETLoader", "_meta": {"title": "Klein 4B (Apache 2.0)"},
              "inputs": {"unet_name": "flux-2-klein-4b.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "_meta": {"title": "Text encoder"},
              "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "_meta": {"title": "Flux 2 VAE"},
              "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "_meta": {"title": "Lens words + scene"},
              "inputs": {"clip": ["2", 0], "text": prompt or build_prompt(scene, lens_mm)}},
        "5": {"class_type": "ConditioningZeroOut", "_meta": {"title": "Empty negative"},
              "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "CFGGuider", "_meta": {"title": "Guider"},
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "cfg": 1}},
        "7": {"class_type": "RandomNoise", "_meta": {"title": "Seed"},
              "inputs": {"noise_seed": seed}},
        "8": {"class_type": "KSamplerSelect", "_meta": {"title": "Sampler"},
              "inputs": {"sampler_name": "euler"}},
        "9": {"class_type": "Flux2Scheduler", "_meta": {"title": "Scheduler (4 steps)"},
              "inputs": {"steps": 4, "width": width, "height": height}},
        "10": {"class_type": "EmptyFlux2LatentImage", "_meta": {"title": f"Frame {width}x{height}"},
               "inputs": {"width": width, "height": height, "batch_size": 1}},
        "11": {"class_type": "SamplerCustomAdvanced", "_meta": {"title": "Render still"},
               "inputs": {"noise": ["7", 0], "guider": ["6", 0], "sampler": ["8", 0],
                          "sigmas": ["9", 0], "latent_image": ["10", 0]}},
        "12": {"class_type": "VAEDecode", "_meta": {"title": "Decode"},
               "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "_meta": {"title": "Save still"},
               "inputs": {"images": ["12", 0], "filename_prefix": "rig/still"}},
    }
    wf.update(_depth_nodes(["12", 0]))     # 14-19: 17 = depth EXR, 19 = sky mask
    wf.update(_matte_nodes(["12", 0]))     # 22-25: 25 = subject matte
    return wf


STILL_NODE, DEPTH_NODE, SKY_NODE, MATTE_NODE, HERO_NODE = "13", "17", "19", "25", "32"


def shoot(scene, lens_mm=50, seed=20261012, width=1344, height=768, shot=None):
    """Generate the still (13), its metric depth (17, EXR) + sky mask (19) and the subject
    matte (25). `shot` (a SHOTS key) overrides the lens framing with a hero preset.
    Then build the disparity map locally: optics.still_disparity(exr, sky, lens_mm, out)."""
    if shot:
        return _shoot_nodes(scene, SHOTS[shot]["lens"], seed, width, height, shot_prompt(scene, shot))
    return _shoot_nodes(scene, lens_mm, seed, width, height)


def scan(image_name):
    """Bring your own frame: metric depth (17) + sky mask (19) + subject matte (25), no
    generation."""
    wf = {"20": {"class_type": "LoadImage", "_meta": {"title": "Your frame"},
                 "inputs": {"image": image_name}}}
    wf.update(_depth_nodes(["20", 0]))
    wf.update(_matte_nodes(["20", 0]))
    return wf


def develop(still_name, disparity_name, blur, focus_d, stock="film", seed=7, layers=None,
            round_masks=None, width=None, height=None, guide=0, guide_masks=None):
    """Re-develop a still with a disparity map (from optics) without re-rendering.

    blur = optics.blur_scale(lens_mm, f_stop, width, z_near, strength); focus_d = the
    disparity at the focus point. For a blurred foreground to spill naturally, pass a map
    run through optics.spread_foreground(d, focus_d, blur). Output node 32.

    layers = (plate_name, plate_disparity_name, alpha_name), from optics.subject_layers:
    then still_name / disparity_name are the SUBJECT LAYER (colours pushed past its edge)
    and its flat disparity. Three lens passes: the subject layer, its matte (blurred by the
    same map, so an out-of-focus subject melts over what is behind it), and the clean
    background plate; the layer is composited over the plate in linear light. The subject's
    colours never enter the background bokeh (no glow), and a held subject stays crisp.
    """
    guide = guide_eye.amount(guide)
    if guide and (round_masks is None or not guide_masks):
        raise ValueError("Guide requires a round-lens graph and saved guide masks.")
    wf = {
        "20": {"class_type": "LoadImage", "_meta": {"title": "Subject layer" if layers else "Still"},
               "inputs": {"image": still_name}},
        "21": {"class_type": "LoadImage",
               "_meta": {"title": "Subject layer disparity" if layers else "Disparity"},
               "inputs": {"image": disparity_name}},
    }
    wf.update(_to_linear("30", ["20", 0]))
    def lens_branch(prefix, image_ref, disparity_ref, mask_group):
        if round_masks is None:
            wf.update(_lens(prefix, image_ref, disparity_ref, blur, focus_d))
            return [prefix, 0]
        branch, output = round_lens.graph(image_ref, width, height, blur,
                                          round_masks[mask_group], id_prefix=f"round_{prefix}")
        wf.update(branch)
        return output

    subject_out = lens_branch("31", ["30", 0], ["21", 0], "subject" if layers else "main")
    def guide_branch(image_ref, role):
        if not guide:
            return image_ref
        nodes, output = guide_eye.graph(image_ref, guide_masks[role], guide, role)
        wf.update(nodes)
        return output

    subject_out = guide_branch(subject_out, "subject" if layers else "main")
    lens_out = subject_out
    if layers:
        plate_name, plate_disp, alpha_name = layers
        wf["22"] = {"class_type": "LoadImage", "_meta": {"title": "Background plate (subject removed)"},
                    "inputs": {"image": plate_name}}
        wf["23"] = {"class_type": "LoadImage", "_meta": {"title": "Plate disparity"},
                    "inputs": {"image": plate_disp}}
        wf["24"] = {"class_type": "LoadImage", "_meta": {"title": "Subject matte"},
                    "inputs": {"image": alpha_name}}
        matte_out = lens_branch("26", ["24", 0], ["21", 0], "subject")
        wf["25"] = {"class_type": "ImageToMask", "_meta": {"title": "Matte as mask"},
                    "inputs": {"image": matte_out, "channel": "red"}}
        wf.update(_to_linear("37", ["22", 0]))
        plate_out = lens_branch("38", ["37", 0], ["23", 0], "plate")
        plate_out = guide_branch(plate_out, "plate")
        wf["39"] = {"class_type": "ImageCompositeMasked", "_meta": {"title": "Subject layer over background"},
                    "inputs": {"destination": plate_out, "source": subject_out, "x": 0, "y": 0,
                               "resize_source": False, "mask": ["25", 0]}}
        lens_out = ["39", 0]
    # Keep the still look stage on CPU for portable recipe replay. The exported graph
    # records this choice; prior takes keep their original GPU setting and pixels.
    wf.update(_finish(("33", "34", "35", "36"), lens_out, stock, seed, grain_use_gpu=False))
    wf["32"] = {"class_type": "SaveImage", "_meta": {"title": "Save hero frame"},
                "inputs": {"images": ["36", 0], "filename_prefix": "rig/hero"}}
    if round_masks is not None:
        wf["32"]["_meta"]["title"] = "Save hero frame / " + round_lens.VERSION
    if guide:
        wf["32"]["_meta"]["title"] += " / " + guide_eye.VERSION
    return wf


# ---------------- clips ----------------
# Animation order (task 2, Oct 12): LTX-2.3 strips grain and lets baked blur drift, so
# animate the SHARP still, then apply focus, look and shoulder to every frame.

def clip_depth(video_name, fps=25, depth_scale=4):
    """Per-frame metric depth for a clip, packed into ONE 10-bit animated AVIF (raw / depth_scale,
    video range) + a sky-mask AVIF, so the planner downloads two files, not 2 x N EXRs.
    Decode with rack_focus.read_clip_depth (checked against EXR: 0.4% median error).
    New clips use scale4 for finer indoor precision; saved graphs and recipes retain
    their recorded scale. Reject clipping and pass that same scale to
    rack_focus.read_clip_depth. Outputs: node 8 = depth, node 10 = sky."""
    import math
    if isinstance(depth_scale,bool) or not isinstance(depth_scale,(int,float)) or not math.isfinite(depth_scale) or depth_scale<=0:
        raise ValueError('Depth packing scale must be positive and finite.')
    anim = {"format": "avif", "format.input_color_space": "sRGB", "format.save_mode": "animated",
            "format.save_mode.fps": fps, "format.save_mode.loop_count": 0}
    return {
        "1": {"class_type": "LoadVideo", "_meta": {"title": "Animated clip (sharp)"},
              "inputs": {"file": video_name}},
        "2": {"class_type": "GetVideoComponents", "_meta": {"title": "Frames"},
              "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "LoadDA3Model", "_meta": {"title": "Depth model (metric)"},
              "inputs": {"model_name": METRIC_MODEL, "weight_dtype": "default"}},
        "4": {"class_type": "DA3Inference", "_meta": {"title": "Depth per frame"},
              "inputs": {"da3_model": ["3", 0], "image": ["2", 0], "resolution": DEPTH_RESOLUTION,
                         "resize_method": "upper_bound_resize", "mode": "mono"}},
        "5": {"class_type": "DA3Render", "_meta": {"title": "Metric depth (raw units)"},
              "inputs": {"da3_geometry": ["4", 0], "output": "depth",
                         "output.normalization": "raw", "output.apply_sky_clip": False}},
        "6": {"class_type": "GPUTensorOps", "_meta": {"title": f"Scale 1/{depth_scale:g}"},
              "inputs": {"image": ["5", 0], "operation": "Exposure", "value": -math.log2(depth_scale), "force_gpu": True}},
        "7": {"class_type": "GPUTensorOps", "_meta": {"title": "Clamp 0-1"},
              "inputs": {"image": ["6", 0], "operation": "Clamp", "value": 0, "force_gpu": True}},
        "8": {"class_type": "SaveImageAdvanced", "_meta": {"title": "Save depth (10-bit)"},
              "inputs": {"images": ["7", 0], "filename_prefix": "rig/clip_depth",
                         "format.bit_depth": "10-bit YUV420", "format.crf": 1, **anim}},
        "9": {"class_type": "DA3Render", "_meta": {"title": "Sky mask"},
              "inputs": {"da3_geometry": ["4", 0], "output": "sky_mask", "output.colored": False}},
        "10": {"class_type": "SaveImageAdvanced", "_meta": {"title": "Save sky mask"},
               "inputs": {"images": ["9", 0], "filename_prefix": "rig/clip_sky",
                          "format.bit_depth": "8-bit YUV420", "format.crf": 10, **anim}},
    }


def rack_focus_video(video_name, disparity_video_name, segments, blur, stock="film",
                     fps=25, crf=12):
    """Legacy Radiance Gaussian lens for video; still round-mask graphs are not applied.

    Focus per frame range on an animated clip (rack focus, follow focus, or one fixed
    focus = develop_video). `segments` and the disparity video come from rack_focus.plan.
    Linear light once for the whole clip, one lens node per segment, rejoin (<= 50), then
    shoulder, sRGB, look and shoulder once. Output: node 41 = hero clip."""
    if not 1 <= len(segments) <= 50:
        raise ValueError("Batch Images joins 1-50 segments; merge segments with a larger step.")
    wf = {
        "1": {"class_type": "LoadVideo", "_meta": {"title": "Animated clip (sharp)"},
              "inputs": {"file": video_name}},
        "2": {"class_type": "GetVideoComponents", "_meta": {"title": "Frames"},
              "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "LoadVideo", "_meta": {"title": "Disparity per frame"},
              "inputs": {"file": disparity_video_name}},
        "4": {"class_type": "GetVideoComponents", "_meta": {"title": "Disparity frames"},
              "inputs": {"video": ["3", 0]}},
    }
    wf.update(_to_linear("5", ["2", 0]))
    batch = {}
    for i, seg in enumerate(segments):
        img, dep, dof = f"{100 + i}", f"{200 + i}", f"{300 + i}"
        wf[img] = {"class_type": "ImageFromBatch", "_meta": {"title": f"Frames {seg['start']}+{seg['length']}"},
                   "inputs": {"image": ["5", 0], "batch_index": seg["start"], "length": seg["length"]}}
        wf[dep] = {"class_type": "ImageFromBatch", "_meta": {"title": f"Disparity {seg['start']}+{seg['length']}"},
                   "inputs": {"image": ["4", 0], "batch_index": seg["start"], "length": seg["length"]}}
        wf.update(_lens(dof, [img, 0], [dep, 0], blur, seg["focus"],
                        title=f"Focus D={seg['focus']} (frames {seg['start']}+{seg['length']})"))
        batch[f"images.image{i}"] = [dof, 0]
    wf["30"] = {"class_type": "BatchImagesNode", "_meta": {"title": "Rejoin segments"}, "inputs": batch}
    wf.update(_finish(("31", "32", "33", "34"), ["30", 0], stock))
    save = {"format": "mp4", "format.codec": "h264", "format.codec.encoding": "re-encode",
            "format.codec.encoding.crf": crf}
    wf.update({
        "40": {"class_type": "CreateVideo", "_meta": {"title": "Hero clip"},
               "inputs": {"images": ["34", 0], "fps": fps}},
        "41": {"class_type": "SaveVideo", "_meta": {"title": "Save hero clip"},
               "inputs": {"video": ["40", 0], "filename_prefix": "rig/hero_clip", **save}},
    })
    return wf


def develop_video(video_name, disparity_video_name, n_frames, focus_d, blur, stock="film",
                  fps=25, crf=12):
    """One fixed focus for a whole clip (still per-frame depth, so focus holds through
    camera moves)."""
    return rack_focus_video(video_name, disparity_video_name,
                            [{"start": 0, "length": n_frames, "focus": focus_d}], blur, stock, fps, crf)


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else "A street at night."
    print(json.dumps(shoot(scene), indent=1))
