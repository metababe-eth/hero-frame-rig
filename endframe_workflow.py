"""Bounded first/last-frame graph; building it does not submit a job.

Both uploaded images must already use the same full-frame fit to 704 x 1280.
The retained start-image preprocessing and LTX guide encoder then resize that
canvas to each sampling stage. This helper never opens or changes either image.

Guide wiring follows core ComfyUI nodes_lt.py at
f9c6185d1acc05e0a01166567c8e292c23e8bd2f. This is a local source contract, not a
verification of the Cloud worker revision or of the generated clip's continuity.
"""
from __future__ import annotations

from typing import Any

import animation_workflow


WIDTH = 704
HEIGHT = 1280
FRAMES = 121
FPS = 25
END_FRAME_INDEX = FRAMES - 1
FIRST_IMAGE_NODE = "269"
LAST_IMAGE_NODE = "run06:end_frame"
STAGE_ONE_GUIDE_NODE = "run06:end_guide_1"
STAGE_TWO_GUIDE_NODE = "run06:end_guide_2"
FINAL_CROP_NODE = "run06:crop_end_guide_2"
OUTPUT_NODE = animation_workflow.OUTPUT_NODE

# The brief specifies an end-frame guide, but no guide strength. Use the pinned
# core schema's default in both stages; this is an implementation choice, not an
# empirically tuned value or a promise of exact last-image reconstruction.
END_GUIDE_STRENGTH_STAGE_ONE = 1.0
END_GUIDE_STRENGTH_STAGE_TWO = 1.0
GUIDE_STRENGTH_BASIS = (
    "Pinned core LTXVAddGuide default 1.0, applied independently in both stages; "
    "inferred choice because Brief06 does not specify guide strength."
)

PROMPT = (
    "He finishes pouring and lowers the green bottle down out of the frame with "
    "his right hand. Keeping the glass in his left hand, he turns to his left "
    "to face the camera and steps closer, carrying the glass around in front "
    "of him and raising it beside his face in a toast. Soap bubbles drift "
    "upward the whole time. One continuous shot, no cut and no dissolve."
)


def _end_guide(
    positive: list[Any], negative: list[Any], latent: list[Any], strength: float,
) -> dict[str, Any]:
    return {
        "class_type": "LTXVAddGuide",
        "inputs": {
            "positive": positive,
            "negative": negative,
            "vae": ["320:316", 2],
            "latent": latent,
            "image": [LAST_IMAGE_NODE, 0],
            "frame_idx": END_FRAME_INDEX,
            "strength": strength,
        },
    }


def build_endframe_workflow(
    first_image: str, last_image: str, seed: int, prompt: str = PROMPT,
    *, frames: int = FRAMES,
) -> dict[str, Any]:
    """Return a fresh, bounded API graph with a guide in both sampling stages.

    Only the first-stage noise seed is parameterized; the recovered refinement
    seed remains 42. The first-image strengths remain 0.7 then 1.0. Models,
    samplers, CFG, sigmas, preprocessing and the audio branch stay as recovered.
    A guide appends a reference latent: crop it after each stage, before video
    upsampling or final decoding. Never attach a guide to the combined AV latent.
    """
    if (not isinstance(last_image, str) or not last_image.strip()
            or "\x00" in last_image):
        raise ValueError("An uploaded last-image filename is required.")
    graph = animation_workflow.build_animation_workflow(
        first_image, prompt, seed,
        width=WIDTH, height=HEIGHT, frames=frames, fps=FPS,
    )
    graph[LAST_IMAGE_NODE] = {
        "class_type": "LoadImage",
        "inputs": {"image": last_image},
    }

    # Stage 1: condition the separate video latent and the corresponding guider.
    graph[STAGE_ONE_GUIDE_NODE] = _end_guide(
        ["320:304", 0], ["320:304", 1], ["320:296", 0],
        END_GUIDE_STRENGTH_STAGE_ONE,
    )
    graph["320:318"]["inputs"]["video_latent"] = [STAGE_ONE_GUIDE_NODE, 2]
    for node_id in ("320:314", "320:284"):
        graph[node_id]["inputs"]["positive"] = [STAGE_ONE_GUIDE_NODE, 0]
        graph[node_id]["inputs"]["negative"] = [STAGE_ONE_GUIDE_NODE, 1]
    # The recovered graph bypasses this crop's latent output when unguided.
    # With appended guides, that bypass would leak a reference frame downstream.
    graph["320:287"]["inputs"]["samples"] = ["320:284", 2]

    # Stage 2 starts from the cropped/upscaled video with start image reapplied.
    # Cleared stage-1 conditioning prevents duplicating its guide metadata.
    graph[STAGE_TWO_GUIDE_NODE] = _end_guide(
        ["320:284", 0], ["320:284", 1], ["320:288", 0],
        END_GUIDE_STRENGTH_STAGE_TWO,
    )
    graph["320:278"]["inputs"]["video_latent"] = [STAGE_TWO_GUIDE_NODE, 2]
    graph["320:282"]["inputs"]["positive"] = [STAGE_TWO_GUIDE_NODE, 0]
    graph["320:282"]["inputs"]["negative"] = [STAGE_TWO_GUIDE_NODE, 1]
    graph[FINAL_CROP_NODE] = {
        "class_type": "LTXVCropGuides",
        "inputs": {
            "positive": [STAGE_TWO_GUIDE_NODE, 0],
            "negative": [STAGE_TWO_GUIDE_NODE, 1],
            "latent": ["320:309", 0],
        },
    }
    graph["320:315"]["inputs"]["samples"] = [FINAL_CROP_NODE, 2]
    for node in (STAGE_ONE_GUIDE_NODE, STAGE_TWO_GUIDE_NODE):
        graph[node]['inputs']['frame_idx'] = frames - 1

    # Reuse the pinned builder's missing-link/cycle validation and prune to the
    # single retained SaveVideo. The helper has no client or upload side effects.
    return animation_workflow._output_closure(graph, OUTPUT_NODE)
