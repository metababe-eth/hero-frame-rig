"""Build a bounded LTX I2V API graph from the recovered October 12 job.

No client, uploads or job submission occur here. The caller must supply an image
already fitted to the requested canvas: the inherited resize uses center crop.
Source blur/look remains baked into that image; this is not post-animation focus.
The original graph and its recovery provenance are preserved under workflows/.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE_JOB_ID = "50e5ea56-949b-49c8-aefa-ec6f247f1c52"
SOURCE_GRAPH_SHA256 = "508e14b76c40aeb63738d0758af9b04daf4ef2f912b91e2b2934a91a645030d9"
SOURCE_GRAPH_PATH = Path(__file__).resolve().parent / "workflows" / "ltx_sharp_recovered.json"
OUTPUT_NODE = "75"
REFINEMENT_SEED = 42
MAX_PIXELS = 704 * 1280
MAX_FRAMES = 121


def _output_closure(graph: dict[str, Any], output: str) -> dict[str, Any]:
    """Keep only ancestors of the output; reject broken or cyclic API links."""
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("The animation template has a dependency cycle.")
        if node_id in visited:
            return
        node = graph.get(node_id)
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            raise ValueError(f"The animation template has an invalid node: {node_id}.")
        visiting.add(node_id)
        for value in node["inputs"].values():
            if isinstance(value, list):
                if (len(value) != 2 or not isinstance(value[0], str)
                        or type(value[1]) is not int or value[1] < 0):
                    raise ValueError("The animation template has an invalid API link.")
                visit(value[0])
        visiting.remove(node_id)
        visited.add(node_id)

    visit(output)
    return {node_id: node for node_id, node in graph.items() if node_id in visited}


def build_animation_workflow(
    image: str,
    prompt: str,
    seed: int,
    *,
    width: int = 704,
    height: int = 1280,
    frames: int = 121,
    fps: int = 25,
) -> dict[str, Any]:
    """Return a fresh API dict; default output is 121 / 25 = 4.84 seconds.

    First-stage dimensions are half-size, so final dimensions must be multiples
    of 64. This helper deliberately caps work at the recovered output pixel and
    frame budgets; these bounds are application limits, not model maxima.
    The supplied seed controls first-stage noise; refinement retains seed 42.
    Audio generation, samplers, sigmas, conditioning and model choices are kept.
    """
    if not isinstance(image, str) or not image.strip() or "\x00" in image:
        raise ValueError("An uploaded image filename is required.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("An animation prompt is required.")
    if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
        raise ValueError("Seed must be an integer between 0 and 2**64 - 1.")
    if (type(width) is not int or type(height) is not int
            or width < 64 or height < 64 or width % 64 or height % 64
            or width * height > MAX_PIXELS):
        raise ValueError("Dimensions must be positive multiples of 64 within 704 x 1280 pixels.")
    if type(frames) is not int or not 9 <= frames <= MAX_FRAMES or (frames - 1) % 8:
        raise ValueError("Frame count must be 8n + 1, between 9 and 121.")
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError("Frame rate must be an integer between 1 and 60.")

    source = SOURCE_GRAPH_PATH.read_bytes()
    # A Windows Git checkout may use CRLF; permit that textual transformation
    # without relaxing the check for changes to graph content.
    if hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest() != SOURCE_GRAPH_SHA256:
        raise ValueError("The recovered animation template checksum does not match.")
    graph = json.loads(source)
    graph["269"]["inputs"]["image"] = image
    graph["320:319"]["inputs"]["value"] = prompt.strip()
    graph["320:277"]["inputs"]["noise_seed"] = seed
    graph["320:312"]["inputs"]["value"] = width
    graph["320:299"]["inputs"]["value"] = height
    graph["320:300"]["inputs"]["value"] = fps

    # Explicit aligned lengths replace the historical duration * fps + 1 graph.
    graph["320:295"]["inputs"]["length"] = frames
    graph["320:305"]["inputs"]["frames_number"] = frames
    # The recovered enhancer switch was false. Direct wiring makes the unused
    # prompt LLM and its auxiliary LoRA absent from the executable closure.
    graph["320:303"]["inputs"]["text"] = ["320:319", 0]
    return _output_closure(graph, OUTPUT_NODE)
