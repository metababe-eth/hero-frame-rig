# ComfyUI-HeroFrameRig

Direct focus after the shot is made. MIT-licensed, model-free nodes and a shared
numpy/torch CPU implementation. Source blur cannot be undone. Estimated depth,
soft round Gaussian blur and spotlight grading are stylized effects.

Copy this entire folder to `ComfyUI/custom_nodes/ComfyUI-HeroFrameRig`, install
`requirements.txt` into ComfyUI's Python environment, and restart ComfyUI. For a
CPU installation use a CPU build of torch and run `python main.py --cpu`.
No model, network call, Cloud credential or app installation is needed to render
an existing RGB batch with an existing depth batch.

## Nodes

- **HFR Focus Pull**: RGB IMAGE batch and bright-near IMAGE/MASK depth; normalized
  start/end taps, blur strength 0–50, arrival 0–1, stay/closest/face, spotlight
  0–100, seed, and optional `edge_cleanup` 0–1 (default 0.5). Arrival maps to 50–95% of the clip, matching the app. The start and
  end depths are sampled on the first and last frames. Closest tracks central
  depth; face accepts an optional precomputed `head_targets_json`. Face without
  these targets fails clearly; the node does not hide a face model or silently
  change modes. `focus_curve_json` replays an exact saved schedule.
- **HFR Beat Focus**: AUDIO, images, depth and two taps. Detects spectral-flux
  transient peaks across the supplied audio, rounds them to the nearest video
  frame, and switches focus between taps at those frames. `audio_offset` selects
  a later section of a song. It detects onsets, not every musical beat; silence
  holds the near tap. The second output reports times and actual snap frames.
- **HFR Depth Sample**: trimmed-mean depth at a normalized tap and frame index.
- **HFR Load Depth Batch (replay)**: loads an exact float32 `depth` array from an
  NPZ in ComfyUI/input. No pickle; bounded input size. This small utility avoids
  depth quantization when replaying a recipe.

Depth can have one frame (broadcast) or match the full clip; a size mismatch is
resampled. Pixels must be finite RGB/depth values in 0–1. The renderer works on
CPU frame by frame; the total batch limit is 200 million pixels. Large batches
still require RAM for the input/output tensors. Split longer clips deliberately;
each independently planned clip has its own schedule.

## Workflows

The files in `example_workflows/` are **ComfyUI API-format graphs**, ready for
`client.workflows.from_json(...)` or `POST /prompt` with `{"prompt": graph}`.
They are not LiteGraph canvas exports. Set input filenames and normalized taps
before submitting. Put the referenced files in ComfyUI/input.

| File | Dependencies |
| --- | --- |
| `cached_depth_cpu.json` | `sharp.mp4`, `hfr_depth.npz`; CPU only, no models |
| `focus_pull_from_video.json` | Load Video → DA3 Metric → Focus Pull → Save Video; DA3 model required |
| `beat_focus.json` | `sharp.mp4`, `hfr_depth.npz`, `song.flac`; CPU only |
| `still_to_directed_clip.json` | FLUX 3 Klein → LTX-2.3 → DA3 → Focus Pull |
| `still_to_directed_clip_moss.json` | MOSS variant of the same chain |
| `still_to_directed_clip_klein4b.json` | FLUX.2 Klein 4B variant |

Generation/depth examples require the named weights and a current ComfyUI. They
are provided as graph templates; a CPU lens replay is not evidence that these
large models have been run on CPU. DA3 min/max outputs bright-near depth; unlike
the app's metric-depth conversion it normalizes each frame, which can breathe.
Prefer an app recipe's exact cached depth for a matched replay.

The full generation examples use the project's retained LTX-2.3 graph and
official Comfy still templates. Model licenses remain separate from this MIT
node code. See the app's workflow provenance and model notices before distributing
weights. No weights or media are bundled here.

## One implementation and limitations

The app imports `hfr_core` from this folder. Local ComfyUI runs the node; Cloud
analysis returns depth and the app runs this exact core on its host CPU. The
shared kernel retains HFR's focus scheduling, round blur footprint, foreground
spill, normalized background gather and spotlight. Its color shoulders and very
fine seeded sensor noise are independent implementations, not copied Radiance
code. Exact prior Radiance grain is not part of this MIT pack; compare the saved
regression evidence when reviewing the change.

The shared edge-ownership pass uses only RGB and depth: eroded near-depth support,
a bounded local search across depth jumps, thin RGB structure evidence, and a
feathered blend of the owning region's blur. It adds no model and contains no
shot-specific region, ellipse or object name. A small smooth blur floor removes
identity-image remnants from defocused thin edges. `edge_cleanup=0` exactly
bypasses the pass; new nodes default to 0.5. Both Focus Pull and Beat Focus expose
it. Old app recipes without this setting explicitly replay at 0. The raw depth
is never overwritten. This heuristic can soften other contrast edges, and it
cannot infer ownership when no valid near region exists nearby. Transparent
objects, reflections, fine occlusions and inaccurate depth remain limitations. These nodes never regenerate motion or recover detail
already blurred in the source.
