# Optical mode — local candidate for Brief 43

Optical is explicitly selected with `lens_mode="optical"`. The default of the
existing renderer, node and older recipe paths stays `legacy`. No public app or
Playground default is switched by this Part 1 change. Artist acceptance and a
live cpu-basic benchmark are pending.

The app adapter and ComfyUI node import the same implementation in `hfr_core`.
`optical.py` is part of that shared package; it is not a second app renderer.

## Appearance and settings

The defaults describe a **virtual** 50 mm, f/2.8 lens, 36 mm sensor long edge,
with normalized bright-near disparity mapped to inverse distances between
0.5 m and 10 m. DA3 relative depth does not measure these metres. The mapping is
a declared artistic calibration, not recovered camera metadata.

The thin-lens circle-of-confusion radius is
`f² × abs(1/z − 1/z_focus) / (2 × N × (1 − f/z_focus))`, converted to pixels.
A 0.55 pixel tolerance and a continuous soft knee preserve nearly focused
detail. Radius scales with resolution and stops at 4.5% of the long edge.
`blur_strength` remains a 0–50 gain; aperture is separately stored. Lower
f-numbers increase defocus; Optical's normal starting aperture is **2.8**.

Normalized antialiased disc kernels operate on linear RGB. No synthetic HDR
highlight gain, extra grading shoulder or grain is added. Highlights spread
into round patches without changing the source's overall display treatment.
This cannot recover a light already clipped or blurred in the source.

Twenty-four fixed inverse-depth layers composite premultiplied colour/alpha
from far to near. A normalized gather estimates the background immediately
behind a soft foreground edge; it excludes in-focus foreground colours and
weights the estimate by its coverage. A lone distant light cannot supply the
colour of a whole hidden surface.
Foreground blur spreads outward; sharp foreground regions overwrite farther
blur. RGB-guided local depth cleanup uses the existing `edge_cleanup` control.
It cannot identify transparency, reconstruct unseen surfaces, or fix a subject
whose depth is wrong across its whole area. Layers and occlusion fill are a CPU
approximation, not a ray tracer.

`depth_smoothing=0.5` enables a centered three-frame depth filter. Both RGB and
depth disagreement gate its neighbours so moving silhouettes and scene cuts
reset locally. RGB frames are never temporally blended. Set smoothing to 0 to
disable it; the original depth cache remains unchanged.

The automatic Optical plan holds at the start, eases through a broader quintic
pull, settles briefly at zero velocity and holds the final target. `arrival`
still maps to 50–95% of the clip. A supplied `focus_curve` is replayed verbatim.
Focus breathing is an optional crop, `breathing=0.01` to `0.02` for up to 1–2%
over a full-depth pull; **0 is the default**. It is not regenerated camera motion.

## Portable recipes

`hfr_core.optical.settings()` returns all mode values, including defaults:

```python
from hfr_core import render
from hfr_core.optical import settings

lens = settings(aperture=2.8, depth_smoothing=.5, breathing=0)
result = render(rgb, depth, focus_curve=saved_curve, **lens)
```

Store the complete `lens` dictionary alongside the exact `render_curve`, taps,
`blur`, `guide`, `edge_cleanup`, timing and original cached depth in the plan.
`focus_replay.workflow` writes these exact values to the HFR Focus Pull node;
the default node exposes Legacy / Optical as optional modes. An older plan
without `lens_mode` does not acquire any Optical fields or processing.
The Legacy function body, schedule, colour treatment and RNG remain unchanged.
New Comfy node widgets are appended after the existing optional widgets, so
old positional widget values retain their meanings.

## Validation

Run `python -m unittest tests.test_optical_focus tests.test_edge_cleanup -v` from
the app repository. Tests cover radiometric conservation, disc shape, sharp
foreground isolation, soft foreground coverage, aperture response, temporal
filter gating, timing, explicit-curve replay, node parity, saved widget order,
sparse-highlight fill and recipe settings.
Brief 43's separate local evidence compares the preserved baseline renderer
with this Legacy entry, including full encoded outputs for all eight clips.

Principles consulted, with no third-party implementation copied:
[GPU Gems 3, chapter 28](https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-28-practical-post-process-depth-field)
and [PBRT, projective cameras](https://pbr-book.org/4ed/Cameras_and_Film/Projective_Camera_Models).
