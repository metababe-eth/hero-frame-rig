# Optical mode — local candidate for Brief 43

Optical is explicitly selected with `lens_mode="optical"`. The default of the
renderer, node and older recipe paths stays `legacy`. New clip sessions in the
locally staged app start with Optical / Cinematic. No public deployment has
changed. Maxwell selected Optical on seven of eight Part 1 clips; Walkie's
truck needs a depth correction. That correction still needs visual acceptance,
and a live cpu-basic benchmark is pending.

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

## App presets and depth correction (Part 2)

One tap changes both aperture and the time at which focus arrives:

| Preset | Aperture | Arrival in the clip |
|---|---|---|
| Subtle | f/5.6 | 85% |
| Cinematic | f/2.8 | 70% |
| Dramatic | f/1.4 | 60% |

The exact numeric values, lens mode and curve are saved. A custom aperture or
timing does not acquire a preset name. Existing imported recipes retain their
original mode and settings. Optical can be switched back to Legacy.

The app warns when the two endpoint taps differ by less than 0.12 in normalized
estimated depth. This is a heuristic, not a measured scene distance. After a
depth analysis, further adjustments reuse the cached depth. Head-follow mode
can still require its own subject-analysis job; the job label remains explicit.

Under **More controls → Adjust a depth region**, tap a region on the Start or
End preview, then move **Farther ← depth nudge → nearer**. Negative values make
it farther; positive values make it nearer. The amber preview shows an estimated
depth selection inside a feathered screen-space ellipse. Adjust area size or
depth range to keep nearby subjects out. One correction is supported at a time.

This is **not object tracking or segmentation**. The screen-space region stays
fixed while its depth gate is evaluated in every frame; subjects at the same
estimated depth may also be affected. The preview uses the original endpoint
depth; rendering applies the saved correction after temporal and edge cleanup.
Inspect the full clip for moving-edge spill. Zero or Clear restores the original
depth treatment. Source RGB and the original depth cache are never edited.

Recipes store `depth_nudge` (position, amount, radius, depth band, anchor frame
and resolved anchor depth), the original depth and the final focus curve. The
node receives the same correction through the appended `depth_nudge_json`
widget. Apply it once, after depth preparation; do not export pre-nudged depth.
Changing a preset or correction invalidates the displayed directed result until
it is rendered again. Changing only the preview frame does not invalidate it.

Original cached relative-depth recipes at 24 or 25 fps can be reopened without
a new analysis. Neither a preview, a preset nor a saved-recipe download creates
a backend job.

## Validation

Run `python -m unittest discover -s tests -v` from
the app repository. Tests cover radiometric conservation, disc shape, sharp
foreground isolation, soft foreground coverage, aperture response, temporal
filter gating, timing, explicit-curve replay, node parity, saved widget order,
sparse-highlight fill and recipe settings.
Brief 43's separate local evidence compares the preserved baseline renderer
with this Legacy entry, including full encoded outputs for all eight clips.

Principles consulted, with no third-party implementation copied:
[GPU Gems 3, chapter 28](https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-28-practical-post-process-depth-field)
and [PBRT, projective cameras](https://pbr-book.org/4ed/Cameras_and_Film/Projective_Camera_Models).
