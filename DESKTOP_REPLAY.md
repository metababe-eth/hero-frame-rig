# Saved still recipes: desktop replay

This document describes replaying an exported **saved take**, with its existing pixels and maps.
It does not run FLUX, estimate depth again, animate, or connect to Comfy Cloud.

## What has actually run

October 12–13, 2026: an isolated Windows x64 / Python 3.11.1 installation executed the
exported NOLA take 6 through the local ComfyUI HTTP API. Processing parameters were
unchanged; input filenames were namespaced for the local installation. Comfy history
records a successful execution with no cached nodes. The output was retrieved from
that prompt's SaveImage result, not copied from the recipe.

- 900 x 1600 RGB output; 19 of 1,440,000 pixels differ from the Cloud reference.
- Maximum channel error: **1/255**; mean absolute error: **0.000004398/255**.
- 2.11 seconds submission-to-retrieval on this machine; core execution about 1.13 s.
- No Cloud jobs, model downloads, or credentials were used for desktop replay.
- This is numerical near-parity, not byte/pixel identity or a perceptual quality score.

The shipped installer then created a second clean environment. Its generated launcher
and `replay_desktop.py` executed the NOLA Digital recipe and two new layered look
recipes through ComfyUI. All three matched their Cloud references within **1/255**:

| Saved take | Pixels with any changed channel | Maximum error | Mean absolute error, 0–255 scale |
| --- | ---: | ---: | ---: |
| NOLA Digital, take 6 in the earlier archive | 19 | 1 | 0.000004398 |
| New NOLA Film, take 6 | 304 | 1 | 0.000070833 |
| New NOLA CineStill, take 7 | 348 | 1 | 0.000080556 |

The new Film/CineStill references required **two Cloud Develop jobs**, using an
existing frame and maps; no new Shoot or Scan ran. Still development now selects CPU
for the grain stage so this tested local path can reproduce the look. The saved
graphs record that choice. Video retains its previous GPU default; earlier takes
and input assets are unchanged.

**Older GPU-grain recipes are a known limitation.** Two original Product Film and
CineStill takes executed locally but failed strict comparison (maximum errors 16
and 15 respectively). Their exact cause remains unresolved; the CPU change does
not retroactively make those old recipes pass. A Product Digital take did pass at
1/255. This evidence supports the tested recipes and runtime, not every input.

These historical measurements cover saved still recipes on the pinned environment
below. They do not establish full application generation, analysis or animation
performance. Historical test media is not bundled in this code repository.

## Exact source and environment

| Component | Verified pin |
| --- | --- |
| ComfyUI | [`f9c6185d1acc05e0a01166567c8e292c23e8bd2f`](https://github.com/Comfy-Org/ComfyUI/tree/f9c6185d1acc05e0a01166567c8e292c23e8bd2f), version 0.39.0 |
| Radiance | [Registry 2.3.3 ZIP](https://cdn.comfy.org/fxtdstudios/radiance/2.3.3/node.zip), 1,568,338 bytes |
| Radiance ZIP SHA-256 | `9d53a1065bd46ab8d4d76b456e89b69c04ff0b9d8ba2fb5cbc71db2debd2fa66` |
| Python / Torch | Windows x64, Python 3.11.1; Torch 2.10.0+cpu, torchvision 0.25.0+cpu |
| Resolved Python packages | [`desktop-requirements-windows-cpu.txt`](desktop-requirements-windows-cpu.txt) |

All 90 files in the installed Radiance archive were hash-checked against that ZIP.
No node source was patched. The package installation uses OpenEXR's supplied Imath
module rather than the old unsatisfiable `Imath>=3.1.0` requirement, and installs the
headless OpenCV distribution. `uv pip check` passed for the resulting 91 packages.
The lock records package versions; it is not a hash lock of every platform wheel.

The matching output **does not identify the Cloud worker build**. It establishes
that this public source/environment can reproduce these saved inputs to the reported
tolerance. The Registry package and the September Git branch called 2.3.3 are not
interchangeable. Do not substitute the Git version label or latest Radiance.

## Fresh setup

From this repository checkout, with 64-bit Python 3.11 and Git available:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\setup_desktop_replay.ps1 -Destination .\desktop-replay -Python python
```

The installer requires a new directory and creates its own virtual environment,
pinned ComfyUI checkout and verified Radiance package. It leaves existing ComfyUI
installations alone. It installs runtime packages and the official Comfy frontend;
it does not download model weights or start a public server. This installer completed
a second clean installation, including package consistency and module checks. Its
generated launcher served the successful Digital, Film and CineStill replays above.
The installer itself does not submit a replay. The clean environment has the 91
locked runtime packages plus pip and setuptools supplied with the virtual environment.

Start the generated `desktop-replay/start-replay.ps1` in a separate terminal. It binds
only `127.0.0.1:8190`, uses CPU and offline mode, and loads only the Radiance custom
pack. Stop it with Ctrl+C when finished.

## Execute and compare a recipe

With the local server running, from the repository checkout:

```powershell
.\desktop-replay\.venv\Scripts\python.exe .\replay_desktop.py .\hero_frame_recipe.zip --server http://127.0.0.1:8190 --output .\replay-evidence
```

The historical seven-take example selected CineStill take 7; take 6 was Film.
That archive is not bundled. Supply your own exported recipe ZIP to the command above.

Add `--take 4` to choose another take or use a legacy recipe without `selected_take`.
The output directory must be new. This command really executes local processing:
it validates the archive, checks the runtime's nodes, uploads only the selected
take's inputs, records the submitted graph, waits for its result and compares that
result to the recipe. It does not submit `frame_graph.json`, regenerate the source,
or silently retry. Keep its history, input/graph audit, runtime and comparison files.

For comparison only, without submitting a job:

```powershell
python .\compare_replay.py .\hero_frame_recipe.zip .\local-output.png --report .\comparison.json
```

The default engineering gate is MAE <= 0.5 and maximum sample difference <= 2 on an
8-bit 0-255 scale; exact identity is reported separately. Layout/precision/color
metadata mismatches fail. Images are not resized, aligned or color-normalized to
make them pass. A numerical comparison alone never proves that a local job ran.

## Corrected optics claim

The graph's `bokeh_shape="Octagon"` is a legacy parameter. Registry 2.3.3's inspected
source uses Gaussian approximation for all bokeh shapes. The older claim that these
saved takes demonstrate a true octagonal/disc kernel is superseded. We retain the
parameter to reproduce the saved result; a shaped-kernel upgrade would be a deliberate
new rendering version with new visual tests, not an invisible replay fix.

ComfyUI and Radiance carry their own GPL licenses, retained in their downloads. The
Hero Frame Rig application's Apache license does not relicense those dependencies.

## Brief05 round still graphs (October 13)

New application still develops are versioned `core-round-gaussian-legacy-footprint-v1`.
They replace the truncated Radiance lens with core `ImageBlur`, `ImageScale`,
`ImageToMask` and `ImageCompositeMasked`, retaining saved weight PNGs in the recipe.
The helper does not change any historical graph. Replay recognizes the explicit
new topology while continuing to accept the existing Radiance path.

Two actual Cloud develops replayed on the pinned desktop runtime with maximum
1/255 error (rear17 changed pixels, low-angle11); the existing MAE/max gates are
unchanged. Zero-blur new/old outputs were pixel-identical in both tested scopes.
This does not imply identity to the input photograph: the existing expansion,
look and shoulders can change it. See `tests/design-20261013/brief05/RUN_REPORT.md`
for measured values, native comparisons and exact job records. Video lens graphs
still use the legacy kernel.

## Brief08 optional Guide graphs

Positive Guide takes append `guide-linear-blur-smoothstep-v1` to the round still
version. Their PNG masks are ordinary saved recipe inputs. The replay guard
recognizes only the fixed linear Grade/masked-composite branch, validates the
gain/saturation against the recorded Guide amount, and rejects preset files,
reference images or other extra Grade inputs. Old graphs and the Guide 0 graph
retain their previous behavior. Pinned local node checks pass; this is not yet
a measured replay of a guided Cloud output.
