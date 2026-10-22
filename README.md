---
title: Hero Frame Rig
emoji: 🎞️
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Direct the focus of any clip after it's made.
---

# Hero Frame Rig

**Higgsfield's lens controls, rebuilt on open weights, and you direct the focus after the shot is made.**

“Other tools ask the model to rack focus. Hero Frame Rig lets you decide after it's made, and change your mind.”

Bring a clip you care about. Tap what should be sharp at the start and end, compare
the treatment with your original, and keep the clip plus a replayable recipe.
Stills, generated motion and reusable ComfyUI focus nodes are part of the same tool.
This is an independent project, with no affiliation or endorsement from Higgsfield.
The reference is its publicly observed lens/aperture controls and rack-focus prompt
chip; we have not tested its output quality or ruled out its post-generation tools.

## How it works in five lines

1. Upload your own frame/clip, or generate a still with an available engine and motion with LTX-2.3.
2. Depth Anything 3 estimates distance; optional BiRefNet subject analysis supplies a head target.
3. Choose two focus points and a timing; the saved shot and its motion stay the same.
4. One shared CPU core applies depth-directed blur, edge cleanup and optional spotlight grading.
5. Compare Before/After, download the clip, and export source, depth, settings and a ComfyUI workflow.

## Run on your own ComfyUI

Use Python 3.11+, FFmpeg/ffprobe on PATH and a current ComfyUI. Keep the app's CPU
Python environment separate from your GPU ComfyUI environment. Start ComfyUI as
usual; its default address is `http://127.0.0.1:8188`.

```powershell
git clone https://huggingface.co/spaces/metababa/hero-frame-rig-release
cd hero-frame-rig-release
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:HFR_BACKEND = "local"
$env:COMFY_LOCAL_URL = "http://127.0.0.1:8188"
python app.py
```

This release candidate is private pending maintainer review. Authorized Hugging Face
access is currently required to clone. On macOS/Linux activate with `source .venv/bin/activate` and use
`export HFR_BACKEND=local`. The app manages the official loopback-only
`comfy-api-proxy`, adapting standard ComfyUI to the SDK's v2 API. It exits with the
app. Select **Local ComfyUI → Connect / refresh models** in Backend.

Install the focus pack below. For fresh analysis/generation also install the exact
models listed in the [engine guide](workflows/STILL_ENGINES.md) and the example
graphs. The existing still-development and metric-depth packing graphs require
Radiance nodes (the tested API was 2.3.3) as well as current ComfyUI's DA3/BiRefNet
nodes. [Model and runtime licences](MODEL_LICENSES.md) remain separate. Missing
models are shown explicitly; selecting an engine does not download weights.

## Run on Comfy Cloud with your key

Run the app locally or open the hosted Space. Choose **Comfy Cloud**, enter your
Comfy API key in the password field, then **Connect / refresh models**. The entered
key belongs to that browser session. Do not put it in a prompt or recipe.
Cloud does depth/generation work; directed-clip focus runs on the app host CPU.
A cached clip can be redirected without another Cloud job. Fresh `/direct_clip`
analysis is one Cloud job; `/shoot_still` is one. Failed calls do not auto-retry.

A loopback-only local session can use its own `COMFY_API_KEY` environment variable.
Public origins require BYOK. The private Space's owner-key fallback is gated by
`HFR_PRIVATE_HOSTED=1` and a private-visibility check; remove that flag **and the
owner secret before public release**. Public metadata disables the fallback.
No key is serialized into workflows or recipes. A hosted Space cannot reach your
PC at localhost; run the app on your computer to use local ComfyUI.

## Install the node pack

Copy `ComfyUI-HeroFrameRig` into `ComfyUI/custom_nodes/ComfyUI-HeroFrameRig`, then
run this with **ComfyUI's Python**, and restart ComfyUI:

```powershell
python -m pip install -r custom_nodes/ComfyUI-HeroFrameRig/requirements.txt
python main.py
# CPU-only replay: python main.py --cpu
```

The pack is MIT, with numpy/torch only: **HFR Focus Pull**, **HFR Beat Focus**,
**HFR Depth Sample**, and **HFR Load Depth Batch**. A cached RGB/depth replay needs
no model or network. Beat Focus detects audio transients and snaps focus at the
nearest frame; this is not guaranteed musical downbeat detection.

[Pack instructions and six example workflows](ComfyUI-HeroFrameRig/README.md)
cover cached-depth CPU replay, video → depth → focus, beat focus, and still → LTX
→ depth → focus for each of the three still engines. These are **API-format JSON**,
submitted through comfy-sdk or ComfyUI `/prompt`, not canvas exports. Generation
templates require the named weights; they are not proof that those models ran.
The app and nodes import the same `hfr_core`; no duplicate renderer is maintained.

## Still engines: what runs where

| Engine | Local ComfyUI | Comfy Cloud |
| --- | --- | --- |
| FLUX 3 Image Klein, distilled | Enabled with installed weights/core nodes. Local generation not yet validated. | Core-node execution confirmed 21 October 2026. Default when available. |
| MOSS-Image SFT | Enabled with installed weights/core nodes. Local generation not yet validated. | Core-node execution confirmed 21 October 2026, including 2K Shoot, analysis, development and recipe. |
| FLUX.2 Klein 4B | Supported when installed. | Executed and checked; selectable fallback. |

Cloud discovery checks both the live model asset catalogue and node schema;
Cloud's loader dropdown enums alone can omit executable models. Model filenames
come from current official Comfy-Org templates, refreshed hourly; local discovery
uses the dated bundled template snapshot against only that server's inventory.
A failed template refresh disables that engine until discovery succeeds. Discovery
is not an execution guarantee. A failed job is never automatically resubmitted.

FLUX 3 and MOSS Shoot default to **2K area** (2768x1520 widescreen), with a **1K** option
(1376x752). Expand prompt runs the open-weight BFL caption model in the same job,
using its runtime-fetched t2i instructions plus deep-focus, style, distance and
no-brand rules. Turn it off to supply your own caption verbatim. Klein 4B remains
selectable and keeps the existing UI generation size. The user request, actual
expanded caption when returned by history, model files and graph are preserved
in the recipe/API PNG. Missing caption weights fail before submission with an
explicit suggestion to turn expansion off; there is no silent hosted LLM fallback.
MOSS uses the official ordinary sampling path, 25 er_sde/simple steps at CFG 2.5,
with caption expansion and SLG off. Its two tested scene requests produced softer
foreground detail than requested; successful execution does not imply a quality win.
The caption model can still change composition or ignore constraints, including
text suppression. Review the result; use a supplied caption with expansion off
when exact wording matters. Deep focus in a prompt is not a sharpness guarantee.

## Agent API

Two stateless Gradio endpoints are documented in the live schema and `agents.md`.
The private Space additionally needs an authorized HF token. For a future public
Space omit that token, but keep your own Comfy key header. This example performs
paid Cloud work when executed:

```python
import os
from gradio_client import Client, handle_file
from huggingface_hub import get_token

client = Client("metababa/hero-frame-rig", token=get_token(),
                headers={"X-Comfy-API-Key": os.environ["COMFY_API_KEY"]})
directed, share, recipe = client.predict(
    video=handle_file("clip.mp4"), start_x=.25, start_y=.45,
    end_x=.75, end_y=.55, edge_cleanup=.5, api_name="/direct_clip")
png = client.predict(prompt="A weathered brass payphone beside a red cup, deep focus",
                     seed=1102, width=1280, height=704,
                     engine="auto", api_name="/shoot_still")
client.close()
```

`/direct_clip`: video, four normalized coordinates, then optional
`in_between='face'` (`closest`/`stay`), `blur=1`, `arrival=7/9`, `spotlight=60`,
`edge_cleanup=.5`. The existing in-between default is unchanged. Arrival 0–1 maps
to 50–95% of the clip. Returns directed MP4, muted portrait Share MP4, recipe ZIP.
The upload path takes the first 97 frames at 25 fps (3.88 seconds), preserves aspect
within 1280 pixels / 901120 pixels total, does not upscale, and removes audio.
Share plays the comparison twice and crops around the end focus point.

`/shoot_still`: prompt, optional seed=1102, width=2768, height=1520, engine='auto'
(`flux3`/`moss`/`klein4b`), expand_prompt=True. Returns a PNG with embedded
provenance. Explicit smaller dimensions such as 1280x704 remain supported.
Dimensions must be multiples of 16, 256–4096 per side, at most 4300000 pixels.
Auto prefers FLUX 3 when found. Expansion applies only to FLUX 3; set it False
for verbatim conditioning. Explicit unavailable engines fail before submission.

## Replay and limits

Directed recipes contain `workflow.json`, `sharp.mp4`, exact float `hfr_depth.npz`,
taps, schedule, seed, edge setting, graphs and job receipts. Put the two inputs
in ComfyUI/input and submit the workflow with the pack installed. Older recipes
without an edge setting reopen at **0**, preserving the previous treatment;
new clips default to **0.5**. Video codecs can introduce small pixel differences.

This is **depth-directed focus, not real optics**. It cannot recover detail already
blurred in a source or repair malformed generated motion. Prompt-based lens framing
can regenerate composition. Depth estimates can fail at mirrors, transparency,
hair and occlusions. Face follow uses a subject-top heuristic, not recognition.
The no-model edge pass looks for RGB structure beside a nearer solid depth region,
then feathers the owning region's blur treatment instead of interpolating depth
through the focus plane. It is a local heuristic, not a glass/material classifier.
Use **More controls → Edge cleanup** (0–1); 0 exactly bypasses the new pass.
The same setting is on Focus Pull and Beat Focus. The still-test harness uses the
shared node core; the separate legacy Stills development graph is unchanged.
The renderer is CPU-bound and large batches use RAM. No new model weights are added.

## Credits and licences

App code: [Apache-2.0](LICENSE). Standalone focus pack:
[MIT](ComfyUI-HeroFrameRig/LICENSE). Maxwell's artwork, source footage, portraits,
recipes containing his media and demo exports are his creative material; neither
code licence grants blanket reuse rights to that media or third-party references.
Only media explicitly selected by Maxwell should be published.

| Model or dependency | Licence | Role |
| --- | --- | --- |
| [FLUX 3 Image Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) | FLUX Kommunity License v1.0, custom terms | Default available still engine; executed on Cloud |
| [FLUX 3 prompt upsampler](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup) | Apache-2.0 | Optional open-weight caption expansion; system text fetched at runtime |
| [MOSS-Image SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) | Apache-2.0 | Optional prepared still engine |
| [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Apache-2.0 | Executed still engine |
| [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3/blob/main/LICENSE) | LTX-2 Community License Agreement, custom terms | Core motion model |
| [Depth Anything 3 Metric Large](https://huggingface.co/depth-anything/DA3METRIC-LARGE) | Apache-2.0 | Estimated depth |
| [ACE-Step 1.5 XL Turbo](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo) | MIT | Locally assembled demo soundtrack; not an app generation button |
| [comfy-sdk](https://github.com/Comfy-Org/comfy-python-sdk/blob/main/LICENSE) | MIT | Local/Cloud workflow client |
| [Comfy workflow templates](https://github.com/Comfy-Org/workflow_templates/blob/main/LICENSE) | MIT | Adapted graph sources; [notice retained](THIRD_PARTY_NOTICES.md) |

See [the detailed licence list](MODEL_LICENSES.md) for text encoders, VAEs, matte
model and runtime dependencies. Open weights does not mean every model is under
an OSI open-source licence. No weights or runtime binaries are redistributed.

This code-only release candidate is private while the maintainer reviews deployment
and the demo. No sample media or owner credentials are bundled.
