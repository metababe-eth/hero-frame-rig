# Current Shoot additions — 2026-10-21

FLUX 3 Image Klein distilled now uses core Cloud nodes with the current hyphenated
INT8 checkpoint, Qwen3-VL 4B INT8 encoder, FLUX.2 VAE and optional W4A8 BFL prompt
upsampler. See MODEL_LICENSES.md and workflows/STILL_ENGINES.md for exact files,
terms and runtime caption source. Local and Cloud routing uses backend.py; older
statements below about mandatory global credentials or Klein-only generation are
historical. No new Python dependency or packaged model weight is added.

# Still-workflow dependencies and replay status

Audit and desktop replay: 2026-10-12–13, America/Chicago. Initial source review: `740392f22e6c88279c6ed57abdad02b49ea8da98`; initial replay used recipes exported by application `a17d580e`. Current private deployment: `aab86d0865bb4c607f27727bd534c8cf19659b79`. Scope: the still-workflow source inventory, app preprocessing, and executed saved-take replay. The initial source audit installed nothing; the first replay phase installed an isolated Windows CPU runtime and used no model weights or Cloud jobs. The Oct 13 follow-up ran the clean installer and exactly two Cloud **Develop** jobs for new CPU-grain Film/CineStill references, followed by local replay. It ran no Shoot, Scan or animation jobs and downloaded no model weights; billed credits were unavailable. Animation is separately documented in `ANIMATION_HANDOFF.md`.

**Clean installation and CPU still-look replay have execution evidence.** See [DESKTOP_REPLAY.md](DESKTOP_REPLAY.md). Untouched Registry Radiance 2.3.3 with ComfyUI `f9c6185d` reproduces the initial NOLA Digital take within 1/255 at 19 pixels. New layered NOLA Film/CineStill takes explicitly record CPU grain and replay within 1/255 at 304/348 pixels respectively (900×1600 RGB; MAE 0.0000708333/0.0000805556 on the 0–255 scale). These are measured near-matches, not pixel-exact results. Legacy Product GPU-grain Film/CineStill takes still fail the comparison tolerance, with maxima of 16/255 and 15/255; their cause is unresolved. Cloud model hashes and worker revisions remain unknown; full local generation/scan/animation are outside this verification.

**Private until Maxwell authorizes publication.** This dependency inventory and successful replay checks do not authorize a public Space, repository mirror or competition submission.

**Correction to the earlier blur hypothesis:** source-compatible Registry 2.3.3 actually reproduces the saved pixels despite using Gaussian blur. The previous claim that the Cloud takes proved a true disc kernel was unsupported. The September Git candidate below is not the Registry artifact; pin the archive hash in DESKTOP_REPLAY.md, not a version label.

## Critical replay finding: Radiance versions are not interchangeable

The app sends `profile`, `intensity`, `seed`, `size`, `iso`, `halation`, `blend_mode`, `gate_weave` and `use_gpu` to `RadianceFilmGrain`. From `aab86d0865bb4c607f27727bd534c8cf19659b79`, still `develop()` sets `use_gpu=False` for this node. The shared `_finish()` default remains `True`; video paths retain GPU grain. Existing recipes retain their original settings and pixels.

- The inspected current upstream revision, [`0174a5811b64665714a366b209cfd73d44418e91`](https://github.com/FXTD-Studios/radiance/tree/0174a5811b64665714a366b209cfd73d44418e91), identifies itself as **3.5.4**. Its registered [grain class](https://github.com/FXTD-Studios/radiance/blob/0174a5811b64665714a366b209cfd73d44418e91/nodes/vfx/optics.py#L455) requires `grain_size`, `grain_strength`, three channel-size offsets, `hdr_aware` and `seed`. The [registration](https://github.com/FXTD-Studios/radiance/blob/0174a5811b64665714a366b209cfd73d44418e91/nodes/vfx/__init__.py) confirms which implementation is published. The exported app graph does not satisfy that schema. **Do not promise replay by installing the latest Radiance.**
- A bounded upstream history check found **schema-compatible candidate** [`cb7693b76de1d985deb16b3c1ecaff180a04020e`](https://github.com/FXTD-Studios/radiance/tree/cb7693b76de1d985deb16b3c1ecaff180a04020e), committed September 26, 2026, before the 3.5.0 replacement. It identifies itself as 2.3.3. Source inspection confirms all five required Radiance class registrations and matching required/optional input names. Its [grain implementation](https://github.com/FXTD-Studios/radiance/blob/cb7693b76de1d985deb16b3c1ecaff180a04020e/film/grain.py#L794) includes the app's profile-based interface.
- **That candidate is not an exact visual-replay pin.** Its [Octagon branch](https://github.com/FXTD-Studios/radiance/blob/cb7693b76de1d985deb16b3c1ecaff180a04020e/film/camera.py#L130) mixes Gaussian and anisotropic blur; it is not the disc kernel described in the app's Cloud development notes. Matching input names cannot establish matching optical behavior.
- The older tag spelled [`radinace2.3.3`](https://github.com/FXTD-Studios/radiance/tree/f262f47ddfda01ece154bf80c22769b1e4cef795) has the same `film/camera.py` and `film/grain.py` source as Registry 2.3.3 after CRLF normalization. [Its source](https://github.com/FXTD-Studios/radiance/blob/f262f47ddfda01ece154bf80c22769b1e4cef795/film/camera.py#L322) states that all bokeh shapes are Gaussian. The initial audit treated that as a reason to doubt replay; executed Digital replay corrected the earlier Cloud disc-kernel assumption. Pin the tested archive hash, not just the version label or these two matching files.
- The historical candidate's [requirements](https://github.com/FXTD-Studios/radiance/blob/cb7693b76de1d985deb16b3c1ecaff180a04020e/requirements.txt) ask for `Imath>=3.1.0`, while [PyPI Imath metadata](https://pypi.org/pypi/Imath/json) lists only 0.0.1 and 0.0.2. Current upstream [requirements](https://github.com/FXTD-Studios/radiance/blob/0174a5811b64665714a366b209cfd73d44418e91/requirements.txt) explain that Imath ships in the OpenEXR wheel. The executed Registry setup uses the bundled exact CPU package lock, including OpenEXR, instead of the obsolete upstream requirement; it installs no separate PyPI Imath package. This dependency overlay leaves the Registry node source untouched.

The initial Digital replay retained all processing parameters. The later Film/CineStill fix deliberately selects CPU grain in **new** still takes; two Cloud developments establish new references and leave historical takes unchanged. Recovering exact Cloud provenance remains necessary to explain the legacy GPU-grain mismatch. A true shaped-kernel upgrade would change behavior and needs separate visual evaluation; it is not part of this replay pass.

## Three different replay scopes

| Operation | Required model weights | Runtime nodes |
| --- | --- | --- |
| Replay a saved `takeNN_develop_graph.json` | **None**: saved image, disparity, layer, matte and plate PNGs are its inputs | ComfyUI image/mask/compositing nodes plus the five Radiance classes below |
| Analyze an uploaded working frame with `scan()` | DA3 Metric Large and BiRefNet | ComfyUI depth/background-removal/EXR output nodes |
| Generate a new frame with `shoot()` | Klein 4B, Qwen3 4B encoder, FLUX.2 VAE, DA3 Metric Large, BiRefNet | ComfyUI generation, depth/background-removal and output nodes |

The app's local `optics.py` computes disparity and subject layers between Cloud jobs. The recipe saves those derived PNGs, so replaying a developed take does not require rerunning those calculations or either analysis model. Re-running `frame_graph.json` is a separate scope.

## Model inventory: exact public artifacts

These are source-pinned, published artifacts matching the filenames in `rig_workflow.py`. Hugging Face's public file metadata supplied their SHA-256 values; this audit did not download the weights or compare them to Cloud's copies. `dependency_provenance.json` contains full revisions, sizes, hashes and download URLs.

| Graph filename | Local path under `ComfyUI/models/` | Public artifact at inspected revision | License evidence |
| --- | --- | --- | --- |
| `flux-2-klein-4b.safetensors` | `diffusion_models/` | [BFL Klein 4B, `e7b7dc27…`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/e7b7dc27f91deacad38e78976d1f2b499d76a294/flux-2-klein-4b.safetensors) | [BFL Apache-2.0 license](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/e7b7dc27f91deacad38e78976d1f2b499d76a294/LICENSE.md). This is the distilled **4B**, not base-4B or 9B. |
| `qwen_3_4b.safetensors` | `text_encoders/` | [Comfy repack, `5f526678…`](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/blob/5f526678002e43af5551dadb73ce2e8c91b43afe/split_files/text_encoders/qwen_3_4b.safetensors) | Repack [model card](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/blob/5f526678002e43af5551dadb73ce2e8c91b43afe/README.md) declares Apache-2.0; upstream [Qwen3-4B license](https://huggingface.co/Qwen/Qwen3-4B/blob/main/LICENSE) is Apache-2.0. |
| `flux2-vae.safetensors` | `vae/` | [Comfy Klein repack, `5f526678…`](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/blob/5f526678002e43af5551dadb73ce2e8c91b43afe/split_files/vae/flux2-vae.safetensors) | Repack card declares Apache-2.0 and points to BFL Klein sources. Use this identified artifact; a filename reused in another model repository is not itself license evidence. |
| `depth_anything_3_metric_large.safetensors` | `geometry_estimation/` | [Comfy DA3 repack, `1aaeea51…`](https://huggingface.co/Comfy-Org/Depth-Anything-3/blob/1aaeea516f22fb2f4c7fae761c8ed470c6cf9f93/geometry_estimation/depth_anything_3_metric_large.safetensors) | Repack and upstream [DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE) declare Apache-2.0. This exact variant matters; do not generalize a license to the whole DA3 family. |
| `birefnet.safetensors` | `background_removal/` | [Comfy BiRefNet repack, `25511f87…`](https://huggingface.co/Comfy-Org/BiRefNet/blob/25511f8787e51912e1480706b4e47b8f467fbf72/background_removal/birefnet.safetensors) | Repack card and upstream [BiRefNet card](https://huggingface.co/ZhengPeng7/BiRefNet) declare MIT; upstream [license text](https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE). This workflow does not use the separate Lucida checkpoint. |

The older `Comfy-Org/flux2-klein` URL currently redirects to `Comfy-Org/vae-text-encorder-for-flux-klein-4b` (spelling as published). The repacked Klein diffusion file has the same published SHA-256 as BFL's standalone file. Text-encoder and VAE repacks were not byte-compared to the original Diffusers layouts.

DA3's upstream [documentation](https://github.com/ByteDance-Seed/Depth-Anything-3#-model-zoo) describes canonical metric output and the focal-length correction `focal * net_output / 300`. This app assumes a selected full-frame-equivalent lens and 1008-pixel inference setting; its distance estimates are not measured scene geometry. Dependency availability does not validate that assumption for every uploaded image.

## Node inventory and code licenses

| Package | Classes used by the current still path | Version/provenance status |
| --- | --- | --- |
| **ComfyUI core**, [GPLv3 license](https://github.com/Comfy-Org/ComfyUI/blob/f9c6185d1acc05e0a01166567c8e292c23e8bd2f/LICENSE) | `UNETLoader`, `CLIPLoader`, `VAELoader`, `CLIPTextEncode`, `ConditioningZeroOut`, `CFGGuider`, `RandomNoise`, `KSamplerSelect`, `Flux2Scheduler`, `EmptyFlux2LatentImage`, `SamplerCustomAdvanced`, `VAEDecode`, `LoadImage`, `SaveImage`, `ImageToMask`, `MaskToImage`, `ImageCompositeMasked`, plus the analysis/output nodes in the next rows | Executed saved-take CPU replay pin: `f9c6185d1acc05e0a01166567c8e292c23e8bd2f`, identifies itself as 0.39.0. **Not the verified Cloud revision; generation and analysis classes were inspected, not locally executed.** |
| ComfyUI native DA3 | `LoadDA3Model`, `DA3Inference`, `DA3Render` | [Core source](https://github.com/Comfy-Org/ComfyUI/blob/f9c6185d1acc05e0a01166567c8e292c23e8bd2f/comfy_extras/nodes_depth_anything_3.py); no third-party DA3 node pack is required for these class names. |
| ComfyUI native background removal | `LoadBackgroundRemovalModel`, `RemoveBackground` | [Core source](https://github.com/Comfy-Org/ComfyUI/blob/f9c6185d1acc05e0a01166567c8e292c23e8bd2f/comfy_extras/nodes_bg_removal.py). The node implementation is part of ComfyUI; **MIT refers to the BiRefNet model/upstream project, not the entire ComfyUI node runtime.** |
| ComfyUI advanced image output | `SaveImageAdvanced` | [Core source](https://github.com/Comfy-Org/ComfyUI/blob/f9c6185d1acc05e0a01166567c8e292c23e8bd2f/comfy_extras/nodes_images.py#L1777), used here for 32-bit linear EXR. |
| **Radiance**, [current metadata: GPL-3.0-only](https://github.com/FXTD-Studios/radiance/blob/0174a5811b64665714a366b209cfd73d44418e91/pyproject.toml) | `HDRExpandDynamicRange`, `RadianceDepthOfField`, `HDRToneMap`, `ColorSpaceConvert`, `RadianceFilmGrain` | Public upstream is [FXTD-Studios/radiance](https://github.com/FXTD-Studios/radiance). Cloud has accepted the graphs in recorded jobs, but its pack revision is unknown. See the critical compatibility finding above. The older candidate also carries a [GPLv3 LICENSE](https://github.com/FXTD-Studios/radiance/blob/cb7693b76de1d985deb16b3c1ecaff180a04020e/LICENSE). |

The five Radiance operations used here are image processing. The still graph does not reference Radiance's optional AI models, relighting models or upscalers; installing the whole pack does not mean all its model downloads are required for this graph.

The Hero Frame Rig source is declared Apache-2.0 in its own `LICENSE`; it does not relicense ComfyUI, Radiance, their bundled libraries, or model weights. This review pass replaces the former abbreviated notice with the complete Apache text. Preserve upstream licenses and notices when bundling dependency code or weights; distributing a combined implementation requires checking the relevant GPL and other license conditions. This document is an inventory, not an “everything is Apache” claim or confirmation of competition-category eligibility.

## Python application environment

| Dependency | Repository constraint | License checked against upstream package/source | Pin status |
| --- | --- | --- | --- |
| `comfy-sdk` | `==0.5.0` | [MIT, PyPI 0.5.0](https://pypi.org/pypi/comfy-sdk/0.5.0/json) | Direct dependency pinned; transitive packages are not locked. |
| `gradio` | `==6.30.0` in `requirements-local.txt` and README `sdk_version` | [Apache-2.0, PyPI 6.30.0](https://pypi.org/pypi/gradio/6.30.0/json) | This pass pins the Space to the version observed in its live API configuration. |
| `pillow` | `>=10` | [10.0.0 metadata: HPND](https://pypi.org/pypi/Pillow/10.0.0/json) | Open range; audit the actual resolved wheel and bundled notices. |
| `numpy` | `>=1.26` | [BSD-3-Clause source notice at 1.26.0](https://github.com/numpy/numpy/blob/v1.26.0/LICENSE.txt) | Open range. |
| `OpenEXR` | `>=3.3` | [BSD-3-Clause source notice at 3.3.0](https://github.com/AcademySoftwareFoundation/openexr/blob/v3.3.0/LICENSE.md) | Open range; native wheels vary by Python/platform. |
| `opencv-python-headless` | `>=4.9` | [4.9.0.80 package metadata: Apache 2.0](https://pypi.org/pypi/opencv-python-headless/4.9.0.80/json) | Open range; wheel bundles have additional notices. |

These app requirements do not install ComfyUI, GPU PyTorch, Radiance or model weights. Use a separate app environment from the inference environment. Radiance's requirements use `opencv-python` while the app uses its headless variant; keeping the environments separate also avoids overlapping `cv2` distributions.

A resolved Windows CPU package lock and source archive hashes are supplied for **saved-take replay** in DESKTOP_REPLAY.md. The clean installer was executed successfully and its new runtime replayed the recorded Digital, CPU-grain Film and CPU-grain CineStill cases. They do not lock the Space application or identify its Cloud worker. Do not interpret the table's checked minimum-version license examples as exact versions installed on Spaces. A separate standalone CPU/CUDA grain diagnostic did not reproduce the legacy mismatch; it was not a full GPU recipe replay. General GPU replay parity remains unverified.

## Practical local replay procedure

**Use the executed Registry setup in [DESKTOP_REPLAY.md](DESKTOP_REPLAY.md).** The historical Git-candidate procedure below is retained as an audit trail, not the recommended replay setup. It was not executed and should not override the verified Registry archive pin.

1. Historical proposal: create an isolated ComfyUI installation using the [official manual installation guide](https://docs.comfy.org/installation/manual_install) and an appropriate GPU/PyTorch build. Record the chosen ComfyUI commit. At the initial audit this was source inspection only; the later pinned CPU saved-take setup above is the executed procedure.
2. Install Radiance under `ComfyUI/custom_nodes/radiance` at a deliberate commit. Prefer the actual recovered Cloud revision. For a **source-compatibility experiment only**, the historical candidate can be obtained with the commands below. Its dependency installation requires resolving the obsolete Imath requirement, and its blur source differs from the tested Registry archive. Do not use it as a pixel-equivalence baseline.

   ```powershell
   # Run from a separate, disposable ComfyUI checkout; these commands were not run by this audit.
   git clone https://github.com/FXTD-Studios/radiance.git custom_nodes/radiance
   git -C custom_nodes/radiance checkout --detach cb7693b76de1d985deb16b3c1ecaff180a04020e
   ```

3. Prepare the recipe offline in the app environment:

   ```powershell
   python validate_recipe.py hero_frame_recipe.zip --report validation.json
   python validate_recipe.py hero_frame_recipe.zip --prepare replay_bundle
   ```

4. Follow `replay_bundle/README.txt` (the validator-generated filename) and copy its namespaced `input/` directory into ComfyUI's input directory. Preserve names and relative paths. For an existing developed take, no model download is necessary.
5. Before queueing, inspect the target runtime's `/object_info` or UI node definitions. Confirm all `class_types` in the generated `validation.json` exist, including the **profile-based FilmGrain inputs**, and ensure no node import failures are hidden in startup logs. Merely finding a node with the same name is insufficient.
6. Load/submit one prepared `takeNN_develop_graph.json` using ComfyUI's API-graph support. This actually executes processing; the validator itself does not. Save job/runtime metadata, compare output size/pixels with the archived take, and inspect focus, bokeh, subject boundaries, grain and highlights. Record differences rather than silently changing settings to hide them.
7. Only for `scan()`/`shoot()` replay, obtain the needed artifacts from the pinned URLs in `dependency_provenance.json`, place them in the model directories above, and verify each local file's SHA-256 (PowerShell: `Get-FileHash -Algorithm SHA256 -LiteralPath <file>`). Rename neither a different model nor a quantization to match a filename.

To run the entire app against local ComfyUI through the SDK, local ComfyUI additionally needs the official [Comfy API proxy](https://github.com/Comfy-Org/comfy-api-proxy); ordinary ComfyUI `/prompt` and the SDK's API v2 are different interfaces. The [SDK guide](https://docs.comfy.org/development/api-development/sdks) describes `COMFY_BASE_URL=http://127.0.0.1:8189`. The app's `_client()` currently requires `COMFY_API_KEY` unconditionally, so local no-auth app operation still needs a deliberate adapter change. Direct desktop recipe replay does not require that app change or a Cloud account.

## Remaining gates before a reproducible public release

1. **Saved-take replay:** clean installer and runtime executed; Digital and new CPU-grain Film/CineStill cases pass the numerical comparison. Retain prompt/history, exact input/graph audits and comparisons. Legacy GPU-grain Film/CineStill failures remain unresolved; this does not certify every platform, input or historical recipe.
2. **Full local workflow:** generation, fresh depth/matte analysis, and animation still need their own execution checks. Cloud runtime identity is not inferred from numerical parity.
3. **Environment and model identity:** the desktop replay package versions are locked; executing-model hashes and the Cloud worker environment remain unverified. Spaces/local Gradio are pinned to 6.30.0; other application package ranges remain.
4. **Distribution/license packaging:** full app Apache text is now present; finish dependency notices and exact executed-model provenance. Keep repository, node-code and weight licenses distinct.
5. **Competition wording:** the inspected still graphs contain no partner-model/Router node calls, and the identified model cards publish their stated licenses. This is not an organizer ruling that the deployment meets every Open Weights Only condition. Cloud execution is not free merely because the weights are available.

The related clip helpers and future LTX feature are outside this still-only dependency certification. They introduce additional weights, licenses, FFmpeg/ffprobe requirements, codec behavior and graph recovery work.

## Brief05 still-kernel addition

The versioned round still graph also uses core `ImageBlur` and `ImageScale`; no
additional custom-node package or model is required. Existing `ImageToMask` and
`ImageCompositeMasked` nodes combine five persisted weight masks. The original
HDR/look nodes still require the pinned Radiance package. Both diagnostic cases
executed on Comfy Cloud and replayed within1/255 on the pinned core0.39.0 desktop
runtime. Cloud worker source identity is still not exposed or inferred. Historical
recipes retain their original graph and node requirements. The round helper is a
soft Gaussian calibrated to the legacy footprint, not a true aperture-disc model.

## Brief08 Guide addition

Positive Guide values use `RadianceGrade` from the already pinned Radiance 2.3.3
package, with manual neutral controls except equal RGB gain and saturation.
No external preset or reference input is used. Core `ImageToMask` and unclamped
`ImageCompositeMasked` blend each saved guide mask. Guide 0 adds no nodes.
Pinned local grading/compositing checks preserve HDR and the exact zero-mask
pixels; Cloud execution of this new branch awaits the approved bar comparisons.
No new models or packages are required.
