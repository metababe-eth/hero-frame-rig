# Still engines and runtime discovery

FLUX 3 is adapted from the fresh official Comfy-Org
[distilled template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_flux3_klein_distilled_t2i.json),
retrieved and executed on 2026-10-21 with ComfyUI v0.42.3. It uses
`flux3-image-klein-int8-convrot.safetensors`, Qwen3-VL 4B INT8, FLUX.2 VAE,
16 Euler/simple steps, CFG 1. No partner generation node is present.

Default expansion uses the core TextGenerate node and
`qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors`. BFL's Apache-2.0 `t2i`
system prompt is loaded from its PUP repository at runtime, then the rig appends
style/deep-focus/distance/no-brand constraints. It is not copied into this source
repository. Each executed graph retains the exact fetched text for replay.
The API PNG includes original prompt, actual expanded caption when available,
seed, dimensions, model, job and graph. The UI recipe retains the same information.

Cloud's object_info loader enums were a false negative for this executed model.
Backend discovery supplements them with live `/api/assets` loader paths for
every engine's UNET, encoder and VAE (plus FLUX 3 PUP), while still checking required core classes and CLIP type.
`engine_templates.py` fetches the current official templates once per hour and
extracts their loader names. A failed refresh disables the affected Cloud engine;
the bundled `engine_templates.json` is used only for offline local matching.
A catalogue entry is discovery evidence, not proof that every variant executes.
Local discovery still requires the installed loader names; no weights download
occurs when selecting an engine. Auto chooses the first available engine, FLUX 3
then MOSS then Klein 4B. Explicit Klein 4B remains selectable.

UI FLUX 3 and MOSS default is 2K area (2768x1520); 1K is 1376x752. API dimensions are
explicit and bounded. MOSS ran two 1280x704 scenes and a real 2768x1520 UI Shoot
with depth, matte, development and recipe on 2026-10-21. Its official
[template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_moss_image_t2i.json)
ordinary path uses 25 er_sde/simple steps and CFG 2.5, with SLG and caption expansion off.
Fresh filenames were `moss_image_sft_int8_convrot.safetensors`,
`qwen3.5_9b_w4a8_gptq.safetensors`, and `flux2-vae_small.safetensors`.
The same scene requests and seed 701 were compared with FLUX 3; conditioning is
not identical because those earlier FLUX 3 shots used PUP. MOSS foreground blur
limits deep-focus reuse and reliable hand inspection in the hallway comparison.
Klein 4B's original generation graph is unchanged. The standalone focus pack is
MIT; app code is Apache-2.0; model licences are separate (MODEL_LICENSES.md).

## Own-image entry and generated formats — 2026-10-23

The app opens with **Use my clip**, **Use my image**, and **Make a clip**.
Use my image presents upload first; generation is inside **Or generate a new image**.
Uploads preserve their own aspect and original file in the recipe. Their processing
copy remains limited to a 1600 px long side; choosing a generation format does not
crop, resize, or regenerate an uploaded image.

All three existing engines accept the generated image Format and Image size controls:

| Size | Landscape | Square | Portrait |
| --- | --- | --- | --- |
| 1K | 1376 × 752 | 1024 × 1024 | 752 × 1376 |
| 2K | 2768 × 1520 | 2048 × 2048 | 1520 × 2768 |

The size names describe approximate image area, not long-side resolution. Every
canvas remains on a 16 px grid and below 4.3 MP. Format and exact dimensions are
saved in the recipe. These are graph-supported choices, not a claim of a new
successful inference at every size. Engine availability remains inventory-based.

Calls to `app.shoot` that omit the new final `image_format` argument retain previous
dimensions, including Klein 4B's 1344 × 768 canvas. `/shoot_still` retains its explicit
width/height defaults and limits. No new model, HD adapter, sampling behavior or
image-editing algorithm is enabled by these UI changes.
