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
FLUX 3 and its encoder, while still checking required core classes and CLIP type.
A catalogue entry is discovery evidence, not proof that every variant executes.
Local discovery still requires the installed loader names; no weights download
occurs when selecting an engine. Auto chooses the first available engine, FLUX 3
then MOSS then Klein 4B. Explicit Klein 4B remains selectable.

UI FLUX 3 default is 2K area (2768x1520); 1K is 1376x752. API dimensions are
explicit and bounded. MOSS's optional SLG remains off and its generation has not
been tested. Its prepared graph uses 25 er_sde/simple steps and CFG 2.5.
Klein 4B's original generation graph is unchanged. The standalone focus pack is
MIT; app code is Apache-2.0; model licences are separate (MODEL_LICENSES.md).
