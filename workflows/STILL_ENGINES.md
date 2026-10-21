# Still-engine source and adaptation

Retrieved 2026-10-20 from the official Comfy-Org workflow templates:

- [FLUX 3 Image Klein distilled](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_flux3_klein_distilled_t2i.json)
- [MOSS-Image](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_moss_image_t2i.json)
- FLUX.2 Klein 4B: existing `rig_workflow._shoot_nodes` generation branch, unchanged.

`still_engines.py` flattens the first two templates into small API graphs. It
omits the optional prompt upsampler, keeping `/shoot_still`'s verbatim prompt
contract. MOSS's optional skip-layer guidance is off, matching the template's
false switch. FLUX 3 distilled uses 16 Euler/simple steps, CFG 1; MOSS uses
25 er_sde/simple steps, CFG 2.5 and the template's negative prompt. No API,
partner or Router node is present.

Availability requires the actual UNET, CLIP and VAE filenames, CLIP type and
graph node classes to appear in the selected backend's object_info. Subfolders
are accepted; the matching full path is used. FLUX 3 is the UI's first choice
when available. `engine=auto` preserves Klein 4B on Cloud and chooses the first
installed engine locally. Explicit unavailable selections fail before submission.

Model names are not weights: no models are installed or downloaded by selecting
an engine. The standalone focus pack is MIT; the app remains Apache-2.0. Model
and third-party node licenses apply separately.
