# Private LTX animation template recovery — October 13, 2026

`ltx_sharp_recovered.json` is the unchanged 51-node API graph recovered from existing
Cloud job `50e5ea56-949b-49c8-aefa-ec6f247f1c52`. Its SHA-256 is
`508e14b76c40aeb63738d0758af9b04daf4ef2f912b91e2b2934a91a645030d9`.
`ltx_sharp_recovery_provenance.json` records the read-only retrieval. No credentials
were exported and no new generation was submitted by recovery.

## Builder contract

`animation_workflow.build_animation_workflow(image, prompt, seed, *, width=704,
height=1280, frames=121, fps=25)` returns a fresh API graph. Upload the already-fitted
image and bind its asset to node `269.image`; save/output node is `75`. The caller
must fit/letterbox to the requested aspect ratio before upload: the inherited
resize node center-crops mismatched input. A source image's existing blur/look is
baked in. This builder does not apply estimated-depth focus after animation.

The builder directly connects the supplied prompt to CLIP text encoding and keeps
only dependencies of output 75. The resulting 44 nodes retain the two-stage
samplers, sigmas, I2V strengths, audio generation, text encoder, checkpoint,
distilled LoRA and spatial upscaler. The inactive prompt enhancer, auxiliary
`gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors`, switch and preview are
absent. This removes an unused feature; it does not change a safety mechanism.
Original enhancement switch `320:328` was false.

Both video and audio latent lengths are explicitly set to 121 by default. The
original duration arithmetic requested 126 before model alignment; historical
output was reported as 121 frames. New default duration is 121/25 = 4.84 seconds.
The supplied seed controls first-stage node `320:277`; refinement node `320:276`
retains 42. Final dimensions must be divisible by 64 because the first stage is
half-resolution. The helper caps pixels at 704x1280 and frames at 121; these are
application test limits, not model capability limits. It neither uploads nor runs
jobs and does not enforce a cross-session number-of-jobs budget.

Seven offline tests check retained core stages, dependency closure, aligned
geometry and frame count, independent calls, invalid requests, and integrity of
the preserved graph. CRLF checkout normalization is accepted by its hash guard.
These tests do not prove current Cloud execution, output geometry, identity
preservation, camera control, exact replay or artist approval.

## Active weights and licenses

Public upstream metadata was checked October 13, 2026. Published file hashes and
revisions below identify available public artifacts, **not verified Cloud worker
bytes**. No model was downloaded by this audit. Matching filenames alone do not
prove the Cloud worker uses those files.

| Active file | Primary upstream and license evidence |
| --- | --- |
| `ltx-2.3-22b-dev-fp8.safetensors` | [Lightricks FP8 model](https://huggingface.co/Lightricks/LTX-2.3-fp8), LTX community license; not Apache-2.0. |
| `ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors` | [Comfy-Org exact repack](https://huggingface.co/Comfy-Org/ltx-2.3/blob/f20f3a54378001e5e6d642acc84719cc78addf4b/split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors); its repo declares the LTX community license. |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | [Lightricks LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3), with the full, distilled and upscaler families covered by the linked license. |
| `gemma_3_12B_it_fp4_mixed.safetensors` | [Comfy-Org repack](https://huggingface.co/Comfy-Org/ltx-2/tree/ccde4ba417d7900669fd56dd292a883cee11ff37/split_files/text_encoders). Google's [base model](https://huggingface.co/google/gemma-3-12b-it) and [Gemma Terms](https://ai.google.dev/gemma/terms) apply to Gemma/model derivatives; the surrounding Comfy-Org repo's LTX label does not replace Gemma terms. |

The pinned LTX model repository contains the January 5, 2026
[LTX-2 Community License](https://huggingface.co/Lightricks/LTX-2.3/raw/3c6a4e66e5d0a684231950b9c74dd4ded7b6fadc/LICENSE).
The current model card instead links the repository's
[LTX-2.x license](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x).
These are versioned license texts, not proof of which terms accompanied Cloud's
copy. Both impose conditions beyond Apache and include a revenue-based commercial
license threshold; preserve the applicable model notices in a release review.
Gemma terms also include distribution/hosted-service and use conditions. This is a
dependency inventory, not an organizer decision on Open Weights Only eligibility
or a completed distribution review. All source and examples remain private.

Active LTX, math and image-resize node implementations are present in the inspected
ComfyUI core `f9c6185d1acc05e0a01166567c8e292c23e8bd2f` under `nodes_lt.py`,
`nodes_lt_audio.py`, `nodes_lt_upsampler.py`, `nodes_math.py` and
`nodes_post_processing.py`; core is GPLv3. This builder introduces no Radiance
nodes. This local core revision was not established as Cloud's executing revision
and no local LTX execution was performed.

## Cost and next test

Comfy SDK 0.5.0 documents read-only `jobs.get(id).get_workflow()` returning API or
frontend-save format; no cost-estimation method was found in its public SDK.
[Source](https://github.com/Comfy-Org/comfy-python-sdk/blob/v0.5.0/src/comfy_sdk/jobs.py).
The current [Comfy credit guide](https://support.comfy.org/articles/5846341390-how-credits-work-in-comfy)
states 0.266 credits per active GPU second. Account-specific billing and runtime
remain unmeasured here. Video duration is not GPU runtime, so this evidence does
not support a per-clip price estimate or a claim that open weights mean free compute.

The proposed first test is one 704x1280, 121-frame, 25fps clip from a prepared
user-selected still, with a restrained prompt and fixed seed. Preserve input,
submitted graph, job ID, output and measured metadata, then visually inspect
subject/material identity and motion. Post-animation depth/focus and reliable
camera direction remain separate checks. Parent controls the authorized two-clip
budget and performs any Cloud submissions.
