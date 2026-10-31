# Model and runtime licence record

This record uses linked primary cards/licences and identifies exact named
variants; Cloud worker revisions and weight bytes have not been independently
audited. Install/download models under their own terms; app licences do not
relicense them.

## App runtime and optional generation dependencies

| Component | Licence | Primary source / scope |
| --- | --- | --- |
| FLUX 3 Image Klein distilled | FLUX Kommunity License v1.0 | [BFL](https://huggingface.co/black-forest-labs/flux-3-image-klein). Custom noncommercial terms with a qualifying-user commercial-output exception; read the full licence. Not Apache/MIT. |
| MOSS-Image SFT | Apache-2.0 | [OpenMOSS](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), [Comfy repack](https://huggingface.co/Comfy-Org/MOSS-Image). |
| FLUX.2 Klein **4B** | Apache-2.0 | [BFL](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B). Do not substitute 9B licence assumptions. |
| DA3 **Metric Large** | Apache-2.0 | [DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE); not a blanket statement about every DA3 variant. |
| LTX-2.3 dev FP8 + distilled LoRA | LTX-2 Community License Agreement | [Lightricks licence](https://huggingface.co/Lightricks/LTX-2.3/blob/main/LICENSE). Custom commercial/use/distribution conditions. [FP8 card](https://huggingface.co/Lightricks/LTX-2.3-fp8). |
| BiRefNet | MIT | [Upstream](https://huggingface.co/ZhengPeng7/BiRefNet), [Comfy repack](https://huggingface.co/Comfy-Org/BiRefNet). Matte, not face recognition. |
| Qwen3-4B and Qwen3-VL-4B-Instruct | Apache-2.0 | [Qwen3](https://huggingface.co/Qwen/Qwen3-4B), [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct). FLUX prompt encoders. |
| Qwen3.5-9B | Apache-2.0 | [Qwen](https://huggingface.co/Qwen/Qwen3.5-9B). MOSS encoder; quantized repack separately identified in the engine guide. |
| FLUX 3 Klein prompt upsampler and system instructions | Apache-2.0 per BFL | [BFL PUP](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup). Uses `qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors` on Cloud. The `t2i` system prompt is fetched at runtime; the exact instructions and rig additions travel in the saved graph. |
| FLUX.2 VAE / MOSS small VAE repacks | Apache-2.0 declared by those repacks | [Klein repack](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b), [MOSS repack](https://huggingface.co/Comfy-Org/MOSS-Image). FLUX 3 package terms remain its own. |
| Gemma 3 12B encoder for LTX | Gemma Terms of Use | [Gemma card](https://huggingface.co/google/gemma-3-12b-it), [terms](https://ai.google.dev/gemma/terms). Custom terms, not Apache-2.0. |
| comfy-sdk / comfy-python-sdk | MIT | [Licence](https://github.com/Comfy-Org/comfy-python-sdk/blob/main/LICENSE). Installed separately. |
| Comfy workflow templates | MIT | [Licence](https://github.com/Comfy-Org/workflow_templates/blob/main/LICENSE). Adapted graph notice retained in THIRD_PARTY_NOTICES.md. |
| ComfyUI | GPL-3.0 | [Licence](https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE). Separate runtime; not bundled with app code. |
| Radiance legacy custom nodes | GPL-3.0 | [Tested historical source](https://github.com/FXTD-Studios/radiance/blob/f262f47ddfda01ece154bf80c22769b1e4cef795/LICENSE). Used by legacy still/depth graphs, not vendored in the MIT shared renderer. |

## Demo-media production models (not app runtime)

These credits describe tools used to prepare demo footage/dialogue, not additional
features or dependencies installed by the app. No weights, demo media or voices
are bundled in this release. Area / Pick details itself uses local Gaussian
compositing and no model, depth inference, tracking or Cloud job.

| Model | Licence | Demo-media role / primary source |
| --- | --- | --- |
| LTX-2.5 | LTX-2.x Community License Agreement, custom terms | Demo footage generated on Comfy Cloud. [Model card](https://huggingface.co/Lightricks/LTX-2.5), [licence](https://github.com/Lightricks/LTX-2/blob/main/LICENSE). Separate from the app's LTX-2.3 default. |
| Chatterbox | MIT | Dialogue production. [Model card](https://huggingface.co/ResembleAI/chatterbox), [code licence](https://github.com/resemble-ai/chatterbox/blob/master/LICENSE). |
| Qwen3-TTS | Apache-2.0 | Dialogue production. [CustomVoice model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice), [code licence](https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE). |
| ACE-Step 1.5 XL Turbo | MIT | Earlier demo soundtrack study, not an app generation button or a claim about the locked V13 period-music source. [Exact variant](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo). |

Primary model cards and upstream licences checked 26 October 2026. The LTX licence
has custom use and distribution conditions; it is not an Apache/MIT licence.
Model terms do not grant rights to Max Krane's artwork or separate soundtrack media.

Other installed dependencies include Gradio (Apache-2.0), PyTorch (BSD-style),
NumPy (BSD-3-Clause), Pillow (HPND), OpenEXR (BSD-3-Clause), OpenCV (Apache-2.0)
and HTTPX (BSD-3-Clause). FFmpeg licensing depends on the installed build; no
FFmpeg executable is distributed. Consult the installed distributions' notices.

Max Krane retains his art and creative direction. CC0 source references do not
automatically make every remix or every asset in an archive CC0.

FLUX and Black Forest Labs names identify third-party models; they do not imply
endorsement of Hero Frame Rig. Model weight terms are separate from the app's
Apache-2.0 code and the focus pack's MIT code.
