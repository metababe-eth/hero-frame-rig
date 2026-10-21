# Model and runtime licence record

Prepared for Brief 33 from linked primary cards/licences. This records exact named
variants; Cloud worker revisions and weight bytes have not been independently
audited. Install/download models under their own terms; app licences do not
relicense them. Nothing in this document is a release approval.

| Component | Licence | Primary source / scope |
| --- | --- | --- |
| FLUX 3 Image Klein distilled | FLUX Kommunity License v1.0 | [BFL](https://huggingface.co/black-forest-labs/flux-3-image-klein). Custom noncommercial terms with a qualifying-user commercial-output exception; read the full licence. Not Apache/MIT. |
| MOSS-Image SFT | Apache-2.0 | [OpenMOSS](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), [Comfy repack](https://huggingface.co/Comfy-Org/MOSS-Image). |
| FLUX.2 Klein **4B** | Apache-2.0 | [BFL](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B). Do not substitute 9B licence assumptions. |
| DA3 **Metric Large** | Apache-2.0 | [DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE); not a blanket statement about every DA3 variant. |
| LTX-2.3 dev FP8 + distilled LoRA | LTX-2 Community License Agreement | [Lightricks licence](https://huggingface.co/Lightricks/LTX-2.3/blob/main/LICENSE). Custom commercial/use/distribution conditions. [FP8 card](https://huggingface.co/Lightricks/LTX-2.3-fp8). |
| ACE-Step **1.5 XL Turbo** | MIT | [Exact soundtrack variant](https://huggingface.co/ACE-Step/acestep-v15-xl-turbo), [1.5 family](https://huggingface.co/ACE-Step/Ace-Step1.5). Brief 31 graph requests `acestep_v1.5_xl_turbo_bf16.safetensors`, not the differently licensed v1 3.5B. |
| BiRefNet | MIT | [Upstream](https://huggingface.co/ZhengPeng7/BiRefNet), [Comfy repack](https://huggingface.co/Comfy-Org/BiRefNet). Matte, not face recognition. |
| Qwen3-4B and Qwen3-VL-4B-Instruct | Apache-2.0 | [Qwen3](https://huggingface.co/Qwen/Qwen3-4B), [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct). FLUX prompt encoders. |
| Qwen3.5-9B | Apache-2.0 | [Qwen](https://huggingface.co/Qwen/Qwen3.5-9B). MOSS encoder; quantized repack separately identified in the engine guide. |
| FLUX.2 VAE / MOSS small VAE repacks | Apache-2.0 declared by those repacks | [Klein repack](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b), [MOSS repack](https://huggingface.co/Comfy-Org/MOSS-Image). FLUX 3 package terms remain its own. |
| Gemma 3 12B encoder for LTX | Gemma Terms of Use | [Gemma card](https://huggingface.co/google/gemma-3-12b-it), [terms](https://ai.google.dev/gemma/terms). Custom terms, not Apache-2.0. |
| comfy-sdk / comfy-python-sdk | MIT | [Licence](https://github.com/Comfy-Org/comfy-python-sdk/blob/main/LICENSE). Installed separately. |
| Comfy workflow templates | MIT | [Licence](https://github.com/Comfy-Org/workflow_templates/blob/main/LICENSE). Adapted graph notice retained in THIRD_PARTY_NOTICES.md. |
| ComfyUI | GPL-3.0 | [Licence](https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE). Separate runtime; not bundled with app code. |
| Radiance legacy custom nodes | GPL-3.0 | [Tested historical source](https://github.com/FXTD-Studios/radiance/blob/f262f47ddfda01ece154bf80c22769b1e4cef795/LICENSE). Used by legacy still/depth graphs, not vendored in the MIT shared renderer. |

Other installed dependencies include Gradio (Apache-2.0), PyTorch (BSD-style),
NumPy (BSD-3-Clause), Pillow (HPND), OpenEXR (BSD-3-Clause), OpenCV (Apache-2.0)
and HTTPX (BSD-3-Clause). FFmpeg licensing depends on the installed build; no
FFmpeg executable is distributed. Consult the installed distributions' notices.

Maxwell retains his art and creative direction. CC0 source references do not
automatically make every remix or every asset in an archive CC0. Historical private
Space test assets are not approved for public release by this code change.
