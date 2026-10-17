# Quantization by saved model

Models that `mflux-save` can save. Source: `MODEL_CLASSES` in `src/mflux/models/common/cli/save.py`.


| Line | Model name | TE Q | DiT Q | GS | Text encoder | HS | State(s) Used | FP8 | NVFP4 |
|:---:|---|:---:|:---:|:---:|---|:---:|---|:---:|:---:|
| 1 | `boogu-image-turbo` | ✅ | ✅ | 32 | Qwen3-VL | 4096 | Final |  |  |
| 2 | `ernie-image` | ✅ | ✅ | 64 | Ministral-3 3B | 3072 | Inter -2 |  |  |
| 3 | `ernie-image-turbo` | ✅ | ✅ | 64 | Ministral-3 3B | 3072 | Inter -2 |  |  |
| 4 | `fibo` | ✅ | ✅ | 64 | SmolLM3-3B | 2048 | Inter -2 |  |  |
| 5 | `fibo-edit` | ✅ | ✅ | 64 | SmolLM3-3B | 2048 | Inter -2 |  |  |
| 6 | `fibo-edit-rmbg` | ✅ | ✅ | 64 | SmolLM3-3B | 2048 | Inter -2 |  |  |
| 7 | `fibo-lite` | ✅ | ✅ | 64 | SmolLM3-3B | 2048 | Inter -2 |  |  |
| 8 | `flux2-klein-4b` | ✅ | ✅ | 64 | Qwen3-4B | 2560 | Inter 9, 18, 27 | [X](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) | [X](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4) |
| 9 | `flux2-klein-9b` | ✅ | ✅ | 64 | Qwen3-8B | 4096 | Inter 9, 18, 27 | [X](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8) | [X](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-nvfp4) |
| 10 | `flux2-klein-9b-kv` | ✅ | ✅ | 64 | Qwen3-8B | 4096 | Inter 9, 18, 27 | [X](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv-fp8) |  |
| 11 | `flux2-klein-base-4b` | ✅ | ✅ | 64 | Qwen3-4B | 2560 | Inter 9, 18, 27 | [X](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8) | [X](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-nvfp4) |
| 12 | `flux2-klein-base-9b` | ✅ | ✅ | 64 | Qwen3-8B | 4096 | Inter 9, 18, 27 | [X](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8) | [X](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-nvfp4) |
| 13 | `ideogram-4-fp8` | ✅ | ✅ | 64 | Qwen3-VL FP8 | 4096 | Inter 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35 | [X](https://huggingface.co/ideogram-ai/ideogram-4-fp8) |  |
| 14 | `krea-2` | ❌ | ✅ | 64 | Qwen3-VL-4B | 2560 | Inter 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35 |  |  |
| 15 | `krea-2-raw` | ❌ | ✅ | 64 | Qwen3-VL-4B | 2560 | Inter 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35 |  |  |
| 16 | `ming-image-design` | ✅ | ✅ | 64 | Ling-mini-2.0 MoE | 2048 | Final (layer 20) + Inter 5, 12, 20 |  |  |
| 17 | `qwen-image` | ❌ | ✅ | 64 | Qwen2.5-VL 7B | 3584 | Final |  |  |
| 18 | `qwen-image-2.1` | ❌ | ✅ | 64 | Qwen3-VL | 4096 | Final |  |  |
| 19 | `qwen-image-2.1-controlnet-union` | ✅ | ✅ | 64 | Qwen3-VL | 4096 | Final |  |  |
| 20 | `qwen-image-edit` | ❌ | ✅ | 64 | Qwen2.5-VL 7B | 3584 | Final |  |  |
| 21 | `qwen-image-edit-2511` | ❌ | ✅ | 64 | Qwen2.5-VL 7B | 3584 | Final |  |  |
| 22 | `z-image` | ✅ | ✅ | 64 | Qwen3 | 2560 | Inter -2 |  |  |
| 23 | `z-image-turbo` | ✅ | ✅ | 64 | Qwen3 | 2560 | Inter -2 |  |  |
| 24 | `z-image-turbo-controlnet-union-2.1` | ✅ | ✅ | 64 | Qwen3 | 2560 | Inter -2 |  |  |

**Legend:**
- ✅ means `-q` quantizes the component.
- ❌ means the model keeps the component in BF16 (`skip_quantization=True`).

- `GS` is the affine group size that `-q` uses.
- `HS` is the hidden size of the text encoder.
- `FP8` and `NVFP4` link to a release of that model by its original authors on Hugging Face. A blank cell means no such release was found. Community conversions are not listed.
- `Final` means the last hidden state. `Inter` means intermediate hidden states. Numbers are layer indexes. `-2` means the second from last.


**Notes:**
- `ming-image-design` takes a separate `--text-encoder-quantize` flag for the text encoder. It falls back to `-q`.
- `lens-turbo` is not in the table. `mflux-save` cannot save it. Its DiT takes `-q`. Its GPT-OSS encoder ships pre-quantized.
- `qwen-image-2.1` (text-to-image) skips the text encoder. The `qwen-image-2.1-controlnet-union` model does not skip it.
- Each ✅ means the component can be quantized. Some layers inside it stay in full precision.
- Black Forest Labs also publishes `FLUX.2-dev-NVFP4`. It is not a saveable model here, so it has no row.
- Third-party FP8 and NVFP4 builds of Z-Image-Turbo, ERNIE-Image, Qwen-Image and Ideogram 4 exist. They are not listed.
- The FP8 and NVFP4 links come from a web search. Nobody checked the linked pages.
