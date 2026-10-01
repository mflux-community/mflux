![image](src/mflux/assets/logo.jpg)

[![MFLUX](https://img.shields.io/pypi/v/mflux?label=MFLUX&logo=pypi&logoColor=white)](https://pypi.org/project/mflux/)
[![MLX](https://img.shields.io/pypi/v/mlx?label=MLX&logo=pypi&logoColor=white)](https://pypi.org/project/mlx/)
[![CI](https://github.com/filipstrand/mflux/actions/workflows/tests.yml/badge.svg)](https://github.com/filipstrand/mflux/actions/workflows/tests.yml)
[![Greptile: The War on Bugs](https://www.greptile.com/badge.svg)](https://www.greptile.com/?utm_source=oss_badge&utm_medium=readme&utm_campaign=greptile_for_open_source)

### About

Run the latest state-of-the-art generative image models locally on your Mac in native MLX!

### Table of contents

- [💡 Philosophy](#-philosophy)
- [💿 Installation](#-installation)
- [🎨 Models](#-models)
- [✨ Features](#-features)
- [🦄 Contributors](#-contributors)
- [🌱 Related projects](#related-projects)
- [🙏 Acknowledgements](#-acknowledgements)
- [⚖️ License](#%EF%B8%8F-license)

---

### 💡 Philosophy

MFLUX is a line-by-line MLX port of several state-of-the-art generative image models from the [Huggingface Diffusers](https://github.com/huggingface/diffusers) and [Huggingface Transformers](https://github.com/huggingface/transformers) libraries. All models are implemented from scratch in MLX, using only tokenizers from the [Huggingface Transformers](https://github.com/huggingface/transformers) library. MFLUX is purposefully kept minimal and explicit, [@karpathy](https://gist.github.com/awni/a67d16d50f0f492d94a10418e0592bde?permalink_comment_id=5153531#gistcomment-5153531) style.

---

### 💿 Installation
If you haven't already, [install `uv`](https://github.com/astral-sh/uv?tab=readme-ov-file#installation), then run:

```sh
uv tool install --upgrade mflux
```

After installation, the following command shows all available MFLUX CLI commands: 

```sh
uv tool list 
```

To generate your first image using, for example, the z-image-turbo model, run

```
mflux-generate-z-image-turbo \
  --prompt "A puffin standing on a cliff" \
  --width 1280 \
  --height 500 \
  --seed 42 \
  --steps 9 \
  -q 8
```

![Puffin](src/mflux/assets/puffin.png)

The first time you run this, the model will automatically download which can take some time. See the [model section](#-models) for the different options and features, and the [common README](src/mflux/models/common/README.md) for shared CLI patterns and examples.

<details>
<summary>Python API</summary>

Create a standalone `generate.py` script with inline `uv` dependencies:

```python
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mflux",
# ]
# ///
from mflux.models.z_image import ZImageTurbo

model = ZImageTurbo(quantize=8)
image = model.generate_image(
    prompt="A puffin standing on a cliff",
    seed=42,
    num_inference_steps=9,
    width=1280,
    height=500,
)
image.save("puffin.png")
```

Run it with:

```sh
uv run generate.py
```

For more Python API inspiration, look at the [CLI entry points](src/mflux/models/z_image/cli/z_image_turbo_generate.py) for the respective models.
</details>

<details>
<summary>⚠️ Troubleshooting: hf_transfer error</summary>

If you encounter a `ValueError: Fast download using 'hf_transfer' is enabled (HF_HUB_ENABLE_HF_TRANSFER=1) but 'hf_transfer' package is not available`, you can install MFLUX with the `hf_transfer` package included:

```sh
uv tool install --upgrade mflux --with hf_transfer
```

This will enable faster model downloads from Hugging Face.

</details>

<details>
<summary>DGX / NVIDIA (uv tool install)</summary>

```sh
uv tool install --python 3.13 mflux
```
</details>

---

### 🎨 Models

MFLUX supports the following model families. They have different strengths and weaknesses; see each model’s README for full usage details.

| Model | Release date | Size | Type | Training | Description |
| --- | --- | --- | --- | --- | --- |
|[Z-Image](src/mflux/models/z_image/README.md) | Nov 2025 | 6B | Distilled & Base | Yes | Fast, small, very good quality and realism. |
|[Krea 2](src/mflux/models/krea2/README.md) | Jun 2026 | 12B | Turbo (distilled) | No | Very good quality with a wide range of styles; good for creative exploration. |
|[FLUX.2](src/mflux/models/flux2/README.md) | Jan 2026 | 4B & 9B | Distilled & Base | Yes | Fastest + smallest with very good quality and edit capabilities. |
|[Ideogram 4](src/mflux/models/ideogram4/README.md) | Jun 2026 | 9B | Base | No | JSON-caption-native, typography-focused text-to-image generation. |
|[ERNIE-Image](src/mflux/models/ernie_image/README.md) | Apr 2026 | 8B | Distilled & Base | No | Single-stream DiT from Baidu. Vivid, high-contrast output. |
|[Lens](src/mflux/models/lens/README.md) | May 2026 | 3.8B (+20B TE) | Turbo (distilled) | No | Dual-stream MMDiT from Microsoft with a GPT-OSS text encoder. Strong prompt adherence in 4 steps. |
|[Ming-Image](src/mflux/models/ming_image/README.md) | Sep 2026 | 6.15B (+16B MoE TE) | Base | No | Design-focused (posters, cards, UI) with strong typography; outputs RGBA. |
|[Boogu Image](src/mflux/models/boogu/README.md) | Jun 2026 | 10B | Turbo (distilled) | No | DMD-distilled 4-step model with a photographic look and bilingual (EN/ZH) text rendering. |
|[FIBO](src/mflux/models/fibo/README.md) | Oct 2025+ | 8B | Distilled & Base | No | Very good JSON-based prompt understanding. Has edit capabilities. |
|[SeedVR2](src/mflux/models/seedvr2/README.md) | Jun 2025 | 3B & 7B | — | No | Best upscaling model. |
|[Qwen Image](src/mflux/models/qwen/README.md) | Aug 2025+ | 20B | Base | No | Large model (slower); strong prompt understanding and world knowledge. Has edit capabilities |
|[Qwen Image 2.1](src/mflux/models/qwen21/README.md) | Sep 2026 | 7.1B (+8B TE) | Base | No | Single-stream block-causal DiT with a Qwen3-VL text encoder; 40-step guidance-free sampling. [Multi-reference editing, LoRA, and RGBA](src/mflux/models/qwen21/reference/README.md). |
|[Depth Pro](src/mflux/models/depth_pro/README.md) | Oct 2024 | — | — | No | Very fast and accurate depth estimation model from Apple. |
|[FLUX.1](src/mflux/models/flux/README.md) | Aug 2024 | 12B | Distilled & Base | No (legacy) | Legacy option with decent quality. Has edit capabilities with 'Kontext' model and upscaling support via ControlNet |

---

### ✨ Features

**General**
- Quantization and local model loading
- LoRA support (multi-LoRA, scales, library lookup), including LyCORIS LoKr on FLUX.1 and FLUX.2
- Metadata export + reuse, plus prompt file support

**Model-specific highlights**
- Text-to-image and image-to-image generation.
- LoRA finetuning
- In-context editing, multi-image editing, and virtual try-on
- ControlNet (Canny), depth conditioning, fill/inpainting, and Redux
- Upscaling (SeedVR2 and Flux ControlNet)
- Depth map extraction and FIBO prompt tooling (VLM inspire/refine)

See the [common README](src/mflux/models/common/README.md) for detailed usage and examples, and use the model section above to browse specific models and capabilities.

> [!NOTE]
> As MFLUX supports a wide variety of CLI tools and options, the easiest way to navigate the CLI in 2026 is to use a coding agent (like [Cursor](https://cursor.com), [Claude Code](https://www.anthropic.com/claude-code), or similar). Ask questions like: “Can you help me generate an image using z-image?”




---

<a id="contributors"></a>

### 🦄 Contributors

<img src="https://contrib.rocks/image?repo=mflux-community/mflux" />

MFlux was originally created by [Filip Strand](https://github.com/filipstrand) in August 2024 and moved to this organisation in August 2026. It is maintained by:

<table>
<tr>
<td align="center" width="150"><a href="https://github.com/filipstrand"><img src="https://github.com/filipstrand.png?size=100" width="72" alt="filipstrand"><br><b>Filip Strand</b></a><br><sub>created mflux</sub></td>
<td align="center" width="150"><a href="https://github.com/anthonywu"><img src="https://github.com/anthonywu.png?size=100" width="72" alt="anthonywu"><br><b>Anthony Wu</b></a><br><sub>toolchain, CI, releases, mflux.web</sub></td>
<td align="center" width="150"><a href="https://github.com/plz12345"><img src="https://github.com/plz12345.png?size=100" width="72" alt="plz12345"><br><b>plz12345</b></a><br><sub>devops, Krea 2, Boogu</sub></td>
<td align="center" width="150"><a href="https://github.com/fxd0h"><img src="https://github.com/fxd0h.png?size=100" width="72" alt="fxd0h"><br><b>Mariano Abad</b></a><br><sub>ControlNets, training, Lens</sub></td>
<td align="center" width="150"><a href="https://github.com/ianscrivener"><img src="https://github.com/ianscrivener.png?size=100" width="72" alt="ianscrivener"><br><b>Ian Scrivener</b></a><br><sub>community, CUDA, model builds</sub></td>
</tr>
</table>

Where the models and the main features came from, read from the merge history (`gh pr view <n> --json author,mergedAt`):

| Model or feature | Contributor | PR |
|--|--|--|
| FLUX.1 | [@filipstrand](https://github.com/filipstrand) | initial release, 2024-08 |
| Depth Pro | [@filipstrand](https://github.com/filipstrand) | [#159](https://github.com/mflux-community/mflux/pull/159) |
| Qwen Image | [@filipstrand](https://github.com/filipstrand) | [#269](https://github.com/mflux-community/mflux/pull/269) |
| FIBO | [@filipstrand](https://github.com/filipstrand) | [#279](https://github.com/mflux-community/mflux/pull/279) |
| Z-Image | [@filipstrand](https://github.com/filipstrand) | [#284](https://github.com/mflux-community/mflux/pull/284) |
| SeedVR2 | [@filipstrand](https://github.com/filipstrand) | [#297](https://github.com/mflux-community/mflux/pull/297) |
| FLUX.2 Klein | [@filipstrand](https://github.com/filipstrand) | [#323](https://github.com/mflux-community/mflux/pull/323) |
| FLUX.2 KV cache (klein-9b-kv) | [@michaeltrefry](https://github.com/michaeltrefry) | [#426](https://github.com/mflux-community/mflux/pull/426) |
| ERNIE-Image | [@azrahello](https://github.com/azrahello) | [#417](https://github.com/mflux-community/mflux/pull/417) |
| Ideogram 4 | [@omercelik](https://github.com/omercelik) | [#433](https://github.com/mflux-community/mflux/pull/433) |
| Krea 2 | [@plz12345](https://github.com/plz12345) | [#453](https://github.com/mflux-community/mflux/pull/453) |
| LyCORIS LoKr adapters | [@JanGrohn](https://github.com/JanGrohn) | [#422](https://github.com/mflux-community/mflux/pull/422) |
| Fused-qkv LoRA loading | [@deadmansahil](https://github.com/deadmansahil) | [#459](https://github.com/mflux-community/mflux/pull/459) |
| Boogu-Image | [@plz12345](https://github.com/plz12345) | [#446](https://github.com/mflux-community/mflux/pull/446) |
| PiD pixel-diffusion decoder | [@azrahello](https://github.com/azrahello) | [#490](https://github.com/mflux-community/mflux/pull/490) |
| Z-Image Union ControlNet | [@fxd0h](https://github.com/fxd0h) | [#482](https://github.com/mflux-community/mflux/pull/482) |
| Krea 2 Raw, LoRA training, diffusers loading | [@fxd0h](https://github.com/fxd0h) | [#462](https://github.com/mflux-community/mflux/pull/462) |
| Lens (Turbo) | [@fxd0h](https://github.com/fxd0h) | [#510](https://github.com/mflux-community/mflux/pull/510) |
| mflux-capabilities, the machine-readable option contract | [@fxd0h](https://github.com/fxd0h) | [#499](https://github.com/mflux-community/mflux/pull/499) |
| Gradient checkpointing for training | [@qruz-hq](https://github.com/qruz-hq) | [#711](https://github.com/mflux-community/mflux/pull/711) |
| Denoised prediction for in-loop callbacks | [@IonDen](https://github.com/IonDen) | [#729](https://github.com/mflux-community/mflux/pull/729) |
| Qwen-Image-2.1 | [@ivanfioravanti](https://github.com/ivanfioravanti) | [#736](https://github.com/mflux-community/mflux/pull/736) |
| Qwen-Image-2.1 reference editing, RGBA, prefix cache | [@dreampuf](https://github.com/dreampuf) | [#741](https://github.com/mflux-community/mflux/pull/741), [#777](https://github.com/mflux-community/mflux/pull/777) |
| Qwen-Image-2.1 masks, auto-mask, strength, verify | [@flyingtimes](https://github.com/flyingtimes) | [#749](https://github.com/mflux-community/mflux/pull/749), [#764](https://github.com/mflux-community/mflux/pull/764) |
| PEFT and ComfyUI LoRA formats (Qwen 2.1, Z-Image) | [@phplego](https://github.com/phplego) | [#756](https://github.com/mflux-community/mflux/pull/756), [#768](https://github.com/mflux-community/mflux/pull/768), [#772](https://github.com/mflux-community/mflux/pull/772) |
| Text-prefix KV cache, Metal kernel, step cache | [@murphymatt](https://github.com/murphymatt) | [#778](https://github.com/mflux-community/mflux/pull/778), [#779](https://github.com/mflux-community/mflux/pull/779) |
| Ming-Image-0.1-Design | [@joeynyc](https://github.com/joeynyc) | [#765](https://github.com/mflux-community/mflux/pull/765) |
| load and generate entry points for UIs | [@IonDen](https://github.com/IonDen) | [#780](https://github.com/mflux-community/mflux/pull/780), [#785](https://github.com/mflux-community/mflux/pull/785) |
| mflux.web UI packages | [@anthonywu](https://github.com/anthonywu) | [#776](https://github.com/mflux-community/mflux/pull/776) |
| CI gates, justfile, ty, the pypi environment | [@anthonywu](https://github.com/anthonywu) | [#576](https://github.com/mflux-community/mflux/pull/576), [#590](https://github.com/mflux-community/mflux/pull/590), [#646](https://github.com/mflux-community/mflux/pull/646) |
| Release process and release notes | [@fxd0h](https://github.com/fxd0h) | [#685](https://github.com/mflux-community/mflux/pull/685) |
| test_tiny fixtures for every model | [@ianscrivener](https://github.com/ianscrivener), [@anthonywu](https://github.com/anthonywu) | [#611](https://github.com/mflux-community/mflux/pull/611), [#620](https://github.com/mflux-community/mflux/pull/620), [#599](https://github.com/mflux-community/mflux/pull/599) |

Everyone else who fixed, tested and reviewed is in the [contributor graph](https://github.com/mflux-community/mflux/graphs/contributors).

---

<a id="related-projects"></a>

### 🌱 Related projects

#### Build a UI under `mflux.web`

We welcome independently distributed `mflux.web.*` implementations using Gradio,
FastAPI, FastHTML, or any other UI framework. Each project can choose its own
framework, dependencies, release schedule, and launch command.

`mflux` owns the parent initializer and inference implementation. Its package
path extends across installed distributions, including separate editable
checkouts. `mflux.web` is an implicit namespace package: Python combines the
`mflux/web/` directories contributed by independently installed UI packages.
No intermediate distribution is required; neither core nor UI packages need to
depend on `mflux-web`. The [`mflux-web`](https://pypi.org/project/mflux-web/)
distribution reserves the generic `mflux-web` name on PyPI; it does not provide
the shared namespace and does not need to be installed. Its optional demo is
just another child module, `mflux.web.demo`.

Your UI distribution owns a unique child, for example
`src/mflux/web/example_ui/__init__.py`. With `uv_build`, configure
`module-name = "mflux.web.example_ui"`. Declare `mflux` and your chosen framework
as dependencies with versions your UI supports. The minimum `mflux` version for
split-directory installs must include this parent path extension.

Do not ship `mflux/__init__.py`, which belongs to core, or
`mflux/web/__init__.py`, which must remain absent for the implicit namespace.
Choose a unique child name to avoid collisions. PyPI distribution names can use
hyphens; Python child names must be valid identifiers, such as `example_ui`.
Reserving a PyPI distribution name does not reserve a Python namespace.

After installation in the same environment, consumers can use
`import mflux.web.example_ui`. Installing a UI does not automatically launch it,
discover applications, or mount routes. UI framework dependencies belong to the
individual UI distributions; this namespace support adds none to `mflux`.

#### Community applications

- [MindCraft Studio](https://themindstudio.cc/mindcraft#models) — macOS app built on mflux by [@shaoju](https://github.com/shaoju)
- [mflux-paint](https://github.com/Amo643/mflux-paint) — native macOS inpaint/edit app (pywebview), 16 models across edit/inpaint/text-to-image, mask painting, multi-seed batch, by [@Amo643](https://github.com/Amo643)
- [Mflux-ComfyUI](https://github.com/raysers/Mflux-ComfyUI) by [@raysers](https://github.com/raysers)
- [MFLUX-WEBUI](https://github.com/CharafChnioune/MFLUX-WEBUI) by [@CharafChnioune](https://github.com/CharafChnioune)
- [mflux-fasthtml](https://github.com/anthonywu/mflux-fasthtml) by [@anthonywu](https://github.com/anthonywu)
- [mflux-streamlit](https://github.com/elitexp/mflux-streamlit) by [@elitexp](https://github.com/elitexp)
- [mlx-taef](https://github.com/IonDen/mlx-taef) — TAESD/TAEF tiny-autoencoder live previews and low-memory FLUX decode for mflux, by [@IonDen](https://github.com/IonDen)
- [mlx-teacache](https://github.com/IonDen/mlx-teacache) — TeaCache step-skipping to speed up FLUX generation in mflux, by [@IonDen](https://github.com/IonDen)
- [MLXBits Image Studio](https://github.com/MLXBits/image-studio) - A native macOS Swift app for FLUX, Krea 2, Z-Image and more!
---

### 🙏 Acknowledgements

MFLUX would not be possible without the great work of:

- The MLX Team for [MLX](https://github.com/ml-explore/mlx) and [MLX examples](https://github.com/ml-explore/mlx-examples)
- Black Forest Labs for the [FLUX project](https://github.com/black-forest-labs/flux)
- Bria for the [FIBO project](https://huggingface.co/briaai/FIBO)
- Tongyi Lab for the [Z-Image project](https://tongyi-mai.github.io/Z-Image-blog/)
- Baidu for the [ERNIE-Image project](https://huggingface.co/baidu/ERNIE-Image)
- Ideogram for the [Ideogram 4 project](https://huggingface.co/ideogram-ai/ideogram-4-fp8)
- Krea.ai for the [Krea 2 project](https://www.krea.ai/blog/krea-2-technical-report)
- Qwen Team for the [Qwen Image project](https://qwen.ai/blog?id=a6f483777144685d33cd3d2af95136fcbeb57652&from=research.research-list)
- Microsoft for the Lens (Turbo) model, and Comfy-Org for the [weights repackage](https://huggingface.co/Comfy-Org/Lens)
- inclusionAI for the [Ming-Image project](https://github.com/inclusionAI/Ming-Image)
- The Boogu team for the [Boogu Image project](https://huggingface.co/Boogu/Boogu-Image-0.1-Turbo)
- ByteDance, @numz and @adrientoupet for the [SeedVR2 project](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
- Hugging Face for the [Diffusers library implementations](https://github.com/huggingface/diffusers) 
- Depth Pro authors for the [Depth Pro model](https://github.com/apple/ml-depth-pro?tab=readme-ov-file#citation)
- The MLX community and all [contributors and testers](https://github.com/filipstrand/mflux/graphs/contributors)

---

### ⚖️ License

This project is licensed under the [MIT License](LICENSE).
