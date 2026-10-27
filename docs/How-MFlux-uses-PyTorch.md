# How MFlux uses PyTorch

## Summary

MFlux does all model math in MLX. PyTorch never runs a forward pass. MFlux uses PyTorch for only two jobs:

1. Read weight files that MLX cannot read (`.pt` / `.pth` pickles). The Depth Pro loader calls `torch.load` with `weights_only=False`, so a `.pt` file can run code when it loads. Use only Depth Pro files from a source that you trust.
2. Convert some bf16 safetensors through `torch` to float16 before MLX gets them.

All `import torch` lines are lazy. They are inside the function that needs them. Thus MFlux starts and runs without PyTorch until a code path needs one of these two jobs.

Test result (torch blocked with `sys.modules["torch"] = None`): all 46 CLI entry points import with no error. `transformers` 5.15 shows a warning that PyTorch is not found. The warning is harmless, because MFlux uses only the tokenizers and configs from `transformers`.

## Where MFlux calls torch

| File | What it does with torch | Who calls it |
|---|---|---|
| `models/common/weights/loading/weight_loader.py` | `torch_checkpoint`: `torch.load` of a `.pt` file | Depth Pro |
| same file | `torch_convert`: bf16 safetensors → float16 → MLX | FIBO (VAE, transformer, text encoder) |
| same file | `torch_bfloat16`: same conversion, read from an index file | FIBO VLM |
| `models/common/pid_decoder/pid_weight_mapping.py` | `torch.load` of the PiD `.pth` checkpoint | `--pid-decode` on all models that have it |
| `models/hed/hed.py` | `torch.load` of the HED `.pth` | Z-Image ControlNet, `hed` control |
| `models/openpose/openpose.py` | `torch.load` of the OpenPose `.pth` | Z-Image ControlNet, `pose` control |
| `models/qwen/tokenizer/qwen_vision_language_*.py` | Type check, only if a caller asks for `return_tensors="pt"` | No current call path asks for `"pt"` |
| `models/fibo_vlm/tokenizer/qwen2vl_processor.py` | Same type check | FIBO VLM asks for `"np"`, so this code does not call torch |

Paths are relative to `src/mflux/`.

## What runs without torch

These commands do not need torch, if you do not use a feature from "What needs torch" (for example `--pid-decode`):

- `mflux-generate` (FLUX.1), `-controlnet` (canny), `-fill`, `-redux`, `-kontext`, `-in-context*`, `mflux-concept*`
- `mflux-generate-flux2`, `-flux2-edit`, `-flux3`, `-flux3-edit`
- `mflux-generate-qwen`, `-qwen-edit`, `-qwen-2.1`, `-qwen-2.1-edit`, `-qwen-2.1-controlnet` (you supply a preprocessed control image)
- `mflux-generate-z-image`, `-z-image-turbo`
- `mflux-generate-z-image-controlnet` with `canny` or `mlsd` controls
- `mflux-generate-krea2`, `mflux-generate-krea2-controlnet` with `--depth-image-path`
- `mflux-generate-ernie-image*`, `-ideogram4`, `-lens`, `-ming`, `-iris`, `-boogu`, `-anima`
- `mflux-upscale-seedvr2`, `mflux-upscale-controlnet`
- `mflux-train`, `mflux-save` (for the models above), `mflux-info`, `mflux-check`, `mflux-capabilities`, `mflux-lora-library`
- FIBO, if you load a checkpoint that `mflux-save` wrote. The loader reads the mflux saved format first, and that format uses `mx.load`.

## What needs torch

| Feature | Why |
|---|---|
| `mflux-generate-depth` | It loads Depth Pro (`.pt`) at start, also with `--depth-image-path` |
| `mflux-save-depth` | Depth Pro |
| `mflux-generate-krea2-controlnet` from `--image-path` | Depth Pro estimates the depth map |
| `mflux-generate-z-image-controlnet` with `depth`, `hed` or `pose` | Depth Pro, HED, OpenPose weights are `.pt` / `.pth` |
| `--pid-decode` (any model) | PiD checkpoint is `.pth` |
| `mflux-generate-fibo`, `-fibo-edit` from the original Hugging Face repo | `torch_convert` |
| `mflux-refine-fibo`, `mflux-inspire-fibo` | `torch_bfloat16` |
| `mflux-save` of FIBO or FIBO VLM | It reads the original weights first |

Without torch, these features stop with `ImportError` when they load the weights. Other commands are not affected.

## macOS and Linux/CUDA

`pyproject.toml` lists `torch>=2.13.0,<3.0` for all platforms. The MLX line changes by platform:

- macOS: `mlx>=0.32.0,<0.33.0`
- Linux: `mlx[cuda13]==0.32.2`

On both platforms, MLX does the GPU work. Torch only reads files on the CPU (`map_location="cpu"`). On Linux, the default `pip install torch` downloads the CUDA build with the NVIDIA libraries, which is a large download. MFlux does not use torch CUDA. If you want torch for the features above, the CPU-only wheel is sufficient:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If the CUDA build is already installed, remove it first. `pip uninstall` alone does not install the CPU wheel:

```sh
pip uninstall -y torch && pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## How to install without torch

Torch is a hard dependency, so a normal install pulls it. Two options:

1. Install as usual, then remove torch: `pip uninstall torch`.
2. Install mflux with `--no-deps`, then install the other dependencies from `pyproject.toml` yourself.

After this, the features in "What needs torch" fail, and all other features operate as usual.
