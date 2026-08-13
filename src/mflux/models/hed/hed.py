"""HED (holistically-nested edge detection) soft-edge preprocessor, native in MLX.

This is lllyasviel's ControlNetHED reimplementation (Apache-2.0, the `ControlNetHED_Apache2` weights that
controlnet_aux uses): the input minus a learned per-channel mean is run through five VGG-style blocks, each
emitting a 1-channel side output; the five side outputs are resized to the input resolution, averaged, and
passed through a sigmoid to give a soft edge map. Only weight loading touches torch (a .pth of half tensors);
every forward pass is MLX, matching how the rest of mflux runs.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import PIL.Image

# block name -> number of 3x3 conv layers before the 1x1 side-output projection
_BLOCKS = [("block1", 2), ("block2", 2), ("block3", 3), ("block4", 3), ("block5", 3)]
_HF_REPO = "lllyasviel/Annotators"
_HF_FILE = "ControlNetHED.pth"


class _HEDNet(nn.Module):
    def __init__(self, state: dict[str, mx.array]):
        super().__init__()
        # torch stores the input mean as (1, 3, 1, 1); MLX runs NHWC so it is (1, 1, 1, 3).
        self.norm = state["norm"].transpose(0, 2, 3, 1)
        self.blocks = []
        for name, n_conv in _BLOCKS:
            convs = [_HEDNet._conv(state, f"{name}.convs.{c}", kernel=3, padding=1) for c in range(n_conv)]
            projection = _HEDNet._conv(state, f"{name}.projection", kernel=1, padding=0)
            self.blocks.append((convs, projection))

    @staticmethod
    def _conv(state: dict[str, mx.array], prefix: str, *, kernel: int, padding: int) -> nn.Conv2d:
        # torch Conv2d weight is OIHW; MLX Conv2d weight is OHWI.
        weight = state[f"{prefix}.weight"].transpose(0, 2, 3, 1)
        bias = state[f"{prefix}.bias"]
        conv = nn.Conv2d(weight.shape[3], weight.shape[0], kernel_size=kernel, padding=padding)
        conv.weight = weight
        conv.bias = bias
        return conv

    def __call__(self, image_nhwc: mx.array) -> list[mx.array]:
        h = image_nhwc - self.norm
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        projections = []
        for i, (convs, projection) in enumerate(self.blocks):
            if i > 0:
                h = pool(h)
            for conv in convs:
                h = mx.maximum(conv(h), 0.0)
            projections.append(projection(h))
        return projections


class HED:
    def __init__(self):
        import torch  # weights only; no torch forward pass runs
        from huggingface_hub import hf_hub_download

        raw = torch.load(hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE), map_location="cpu", weights_only=True)
        state = {k: mx.array(v.float().numpy()) for k, v in raw.items()}
        self._net = _HEDNet(state)

    def edge_map(self, image: PIL.Image.Image, max_resolution: int = 512) -> PIL.Image.Image:
        rgb = image.convert("RGB")
        out_width, out_height = rgb.size

        # Run the VGG blocks at a bounded resolution (edges are low-frequency, and a render can be far
        # larger than HED needs), then scale the edge map back to the requested size. Both axes are
        # clamped to >= 16 so a degenerate aspect ratio (e.g. 512x1) does not collapse to zero after
        # the four 2x2 max-pools.
        scale = min(1.0, max_resolution / max(out_width, out_height))
        work_width = max(16, round(out_width * scale))
        work_height = max(16, round(out_height * scale))
        work = (
            rgb
            if (work_width, work_height) == (out_width, out_height)
            else rgb.resize((work_width, work_height), PIL.Image.LANCZOS)
        )

        x = mx.array(np.array(work, dtype=np.float32)[None])  # 1, H, W, 3 in 0-255
        projections = self._net(x)
        mx.eval(projections)

        edge_img = HED._fuse_side_outputs(projections, (work_width, work_height))
        if (work_width, work_height) != (out_width, out_height):
            edge_img = edge_img.resize((out_width, out_height), PIL.Image.BILINEAR)
        return edge_img

    @staticmethod
    def _fuse_side_outputs(projections: list[mx.array], size: tuple[int, int]) -> PIL.Image.Image:
        # Resize each 1-channel side output to `size`, average, sigmoid, and return an 8-bit RGB edge.
        width, height = size
        edges = []
        for p in projections:
            side = np.array(p)[0, :, :, 0]
            edges.append(np.array(PIL.Image.fromarray(side).resize((width, height), PIL.Image.BILINEAR)))
        edge = 1.0 / (1.0 + np.exp(-np.mean(np.stack(edges), axis=0)))
        edge_u8 = (edge * 255.0).clip(0, 255).astype(np.uint8)
        return PIL.Image.fromarray(np.repeat(edge_u8[:, :, None], 3, axis=2))
