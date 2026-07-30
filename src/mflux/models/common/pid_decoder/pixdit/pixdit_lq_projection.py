import mlx.core as mx
from mlx import nn

from mflux.models.common.pid_decoder.pixdit.pixdit_gates import build_gate

_VALID_CONV_PADDING_MODES = {"zeros", "replicate"}


def _pad_conv_input(x: mx.array, conv_padding_mode: str) -> mx.array:
    """Pre-pad for a 3x3 conv when conv_padding_mode != 'zeros' (Conv2d itself does the
    zero-padding for 'zeros'; MLX's Conv2d has no built-in padding_mode option)."""
    if conv_padding_mode == "zeros":
        return x
    if conv_padding_mode == "replicate":
        # NHWC: pad the H, W axes (1, 2) only.
        return mx.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)], mode="edge")
    raise ValueError(f"conv_padding_mode must be one of {sorted(_VALID_CONV_PADDING_MODES)}, got {conv_padding_mode!r}")


class ResBlock(nn.Module):
    def __init__(self, channels: int, num_groups: int = 4, conv_padding_mode: str = "zeros"):
        super().__init__()
        self.conv_padding_mode = conv_padding_mode
        self.norm1 = nn.GroupNorm(num_groups, channels, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=0 if conv_padding_mode != "zeros" else 1)
        self.norm2 = nn.GroupNorm(num_groups, channels, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=0 if conv_padding_mode != "zeros" else 1)

    def __call__(self, x: mx.array) -> mx.array:
        # MLX Conv2d is NHWC; caller is responsible for that layout (see LQProjection2D).
        h = _pad_conv_input(nn.silu(self.norm1(x)), self.conv_padding_mode)
        h = self.conv1(h)
        h = _pad_conv_input(nn.silu(self.norm2(h)), self.conv_padding_mode)
        h = self.conv2(h)
        return x + h


class LQProjection2D(nn.Module):
    """Latent-only LQ projection (Phase 1: lq_in_channels=0, image branch omitted).

    Source: pid/_src/networks/lq_projection_2d.py (latent branch only).
    """

    def __init__(
        self,
        latent_channels: int,
        hidden_dim: int,
        out_dim: int,
        patch_size: int,
        sr_scale: int,
        latent_spatial_down_factor: int,
        num_res_blocks: int,
        num_outputs: int,
        gate_type: str,
        interval: int,
        conv_padding_mode: str = "zeros",
        pit_output: bool = False,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.num_outputs = num_outputs
        self.interval = interval
        self.conv_padding_mode = conv_padding_mode

        z_to_patch_ratio = (sr_scale * latent_spatial_down_factor) / patch_size
        self.z_to_patch_ratio = z_to_patch_ratio
        if z_to_patch_ratio > 1:
            self.latent_upsample_ratio = int(z_to_patch_ratio)
            latent_proj_in_ch = latent_channels
        elif z_to_patch_ratio == 1:
            latent_proj_in_ch = latent_channels
        else:
            self.latent_fold_factor = int(1 / z_to_patch_ratio)
            assert self.latent_fold_factor * z_to_patch_ratio == 1.0, (
                f"fold_factor {self.latent_fold_factor} * z_to_patch_ratio {z_to_patch_ratio} != 1"
            )
            latent_proj_in_ch = latent_channels * self.latent_fold_factor**2

        conv1_padding = 0 if conv_padding_mode != "zeros" else 1
        self.latent_conv1 = nn.Conv2d(latent_proj_in_ch, hidden_dim, kernel_size=3, padding=conv1_padding)
        self.latent_conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=conv1_padding)
        self.latent_res_blocks = [
            ResBlock(hidden_dim, conv_padding_mode=conv_padding_mode) for _ in range(num_res_blocks)
        ]

        self.output_heads = [nn.Linear(hidden_dim, out_dim) for _ in range(num_outputs)]
        self.gate_modules = [build_gate(gate_type, out_dim) for _ in range(num_outputs)]
        self.pit_head = nn.Linear(hidden_dim, out_dim) if pit_output else None

    def is_gate_active(self, block_idx: int) -> bool:
        return block_idx % self.interval == 0 if self.interval > 1 else True

    def output_index(self, block_idx: int) -> int:
        return block_idx // self.interval if self.interval > 1 else block_idx

    def gate(self, x: mx.array, lq: mx.array, sigma: mx.array, out_idx: int) -> mx.array:
        return self.gate_modules[out_idx](x, lq, sigma)

    def _align_latent_to_patch_grid(self, lq_latent_nhwc: mx.array, pH: int, pW: int) -> mx.array:
        if self.z_to_patch_ratio >= 1:
            # Latent at or below the patch grid: nearest upsample to exactly (pH, pW), by the
            # same index map torch's F.interpolate(mode="nearest") uses, out[i] = in[i*z//p].
            # Repeat-by-integer-ratio-then-crop matches that only when the grid is an exact
            # multiple of the latent; when it is not, the crop keeps the leading tiles and
            # drops the tail, so the LQ conditioning drifts against the image tokens
            # progressively across the frame. mflux rounds to multiples of 16 (config.py:52),
            # which makes the wired path exact -- this keeps the off-grid caller honest too.
            zH, zW = lq_latent_nhwc.shape[1], lq_latent_nhwc.shape[2]
            rows = mx.arange(pH) * zH // pH
            cols = mx.arange(pW) * zW // pW
            return mx.take(mx.take(lq_latent_nhwc, rows, axis=1), cols, axis=2)
        # Latent higher-res than patch grid: fold f×f spatial blocks into channels.
        # Channel packing order must match PyTorch's z_aligned.reshape(B, z_dim, pH, f, pW, f)
        # .permute(0, 1, 3, 5, 2, 4).reshape(B, z_dim*f*f, pH, pW) -> packed axis order is
        # (C, fH, fW) with C slowest. So here we must move C *before* (fH, fW), not leave it
        # trailing, or the folded channels land in the wrong order relative to the loaded
        # latent_conv1 weights.
        f = self.latent_fold_factor
        B, H, W, C = lq_latent_nhwc.shape
        z = lq_latent_nhwc.reshape(B, pH, f, pW, f, C)
        z = z.transpose(0, 1, 3, 5, 2, 4)  # [B, pH, pW, C, fH, fW]
        return z.reshape(B, pH, pW, C * f * f)

    def __call__(self, lq_latent: mx.array, target_pH: int, target_pW: int) -> list[mx.array]:
        # lq_latent: [B, C, zH, zW] (mflux's channel-first convention) -> NHWC for MLX conv.
        lq_latent_nhwc = lq_latent.transpose(0, 2, 3, 1)
        z_aligned = self._align_latent_to_patch_grid(lq_latent_nhwc, target_pH, target_pW)

        h = _pad_conv_input(z_aligned, self.conv_padding_mode)
        h = self.latent_conv1(h)
        h = nn.silu(h)
        h = _pad_conv_input(h, self.conv_padding_mode)
        h = self.latent_conv2(h)
        for block in self.latent_res_blocks:
            h = block(h)

        tokens = h.reshape(h.shape[0], target_pH * target_pW, -1)  # [B, N, hidden_dim]
        outputs = [head(tokens) for head in self.output_heads]
        if self.pit_head is not None:
            outputs.append(self.pit_head(tokens))
        return outputs
