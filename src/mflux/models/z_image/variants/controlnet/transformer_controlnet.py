from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx import nn

try:
    from httpx import ProxyError as _ProxyError
except ImportError:  # httpx rides in with huggingface_hub 1.x; degrade gracefully without it

    class _ProxyError(Exception):
        pass


from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.model.z_image_transformer.transformer_block import ZImageTransformerBlock


@dataclass(frozen=True)
class ZImageControlNetConfig:
    # Which base transformer layer indices receive residuals (len == num_control_layers)
    control_layers_places: list[int]
    # Which noise_refiner layer indices receive residuals (len == num_control_refiner_layers)
    control_refiner_layers_places: list[int]
    # Control input channels (33 for Union 2.1 to support inpainting layout)
    control_in_dim: int = 33
    # One of: "control_layers", "control_noise_refiner", or None
    add_control_noise_refiner: str | None = "control_noise_refiner"
    # Patch params (Z-Image Turbo uses 2-1)
    patch_size: int = 2
    f_patch_size: int = 1
    # Architecture
    dim: int = 3840
    n_refiner_layers: int = 2
    n_heads: int = 30
    norm_eps: float = 1e-5
    qk_norm: bool = True

    @staticmethod
    def defaults_union_2_1() -> "ZImageControlNetConfig":
        # The model card states: 15 control layer blocks + 2 refiner layer blocks.
        # The 15 control blocks feed residuals into the 30-block base transformer strided every
        # other block ([0, 2, ..., 28]), matching the VideoX-Fun ceil(num_layers/num_control) = 2
        # convention. A contiguous [0..14] mapping lands the residuals on the wrong base blocks:
        # the control is ignored at low strength and diverges to noise at high strength.
        # We still prefer reading exact indices from config.json when available.
        return ZImageControlNetConfig(
            control_layers_places=list(range(0, 30, 2)),
            control_refiner_layers_places=list(range(2)),
            control_in_dim=33,
            add_control_noise_refiner="control_noise_refiner",
            patch_size=2,
            f_patch_size=1,
        )

    @staticmethod
    def from_pretrained(repo_id: str) -> "ZImageControlNetConfig":
        try:
            local = Path(repo_id)
            # A saved model passes a local directory; read its config.json without touching the hub.
            root = local if local.is_dir() else Path(snapshot_download(repo_id=repo_id, allow_patterns=["config.json"]))
            cfg_path = root / "config.json"
            if not cfg_path.exists():
                return ZImageControlNetConfig.defaults_union_2_1()
            cfg = json.loads(cfg_path.read_text())
        except (OSError, ValueError, RuntimeError, _ProxyError):
            return ZImageControlNetConfig.defaults_union_2_1()

        # Fall back to the documented-correct Union 2.1 values (strided placement, control_in_dim 33)
        # for any key the config.json omits, rather than to plausible-but-wrong literals.
        defaults = ZImageControlNetConfig.defaults_union_2_1()

        # A non-object root, or a malformed value (e.g. control_in_dim: "invalid"), must fall back to
        # the defaults rather than crash model initialization, so the whole read is guarded too.
        if not isinstance(cfg, dict):
            return defaults

        def _get_list(key: str, default: list[int]) -> list[int]:
            # Placement entries index base transformer blocks, so a list with any non-int entry
            # (or an empty list) falls back to the default rather than producing residual keys that
            # silently never match a block index.
            val = cfg.get(key, None)
            if isinstance(val, list) and len(val) > 0 and all(isinstance(i, int) for i in val):
                return list(val)
            return default

        try:
            return ZImageControlNetConfig(
                control_layers_places=_get_list("control_layers_places", defaults.control_layers_places),
                control_refiner_layers_places=_get_list(
                    "control_refiner_layers_places", defaults.control_refiner_layers_places
                ),
                control_in_dim=int(cfg.get("control_in_dim", defaults.control_in_dim)),
                add_control_noise_refiner=cfg.get("add_control_noise_refiner", "control_noise_refiner"),
                patch_size=int(cfg.get("all_patch_size", [2])[0] if isinstance(cfg.get("all_patch_size"), list) else 2),
                f_patch_size=int(
                    cfg.get("all_f_patch_size", [1])[0] if isinstance(cfg.get("all_f_patch_size"), list) else 1
                ),
                dim=int(cfg.get("dim", 3840)),
                n_refiner_layers=int(cfg.get("n_refiner_layers", 2)),
                n_heads=int(cfg.get("n_heads", 30)),
                norm_eps=float(cfg.get("norm_eps", 1e-5)),
                qk_norm=bool(cfg.get("qk_norm", True)),
            )
        except (TypeError, ValueError):
            return defaults


class ZImageControlTransformerBlock(nn.Module):
    """
    Z-Image ControlNet block (MLX), conceptually matching diffusers' `ZImageControlTransformerBlock`:
    it consumes control context `c` and the base sequence `x`, and appends a residual hint each block.
    """

    def __init__(self, *, dim: int, n_heads: int, norm_eps: float, qk_norm: bool, block_id: int):
        super().__init__()
        from mflux.models.z_image.model.z_image_transformer.attention import ZImageAttention
        from mflux.models.z_image.model.z_image_transformer.feed_forward import FeedForward

        self.dim = dim
        self.block_id = block_id

        self.attention = ZImageAttention(dim=dim, n_heads=n_heads, qk_norm=qk_norm, eps=1e-5)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.adaLN_modulation = [nn.Linear(min(dim, 256), 4 * dim, bias=True)]

        # Control projections are zero-initialized so the ControlNet starts as a no-op.
        self.before_proj = nn.Linear(dim, dim).apply(nn.init.constant(0)) if block_id == 0 else None
        self.after_proj = nn.Linear(dim, dim).apply(nn.init.constant(0))

    @staticmethod
    def _unbind0(x: mx.array) -> list[mx.array]:
        # Unbind along axis 0 (like torch.unbind)
        return [mx.squeeze(s, axis=0) for s in mx.split(x, x.shape[0], axis=0)]

    def __call__(self, c: mx.array, x: mx.array, attn_mask: mx.array, freqs_cis: mx.array, t_emb: mx.array) -> mx.array:
        # Stack accumulator logic (matches diffusers semantics)
        if self.block_id == 0:
            if self.before_proj is None:
                raise ValueError("Expected before_proj for block_id==0")
            c = self.before_proj(c) + x
            all_c: list[mx.array] = []
        else:
            all_c = self._unbind0(c)
            c = all_c.pop(-1)

        # AdaLN modulation (global)
        modulation = mx.expand_dims(self.adaLN_modulation[0](t_emb), axis=1)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(modulation, 4, axis=2)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp
        gate_msa = mx.tanh(gate_msa)
        gate_mlp = mx.tanh(gate_mlp)

        # Attention
        attn_out = self.attention(self.attention_norm1(c) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis)
        c = c + gate_msa * self.attention_norm2(attn_out)

        # FFN
        ffn_out = self.feed_forward(self.ffn_norm1(c) * scale_mlp)
        c = c + gate_mlp * self.ffn_norm2(ffn_out)

        # Emit residual hint
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        return mx.stack(all_c, axis=0)


class ZImageControlNet(nn.Module):
    """
    Z-Image ControlNet implementation for MLX.\n
    Returns a dict mapping base transformer layer indices -> residual tensor to add at that layer.
    """

    def __init__(self, config: ZImageControlNetConfig):
        super().__init__()
        self.config = config

        key = f"{config.patch_size}-{config.f_patch_size}"
        control_patch_dim = config.f_patch_size * config.patch_size * config.patch_size * config.control_in_dim

        # Control patch embedding
        self.control_all_x_embedder = {key: nn.Linear(control_patch_dim, config.dim, bias=True)}

        # Control blocks (15 for Union)
        self.control_layers = [
            ZImageControlTransformerBlock(
                dim=config.dim,
                n_heads=config.n_heads,
                norm_eps=config.norm_eps,
                qk_norm=config.qk_norm,
                block_id=i,
            )
            for i in range(len(config.control_layers_places))
        ]

        # Noise-refiner control blocks (2 for Union)
        if config.add_control_noise_refiner == "control_layers":
            self.control_noise_refiner = None
        elif config.add_control_noise_refiner == "control_noise_refiner":
            self.control_noise_refiner = [
                ZImageControlTransformerBlock(
                    dim=config.dim,
                    n_heads=config.n_heads,
                    norm_eps=config.norm_eps,
                    qk_norm=config.qk_norm,
                    block_id=i,
                )
                for i in range(config.n_refiner_layers)
            ]
        elif config.add_control_noise_refiner is None:
            # Legacy / fallback: use standard transformer blocks
            self.control_noise_refiner = [
                ZImageTransformerBlock(config.dim, config.n_heads, config.norm_eps, config.qk_norm)
                for _ in range(config.n_refiner_layers)
            ]
        else:
            # Fail before any weights download; __call__ would reject this value anyway.
            raise ValueError(f"Unsupported add_control_noise_refiner={config.add_control_noise_refiner!r}")

        # Shared modules from base transformer (set via from_transformer)
        self.t_scale: Optional[float] = None
        self.t_embedder = None
        self.all_x_embedder = None
        self.cap_embedder = None
        self.rope_embedder = None
        self.noise_refiner = None
        self.context_refiner = None
        self.x_pad_token = None
        self.cap_pad_token = None

    @classmethod
    def from_transformer(cls, controlnet: "ZImageControlNet", transformer: ZImageTransformer) -> "ZImageControlNet":
        controlnet.t_scale = transformer.t_scale
        controlnet.t_embedder = transformer.t_embedder
        controlnet.all_x_embedder = transformer.all_x_embedder
        controlnet.cap_embedder = transformer.cap_embedder
        controlnet.rope_embedder = transformer.rope_embedder
        controlnet.noise_refiner = transformer.noise_refiner
        controlnet.context_refiner = transformer.context_refiner
        controlnet.x_pad_token = transformer.x_pad_token
        controlnet.cap_pad_token = transformer.cap_pad_token
        return controlnet

    @staticmethod
    def _patchify_control(
        *,
        image: mx.array,
        patch_size: int,
        f_patch_size: int,
    ) -> tuple[mx.array, mx.array]:
        """
        Returns (patches, pad_mask) where:
        - patches: (seq_padded, patch_dim)
        - pad_mask: (seq_padded,) bool
        """
        pH = pW = patch_size
        pF = f_patch_size
        C, F, H, W = image.shape
        F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

        patches = image.reshape(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        patches = mx.transpose(patches, axes=(1, 3, 5, 2, 4, 6, 0))
        patches = patches.reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

        ori_len = patches.shape[0]
        pad_len = (-ori_len) % 32
        if pad_len > 0:
            patches = mx.concatenate([patches, mx.repeat(patches[-1:], pad_len, axis=0)], axis=0)
        pad_mask = mx.concatenate([mx.zeros((ori_len,), dtype=mx.bool_), mx.ones((pad_len,), dtype=mx.bool_)])
        return patches, pad_mask

    @staticmethod
    def _dict_add_inplace(dst: dict[int, mx.array], src: dict[int, mx.array]) -> dict[int, mx.array]:
        for k, v in src.items():
            dst[k] = dst[k] + v if k in dst else v
        return dst

    def __call__(
        self,
        *,
        x: mx.array,
        t: int,
        sigmas: mx.array,
        cap_feats: mx.array,
        control_context: mx.array,
        conditioning_scale: float = 1.0,
    ) -> dict[int, mx.array]:
        if (
            self.t_scale is None
            or self.t_embedder is None
            or self.all_x_embedder is None
            or self.cap_embedder is None
            or self.rope_embedder is None
            or self.noise_refiner is None
            or self.context_refiner is None
            or self.x_pad_token is None
            or self.cap_pad_token is None
        ):
            raise ValueError("ControlNet is missing shared modules. Call `from_transformer(controlnet, transformer)`.")

        key = f"{self.config.patch_size}-{self.config.f_patch_size}"

        # Time embedding (match ZImageTransformer)
        t_value = mx.array([1.0 - sigmas[t].item()])
        t_emb = self.t_embedder(t_value * self.t_scale)

        # Patchify image + caption (reuse transformer helper)
        x_patches, cap_padded, _x_size, x_pos_ids, cap_pos_ids, x_pad_mask, cap_pad_mask = ZImageTransformer._patchify(  # noqa: SLF001
            image=x,
            cap_feats=cap_feats,
            patch_size=self.config.patch_size,
            f_patch_size=self.config.f_patch_size,
        )

        # Embed x
        x_emb = self.all_x_embedder[key](x_patches)
        x_emb = mx.where(x_pad_mask[:, None], self.x_pad_token, x_emb)
        x_freqs_cis = self.rope_embedder(x_pos_ids)
        x_attn_mask = mx.ones((1, x_emb.shape[0]), dtype=mx.bool_)
        x_emb = mx.expand_dims(x_emb, axis=0)

        # Embed control context (patchify control latents)
        control_patches, control_pad_mask = self._patchify_control(
            image=control_context, patch_size=self.config.patch_size, f_patch_size=self.config.f_patch_size
        )
        control_emb = self.control_all_x_embedder[key](control_patches)
        # Prefer base x_pad_mask for consistency
        if control_emb.shape[0] == x_pad_mask.shape[0]:
            control_emb = mx.where(x_pad_mask[:, None], self.x_pad_token, control_emb)
        else:
            control_emb = mx.where(control_pad_mask[:, None], self.x_pad_token, control_emb)
        control_emb = mx.expand_dims(control_emb, axis=0)

        # Optional: generate residuals for noise_refiner
        noise_refiner_block_samples: dict[int, mx.array] | None = None
        if self.config.add_control_noise_refiner is not None:
            if self.config.add_control_noise_refiner == "control_layers":
                layers = self.control_layers
            elif self.config.add_control_noise_refiner == "control_noise_refiner":
                if self.control_noise_refiner is None:
                    raise ValueError(
                        "Expected control_noise_refiner for add_control_noise_refiner=control_noise_refiner"
                    )
                layers = self.control_noise_refiner
            else:
                raise ValueError(f"Unsupported add_control_noise_refiner={self.config.add_control_noise_refiner!r}")

            c = control_emb
            for layer in layers:
                if isinstance(layer, ZImageControlTransformerBlock):
                    c = layer(c, x_emb, x_attn_mask, x_freqs_cis, t_emb)
                else:
                    # Fallback path: transformer-style block on control tokens
                    c = layer(c, x_attn_mask, x_freqs_cis, t_emb)  # type: ignore[misc]

            hints = ZImageControlTransformerBlock._unbind0(c)[:-1]
            noise_refiner_block_samples = {
                layer_idx: hints[idx] * conditioning_scale
                for idx, layer_idx in enumerate(self.config.control_refiner_layers_places)
                if idx < len(hints)
            }
            # The refiner also transforms the control tokens themselves; carry the refined control
            # forward into the main control layers (the reference reassigns control_context here).
            # Without this the main layers see the raw, unrefined embedding and the residuals drift.
            control_emb = ZImageControlTransformerBlock._unbind0(c)[-1]

        # Noise refiner (shared), with optional injected residuals
        for layer_idx, layer in enumerate(self.noise_refiner):
            x_emb = layer(x=x_emb, attn_mask=x_attn_mask, freqs_cis=x_freqs_cis, t_emb=t_emb)
            if noise_refiner_block_samples is not None and layer_idx in noise_refiner_block_samples:
                x_emb = x_emb + noise_refiner_block_samples[layer_idx]

        # Caption embedding + refiner (shared)
        cap_emb = self.cap_embedder[1](self.cap_embedder[0](cap_padded))
        cap_emb = mx.where(cap_pad_mask[:, None], self.cap_pad_token, cap_emb)
        cap_freqs_cis = self.rope_embedder(cap_pos_ids)
        cap_attn_mask = mx.ones((1, cap_emb.shape[0]), dtype=mx.bool_)
        cap_emb = mx.expand_dims(cap_emb, axis=0)
        for layer in self.context_refiner:
            cap_emb = layer(x=cap_emb, attn_mask=cap_attn_mask, freqs_cis=cap_freqs_cis)

        # If no special noise-refiner control path, refine control tokens with standard blocks
        if self.config.add_control_noise_refiner is None:
            if self.control_noise_refiner is None:
                raise ValueError("Expected control_noise_refiner when add_control_noise_refiner is None")
            for layer in self.control_noise_refiner:
                if isinstance(layer, ZImageTransformerBlock):
                    control_emb = layer(x=control_emb, attn_mask=x_attn_mask, freqs_cis=x_freqs_cis, t_emb=t_emb)
                else:
                    control_emb = layer(control_emb, x_emb, x_attn_mask, x_freqs_cis, t_emb)  # type: ignore[misc]

        # Unified sequences (match base transformer ordering: [x, cap])
        unified = mx.concatenate([x_emb, cap_emb], axis=1)
        unified_freqs_cis = mx.concatenate([x_freqs_cis, cap_freqs_cis], axis=0)
        unified_attn_mask = mx.ones((1, unified.shape[1]), dtype=mx.bool_)

        # Control unified: [control(x_tokens), cap_tokens]
        control_unified = mx.concatenate([control_emb, cap_emb], axis=1)

        # Main control layers produce per-layer hints
        c = control_unified
        for layer in self.control_layers:
            c = layer(c, unified, unified_attn_mask, unified_freqs_cis, t_emb)

        hints = ZImageControlTransformerBlock._unbind0(c)[:-1]
        controlnet_block_samples = {
            layer_idx: hints[idx] * conditioning_scale
            for idx, layer_idx in enumerate(self.config.control_layers_places)
            if idx < len(hints)
        }
        return controlnet_block_samples
