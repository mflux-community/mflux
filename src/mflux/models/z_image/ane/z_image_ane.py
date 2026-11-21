"""--ane integration for the Z-Image DiT: split 32 transformer blocks onto ANE + GPU.

Covers the four fused call sites per block (qkv, o, w1w3, w2 K-split 4) in the
30 main blocks and the 2 noise-refiner blocks. The context refiner, adaLN, the
embedders and the final layer stay on the GPU. At load each split layer's
output columns [0:oa) are baked into a cached Core ML package (built once,
content-hashed) and dropped from MLX; the GPU keeps columns [oa:O).

Scope guards for v1 (enforced by AneGate): bfloat16 stream only, dense bf16 or
-q 8 sources, macOS 26+ on M3/M4 (M5 needs --ane-allow-m5), Z-Image only.
"""

import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mflux.models.z_image.ane.ane_extension import AneExtension
from mflux.models.z_image.ane.ane_gate import AneGate
from mflux.models.z_image.ane.ane_layers import AneHandle, FusedSplit, SplitLinear
from mflux.models.z_image.ane.ane_model_builder import AneModelBuilder

W2_HEADROOM = 32.0  # FeedForward.FLOAT16_HEADROOM: keeps the ANE's fp16 input in range


class AneAttention(nn.Module):
    """Drop-in for ZImageAttention: fused QKV split + split output projection."""

    def __init__(self, attn: nn.Module, qkv: FusedSplit, out: SplitLinear):
        super().__init__()
        self.a = attn
        self.qkv = qkv
        self.out = out

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array | None = None, freqs_cis=None) -> mx.array:
        attn = self.a
        batch_size, seq_len, _ = hidden_states.shape
        query, key, value = self.qkv(hidden_states)
        query = query.reshape(batch_size, seq_len, attn.n_heads, attn.head_dim)
        key = key.reshape(batch_size, seq_len, attn.n_heads, attn.head_dim)
        value = value.reshape(batch_size, seq_len, attn.n_heads, attn.head_dim)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if freqs_cis is not None:
            query = attn.apply_rotary_emb(query, freqs_cis)
            key = attn.apply_rotary_emb(key, freqs_cis)
        query = mx.transpose(query, axes=(0, 2, 1, 3))
        key = mx.transpose(key, axes=(0, 2, 1, 3))
        value = mx.transpose(value, axes=(0, 2, 1, 3))
        mask = None
        if attention_mask is not None:
            mask = mx.where(
                attention_mask[:, None, None, :],
                mx.array(0.0, dtype=query.dtype),
                mx.array(float("-inf"), dtype=query.dtype),
            )
        from mlx.core.fast import scaled_dot_product_attention

        out = scaled_dot_product_attention(query, key, value, scale=attn.scale, mask=mask)
        out = mx.transpose(out, axes=(0, 2, 1, 3)).reshape(batch_size, seq_len, attn.dim)
        return self.out(out)


class AneFeedForward(nn.Module):
    """Drop-in for FeedForward: fused w1w3 split + split w2 (K-split 4, headroom 32)."""

    def __init__(self, w13: FusedSplit, w2: SplitLinear):
        super().__init__()
        self.w13 = w13
        self.w2 = w2

    def __call__(self, x: mx.array) -> mx.array:
        h1, h3 = self.w13(x)
        return self.w2(nn.silu(h1) * h3)


class ZImageAne:
    @staticmethod
    def default_cache_dir() -> Path:
        env = os.environ.get("MFLUX_ANE_CACHE")
        if env:
            return Path(env)
        return Path.home() / "Library" / "Caches" / "mflux" / "ane"

    @staticmethod
    def apply(
        transformer: nn.Module,
        cache_dir: Path | None = None,
        precision: str = "w8a16",
        gpu_share: float = 0.5,
        compiled: bool = True,
        tiers: list[int] | None = None,
        limit: int | None = None,
        refiner: bool = True,
    ) -> int:
        """Replace the split layers in place; returns the number of patched blocks.

        limit/refiner exist for the scaling bisection (how step cost grows with the
        number of live ANE models); production callers use the defaults.
        """
        ext = AneExtension.load()
        cache_dir = Path(cache_dir) if cache_dir else ZImageAne.default_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        tiers = tiers or AneModelBuilder.TIERS
        count = 0
        if refiner:
            for idx, block in enumerate(transformer.noise_refiner):
                ZImageAne._patch_block(ext, block, cache_dir, f"r{idx}", precision, gpu_share, compiled, tiers)
                count += 1
        blocks = transformer.layers if limit is None else transformer.layers[:limit]
        for idx, block in enumerate(blocks):
            ZImageAne._patch_block(ext, block, cache_dir, f"b{idx}", precision, gpu_share, compiled, tiers)
            count += 1
        AneGate.warn_if_background_qos()
        return count

    @staticmethod
    def _patch_block(ext, block, cache_dir: Path, tag: str, prec: str, share: float, compiled: bool, tiers):
        attn = block.attention
        dims = [attn.to_q.weight.shape[0], attn.to_k.weight.shape[0], attn.to_v.weight.shape[0]]
        qkv = ZImageAne._build_fused(
            ext, cache_dir, f"{tag}_qkv", [attn.to_q, attn.to_k, attn.to_v], dims, prec, share, tiers, compiled
        )
        out = ZImageAne._build_linear(
            ext, cache_dir, f"{tag}_o", attn.to_out[0], prec, share, tiers, compiled, ksplit=1
        )
        block.attention = AneAttention(attn, qkv, out)
        ff = block.feed_forward
        w13 = ZImageAne._build_fused(
            ext,
            cache_dir,
            f"{tag}_w13",
            [ff.w1, ff.w3],
            [ff.w1.weight.shape[0], ff.w3.weight.shape[0]],
            prec,
            share,
            tiers,
            compiled,
        )
        w2 = ZImageAne._build_linear(
            ext, cache_dir, f"{tag}_w2", ff.w2, prec, share, tiers, compiled, ksplit=4, headroom=W2_HEADROOM
        )
        block.feed_forward = AneFeedForward(w13, w2)

    @staticmethod
    def dequantize_on_load(transformer: nn.Module) -> int:
        """Replace every quantized linear in the transformer with its bfloat16 dense form."""
        replaced = 0
        for path, module in list(transformer.named_modules()):
            if not isinstance(module, nn.QuantizedLinear):
                continue
            dense = mx.dequantize(module.weight, module.scales, module.biases, module.group_size, module.bits)
            has_bias = getattr(module, "bias", None) is not None
            lin = nn.Linear(ZImageAne._in_dims(module), ZImageAne._out_dims(module), bias=has_bias)
            lin.weight = dense.astype(mx.bfloat16)
            if has_bias:
                lin.bias = module.bias.astype(mx.bfloat16)
            ZImageAne._replace(transformer, path, lin)
            replaced += 1
        return replaced

    @staticmethod
    def _replace(root: nn.Module, path: str, new: nn.Module) -> None:
        parts = path.split(".")
        obj = root
        for part in parts[:-1]:
            obj = obj[int(part)] if isinstance(obj, list) else getattr(obj, part)
        last = parts[-1]
        if isinstance(obj, list):
            obj[int(last)] = new
        else:
            setattr(obj, last, new)

    @staticmethod
    def _out_dims(module: nn.Module) -> int:
        return module.weight.shape[0]

    @staticmethod
    def _in_dims(module: nn.Module) -> int:
        if isinstance(module, nn.QuantizedLinear):
            return module.scales.shape[1] * module.group_size
        return module.weight.shape[1]

    @staticmethod
    def _build_fused(ext, cache_dir, tag, modules, dims, prec, share, tiers, compiled) -> FusedSplit:
        in_dim = ZImageAne._in_dims(modules[0])
        total = sum(dims)
        og = ZImageAne._round64(total * share)
        oa = total - og
        ane_counts, gpus = [], []
        off = 0
        prefix_rows = []
        for d, m in zip(dims, modules, strict=True):
            a = max(0, min(d, oa - off))
            ane_counts.append(a)
            prefix_rows.append(ZImageAne._prefix_np(m, a))
            gpus.append(ZImageAne._suffix(m, d, a) if d - a > 0 else None)
            off += d
        w_ane = np.concatenate(prefix_rows, axis=0)
        handle = ZImageAne._handle(ext, w_ane, prec, 1, tag, cache_dir, tiers)
        fused = FusedSplit(handle, dims, ane_counts, gpus, in_dim, prec, tiers)
        fused.drain = not compiled
        return fused

    @staticmethod
    def _build_linear(ext, cache_dir, tag, module, prec, share, tiers, compiled, ksplit=1, headroom=1.0) -> SplitLinear:
        total, in_dim = ZImageAne._out_dims(module), ZImageAne._in_dims(module)
        og = ZImageAne._round64(total * share)
        oa = total - og
        if oa <= 0:
            raise ValueError("gpu_share must leave columns for the ANE")
        w_pref = ZImageAne._prefix_np(module, oa)
        handle = ZImageAne._handle(ext, w_pref, prec, ksplit, tag, cache_dir, tiers)
        gpu = ZImageAne._suffix(module, total, oa) if og > 0 else None
        if gpu is None:  # SplitLinear expects a module for the shape; all-ANE keeps a zero-row stub
            gpu = ZImageAne._suffix(module, total, total)
        layer = SplitLinear(handle, gpu, oa, in_dim, prec, tiers, headroom=headroom)
        layer.drain = not compiled
        return layer

    @staticmethod
    def _handle(ext, w_prefix: np.ndarray, prec, ksplit, tag, cache_dir, tiers) -> AneHandle:
        path = AneModelBuilder.cache_path(w_prefix, prec, ksplit, tag, cache_dir, tiers)
        AneModelBuilder.build(w_prefix, prec, ksplit, path, tiers)
        model = ext.AneModel(str(path), max(tiers), w_prefix.shape[1], w_prefix.shape[0])
        return AneHandle(ext, model)

    @staticmethod
    def _prefix_np(module: nn.Module, rows: int) -> np.ndarray:
        if rows == 0:
            return np.zeros((0, ZImageAne._in_dims(module)), dtype=np.float32)
        if isinstance(module, nn.QuantizedLinear):
            dense = mx.dequantize(
                module.weight[:rows], module.scales[:rows], module.biases[:rows], module.group_size, module.bits
            )
        else:
            dense = module.weight[:rows]
        return np.array(dense.astype(mx.float32))

    @staticmethod
    def _suffix(module: nn.Module, total: int, keep_from: int) -> nn.Module:
        rows = total - keep_from
        has_bias = getattr(module, "bias", None) is not None
        if isinstance(module, nn.QuantizedLinear):
            sq = nn.QuantizedLinear(
                input_dims=ZImageAne._in_dims(module),
                output_dims=rows,
                bias=has_bias,
                group_size=module.group_size,
                bits=module.bits,
            )
            sq.weight = module.weight[keep_from:]
            sq.scales = module.scales[keep_from:]
            sq.biases = module.biases[keep_from:]
            if has_bias:
                sq.bias = module.bias[keep_from:]
            return sq
        sl = nn.Linear(ZImageAne._in_dims(module), rows, bias=has_bias)
        sl.weight = module.weight[keep_from:]
        if has_bias:
            sl.bias = module.bias[keep_from:]
        return sl

    @staticmethod
    def _round64(v: float) -> int:
        return max(0, min(int(round(v / 64)) * 64, 1 << 30))
