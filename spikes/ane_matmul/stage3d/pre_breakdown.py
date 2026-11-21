"""Pre-MLP breakdown of one main layer (real weights, S=4224, bf16). Diagnostic, not a timing claim.

Each piece runs uncompiled on pre-evaluated inputs with mx.eval after it (so MLX cannot fuse across pieces),
median of N. The compiled whole pre graph (what arm C runs) is timed too, for reference: the sum of pieces
is larger than it because nothing fuses.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlx.core as mx  # noqa: E402
from eager_bench import load_real_blocks, make_inputs, make_pre  # noqa: E402
from mlp_program import plan_split  # noqa: E402

from mflux.models.z_image.model.z_image_transformer.attention import ZImageAttention  # noqa: E402

N = 9
S, D, H, HD = 4224, 3840, 30, 128
_, blocks = load_real_blocks(0, 1)
b = blocks[0]
x, mask, freqs, t_emb = make_inputs(S, D)
a = b.attention
m = plan_split(S, 0.5)[0]


def med(fn, *args):
    out = fn(*args)
    mx.eval(out)  # warm-up
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        out = fn(*args)
        mx.eval(out)
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts)), out


rows = []


def piece(name, flops, fn, *args):
    ms, out = med(fn, *args)
    rows.append(
        {"piece": name, "ms": ms, "gflop": flops / 1e9, "tflops": (flops / (ms / 1e3) / 1e12) if flops else 0.0}
    )
    return out


mm = 2 * S * D * D  # one 3840x3840 projection


# 1 adaLN modulation + norm1 + scale
def adaln(x, t_emb):
    mod = mx.expand_dims(b.adaLN_modulation[0](t_emb), axis=1)
    sm, gm, sl, gl = mx.split(mod, 4, axis=2)
    return b.attention_norm1(x) * (1.0 + sm)


hn = piece("adaLN modulation + attention_norm1 * scale", 0, adaln, x, t_emb)
q = piece("Q projection", mm, lambda h: a.to_q(h), hn)
k = piece("K projection", mm, lambda h: a.to_k(h), hn)
v = piece("V projection", mm, lambda h: a.to_v(h), hn)
qh = q.reshape(1, S, H, HD)
kh = k.reshape(1, S, H, HD)
qn = piece("QK-RMSNorm (q and k)", 0, lambda q, k: (a.norm_q(q), a.norm_k(k)), qh, kh)
qr = piece(
    "RoPE (q and k, composed ops)",
    0,
    lambda q, k: (ZImageAttention.apply_rotary_emb(q, freqs), ZImageAttention.apply_rotary_emb(k, freqs)),
    *qn,
)
from mflux.models.qwen21.model.qwen21_transformer.qwen21_fused_kernels import fused_qk_norm_rope  # noqa: E402

fused = piece(
    "FUSED QK-norm + RoPE + layout (replaces the two rows above)",
    0,
    lambda q, k: fused_qk_norm_rope(
        q, k, a.norm_q.weight, a.norm_k.weight, freqs[..., 0], freqs[..., 1], H, HD, eps=a.eps
    ),
    q,
    k,
)
qt, kt, vt = (mx.transpose(z, axes=(0, 2, 1, 3)) for z in (qr[0], qr[1], v.reshape(1, S, H, HD)))
mx.eval(qt, kt, vt)
am = mx.where(mask[:, None, None, :], mx.array(0.0, dtype=qt.dtype), mx.array(float("-inf"), dtype=qt.dtype))
attn_flops = 4 * S * S * HD * H  # QK^T and PV
o = piece(
    "SDPA attention",
    attn_flops,
    lambda q, k, v: mx.fast.scaled_dot_product_attention(q, k, v, scale=a.scale, mask=am),
    qt,
    kt,
    vt,
)
oh = mx.transpose(o, axes=(0, 2, 1, 3)).reshape(1, S, D)
mx.eval(oh)
ao = piece("o-projection (to_out)", mm, lambda h: a.to_out[0](h), oh)


def tail(x, ao, t_emb):
    mod = mx.expand_dims(b.adaLN_modulation[0](t_emb), axis=1)
    sm, gm, sl, gl = mx.split(mod, 4, axis=2)
    x1 = x + mx.tanh(gm) * b.attention_norm2(ao)
    h = b.ffn_norm1(x1) * (1.0 + sl)
    return x1, h[:, :m].astype(mx.float16), h[:, m:], mx.tanh(gl)


piece("attention_norm2 + gate + residual + ffn_norm1 * scale_mlp + row split casts", 0, tail, x, ao, t_emb)
import os  # noqa: E402

os.environ["MFLUX_ZIMAGE_DISABLE_FUSED_QKNORM_ROPE"] = "1"
pre_old = make_pre(b, m)
ms_pre_old, _ = med(lambda x, mk, f, t: pre_old(x, mk, f, t), x, mask, freqs, t_emb)
del os.environ["MFLUX_ZIMAGE_DISABLE_FUSED_QKNORM_ROPE"]
pre = make_pre(b, m)
ms_pre, _ = med(lambda x, mk, f, t: pre(x, mk, f, t), x, mask, freqs, t_emb)
print("compiled pre graph: composed %.2f ms | fused %.2f ms" % (ms_pre_old, ms_pre))
total = sum(r["ms"] for r in rows if not r["piece"].startswith("FUSED"))
for r in rows:
    r["pct_of_sum"] = 100 * r["ms"] / total
    r["pct_of_compiled_pre"] = 100 * r["ms"] / ms_pre
print(json.dumps({"rows": rows, "sum_pieces_ms": total, "compiled_pre_ms": ms_pre}, indent=1))
print("%-72s %8s %7s %9s %8s" % ("piece", "ms", "%sum", "GFLOP", "TFLOP/s"))
for r in rows:
    print("%-72s %8.2f %6.1f%% %9.1f %8.2f" % (r["piece"], r["ms"], r["pct_of_sum"], r["gflop"], r["tflops"]))
print("sum of pieces %.2f ms | compiled whole pre graph %.2f ms" % (total, ms_pre))
