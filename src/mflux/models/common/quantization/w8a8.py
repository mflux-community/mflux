"""W8A8 linear layers for ComfyUI "int8_convrot" checkpoints, on the Metal TensorOps INT8 matmul (Apple M5).

A convrot checkpoint stores each projection rotated by a block-diagonal Hadamard matrix (256-wide groups along
the input dimension) and quantized to int8 with one float32 scale per output row. At run time the activations
get the same rotation and a per-row int8 quantization, so the rotations cancel in the product:
y = (q(x H) q(W H)^T) * x_scale * w_scale ~ x W^T. The rotation spreads outliers across each group, which is
what keeps int8 activations accurate.

The GEMM kernel is mlx-serve's (David Dalcu, MIT, see w8a8_kernels/LICENSE), as used by h3-image-mlx. GPUs
without TensorOps (before Apple M5) run the same layers through a float fallback: the activations are rotated
and multiplied by the dequantized int8 weights, which gives the same result up to the activation rounding.
"""

import re
from functools import cache
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn

KERNELS = Path(__file__).resolve().parent / "w8a8_kernels"
GROUP = 256

_H4 = np.ones((4, 4), dtype=np.float32)
np.fill_diagonal(_H4[:, ::-1], -1)
H256 = mx.array(np.kron(np.kron(_H4, _H4), np.kron(_H4, _H4)) / 16)


@cache
def tensor_ops_available() -> bool:
    """The INT8 TensorOps GEMM needs an Apple M5 (or later) GPU."""
    name = str(mx.device_info().get("device_name", ""))
    return bool(re.search(r"\bApple M([5-9]|\d{2})\b", name))


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="msv_w8a8_gemm",
        input_names=["xq", "wq", "xs", "ws"],
        output_names=["y"],
        header=(KERNELS / "mpp_tensor_ops.metal").read_text(),
        source=(KERNELS / "w8a8_gemm_body.metal").read_text() + "\n" + (KERNELS / "w8a8_epi_plain.metal").read_text(),
    )


def rotate(x: mx.array) -> mx.array:
    """x H256 (block-diagonal along the last axis) in float32."""
    return (x.astype(mx.float32).reshape(-1, GROUP) @ H256).reshape(x.shape)


def rotate_quantize_reference(x: mx.array) -> tuple[mx.array, mx.array]:
    rotated = rotate(x)
    scale = mx.maximum(mx.max(mx.abs(rotated), axis=-1) / 127, 1e-20)
    return mx.clip(mx.round(rotated / scale[:, None]), -127, 127).astype(mx.int8), scale


# One threadgroup per row, one thread per position in a 256-wide group. H256 = H4 (x) H4 (x) H4 (x) H4 with
# H4 = ones - 2 * antidiagonal, so each of the four base-4 digits is a 4-point butterfly
# y_k = (x_0 + x_1 + x_2 + x_3) - 2 x_(3-k). The row is read once, kept in registers, and written once as int8:
# a memory-bound pass instead of a 256x256 float32 matmul per group (7.2 -> ~2 ms for 16384x12288 on M5 Ultra).
_ROTATE_QUANTIZE = """
    const uint t = thread_position_in_threadgroup.x;
    const ushort lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.y;
    const uint K = x_shape[1];
    const uint groups = K / 256;
    threadgroup float buf[256];
    threadgroup float red[8];
    float v[MAX_GROUPS];
    float amax = 0.0f;
    device const T* xr = x + (size_t)row * K;
    for (uint g = 0; g < groups; ++g) {
        float a = float(xr[g * 256 + t]);
        for (uint s = 1; s < 256; s *= 4) {
            buf[t] = a;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint k = (t / s) % 4, base = t - k * s;
            a = buf[base] + buf[base + s] + buf[base + 2 * s] + buf[base + 3 * s] - 2.0f * buf[base + (3 - k) * s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        a *= 0.0625f;
        v[g] = a;
        amax = max(amax, fabs(a));
    }
    amax = simd_max(amax);
    if (lane == 0) red[simdgroup_index_in_threadgroup] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = red[0];
    for (uint i = 1; i < 8; ++i) amax = max(amax, red[i]);
    const float scale = max(amax / 127.0f, 1e-20f);
    device int8_t* qr = xq + (size_t)row * K;
    for (uint g = 0; g < groups; ++g) qr[g * 256 + t] = int8_t(clamp(rint(v[g] / scale), -127.0f, 127.0f));
    if (t == 0) xs[row] = scale;
"""


@cache
def _rotate_quantize_kernel():
    return mx.fast.metal_kernel(
        name="convrot_rotate_quantize",
        input_names=["x"],
        output_names=["xq", "xs"],
        source=_ROTATE_QUANTIZE,
    )


def rotate_quantize(x: mx.array, pad_rows: int = 1) -> tuple[mx.array, mx.array]:
    """Per-row int8 quantization of x H256 (block-diagonal along the last axis) with float32 scales.

    The int8 rows are allocated up to a multiple of pad_rows (the GEMM tile height) so that the GEMM never
    reads past them; the extra rows are not written and only feed output rows that are never stored."""
    rows, k = x.shape
    groups = k // GROUP
    xq, xs = _rotate_quantize_kernel()(
        inputs=[x],
        template=[("T", x.dtype), ("MAX_GROUPS", 1 << (groups - 1).bit_length())],
        grid=(GROUP, rows, 1),
        threadgroup=(GROUP, 1, 1),
        output_shapes=[(-(-rows // pad_rows) * pad_rows, k), (rows,)],
        output_dtypes=[mx.int8, mx.float32],
    )
    return xq, xs


def matmul(x: mx.array, weight: mx.array, scale: mx.array) -> mx.array:
    """x (rows, in) in any float dtype @ an int8 convrot weight (out, in) with float32 scales (out,)."""
    if x.shape[1] != weight.shape[1]:
        # The kernel takes its K from x, so another width would read the weight rows with the wrong stride.
        raise ValueError(f"W8A8 input width {x.shape[1]} does not match the weight {tuple(weight.shape)}.")
    rows, n = x.shape[0], weight.shape[0]
    bm = 128 if rows < 8192 else 64
    bn, bk = 8192 // bm, 64
    if not tensor_ops_available() or n < bn:
        # Rotated activations times the dequantized weight: (x H)(q(W H) s)^T ~ x W^T. Also used for outputs
        # narrower than one column tile, which the kernel cannot place inside the weight.
        w = weight.astype(mx.float32) * scale[:, None]
        return (rotate(x) @ w.T).astype(x.dtype)
    xq, xs = rotate_quantize(x, pad_rows=bm)
    (y,) = _kernel()(
        inputs=[xq, weight, xs, scale],
        template=[("BM", bm), ("BN", bn), ("BK", bk), ("T", x.dtype)],
        grid=((n + bn - 1) // bn * 128, (rows + bm - 1) // bm, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )
    return y


class W8A8Linear(nn.Module):
    """Drop-in for a bias-free nn.Linear holding an int8 convrot weight and its per-row scales."""

    def __init__(self, input_dims: int, output_dims: int):
        super().__init__()
        if input_dims % GROUP:
            raise ValueError(f"W8A8 needs the input width to be a multiple of {GROUP}, got {input_dims}.")
        self.weight = mx.zeros((output_dims, input_dims), dtype=mx.int8)
        self.scale = mx.zeros((output_dims,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        y = matmul(x.reshape(-1, shape[-1]), self.weight, self.scale)
        return y.reshape(*shape[:-1], self.weight.shape[0])

    @staticmethod
    def fused(x: mx.array, packed: tuple[mx.array, mx.array, list[int]]) -> list[mx.array]:
        """Several projections of the same input (packed by pack()) with one activation quantization and one GEMM."""
        weight, scale, sizes = packed
        shape = x.shape
        y = matmul(x.reshape(-1, shape[-1]), weight, scale).reshape(*shape[:-1], weight.shape[0])
        return mx.split(y, [sum(sizes[: i + 1]) for i in range(len(sizes) - 1)], axis=-1)

    @staticmethod
    def pack(layers: tuple["W8A8Linear", ...]) -> tuple[mx.array, mx.array, list[int]]:
        """Concatenate the rows of layers that share an input, then point each layer at its slice of the result."""
        weight = mx.concatenate([layer.weight for layer in layers], axis=0)
        scale = mx.concatenate([layer.scale for layer in layers])
        mx.eval(weight, scale)
        sizes, start = [], 0
        for layer in layers:
            rows = layer.weight.shape[0]
            layer.weight, layer.scale = weight[start : start + rows], scale[start : start + rows]
            sizes.append(rows)
            start += rows
        return weight, scale, sizes

    @staticmethod
    def replace(module: nn.Module, name: str) -> "W8A8Linear":
        linear = getattr(module, name) if not name.isdigit() else module[int(name)]
        w8a8 = W8A8Linear(linear.weight.shape[1], linear.weight.shape[0])
        if name.isdigit():
            module[int(name)] = w8a8
        else:
            setattr(module, name, w8a8)
        return w8a8
