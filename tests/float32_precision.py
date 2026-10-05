import functools

import mlx.core as mx
import numpy as np


class Float32Precision:
    @staticmethod
    @functools.cache
    def matmul_is_full() -> bool:
        # The CPU and the M1 runner in CI keep full float32 in a matmul: a 64x64 product lands about
        # 1e-7 of its largest value from float64. On an M5 Max GPU with MLX 0.32 it lands about 8e-4
        # away (#812). Two orders of the same products, or MLX against a torch reference, then differ
        # by that much, so the tests that compare them get a second, measured bound there.
        rng = np.random.default_rng(0)
        a = rng.standard_normal((64, 64)).astype(np.float32)
        b = rng.standard_normal((64, 64)).astype(np.float32)
        product = np.asarray(mx.matmul(mx.array(a), mx.array(b))).astype(np.float64)
        reference = a.astype(np.float64) @ b.astype(np.float64)
        return float(np.max(np.abs(product - reference)) / np.max(np.abs(reference))) < 1e-5

    @staticmethod
    def bound(full: float, reduced: float) -> float:
        # A test's bound with full float32 matmuls, and the one measured where they are not.
        return full if Float32Precision.matmul_is_full() else reduced
