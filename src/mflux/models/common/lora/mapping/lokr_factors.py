import mlx.core as mx

LOKR_MATRIX_KEYS = frozenset(
    {
        "lokr_w1",
        "lokr_w2",
        "lokr_w1_a",
        "lokr_w1_b",
        "lokr_w2_a",
        "lokr_w2_b",
        "lokr_t2",
    }
)


def is_lokr_adapter(lora_data: dict) -> bool:
    return bool(LOKR_MATRIX_KEYS & lora_data.keys())


def rebuild_lokr_factors(lora_data: dict) -> tuple[mx.array, mx.array, float]:
    w1 = lora_data.get("lokr_w1")
    w2 = lora_data.get("lokr_w2")
    dim: int | None = None

    if w1 is None:
        w1_a = lora_data.get("lokr_w1_a")
        w1_b = lora_data.get("lokr_w1_b")
        if w1_a is None or w1_b is None:
            raise ValueError("Missing LoKr w1 tensors; need lokr_w1 or both lokr_w1_a and lokr_w1_b")
        dim = w1_b.shape[0]
        w1 = mx.matmul(w1_a, w1_b)

    if w2 is None:
        w2_a = lora_data.get("lokr_w2_a")
        w2_b = lora_data.get("lokr_w2_b")
        if w2_a is None or w2_b is None:
            raise ValueError("Missing LoKr w2 tensors; need lokr_w2 or both lokr_w2_a and lokr_w2_b")
        dim = w2_b.shape[0]
        t2 = lora_data.get("lokr_t2")
        if t2 is None:
            w2 = mx.matmul(w2_a, w2_b)
        else:
            w2 = mx.einsum("i j k l, j r, i p -> p r k l", t2, w2_b, w2_a)

    alpha_scale = 1.0
    if "alpha" in lora_data and dim is not None:
        alpha_scale = _scalar_float(lora_data["alpha"]) / dim

    return w1, w2, alpha_scale


def _scalar_float(value: mx.array | float) -> float:
    if isinstance(value, mx.array):
        return float(value.item())
    return float(value)
