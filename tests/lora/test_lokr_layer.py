import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lokr_layer import LoKrLinear
from mflux.models.common.lora.mapping.lokr_factors import rebuild_lokr_factors
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.lora.mapping.lora_saver import LoRASaver
from mflux.models.flux2.weights.flux2_lora_mapping import Flux2LoRAMapping


def test_lokr_linear_applies_kronecker_delta():
    linear = nn.Linear(4, 2, bias=False)
    linear.weight = mx.zeros((2, 4))
    lokr_w1 = mx.array([[1.0]])
    lokr_w2 = mx.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    lokr_linear = LoKrLinear.from_linear(linear, lokr_w1=lokr_w1, lokr_w2=lokr_w2, scale=0.5)

    x = mx.array([[1.0, 1.0, 1.0, 1.0]])
    expected = 0.5 * mx.matmul(x, lokr_w2.T)

    assert mx.allclose(lokr_linear(x), expected)


def test_lokr_matmul_matches_materialized_delta():
    mx.random.seed(0)
    linear = nn.Linear(12, 8, bias=False)
    linear.weight = mx.zeros((8, 12))
    lokr_w1 = mx.random.normal((2, 3))
    lokr_w2 = mx.random.normal((4, 4))
    lokr_linear = LoKrLinear.from_linear(linear, lokr_w1=lokr_w1, lokr_w2=lokr_w2)

    x = mx.random.normal((3, 5, 12))
    dense_delta = lokr_linear.delta_weight()
    # Metal accumulates at lower precision for small shapes; 1e-2 is safe on both CPU and GPU
    assert mx.allclose(lokr_linear.lokr_matmul(x), mx.matmul(x, dense_delta.T), rtol=1e-2, atol=1e-2)


def test_lokr_linear_ignores_alpha_tensor_in_loader():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 2, bias=False)
            self.proj.weight = mx.zeros((2, 4))

    transformer = Transformer()
    weights = {
        "lokr_w1": mx.array([[1.0]]),
        "lokr_w2": mx.ones((2, 4)),
        "alpha": mx.array(99.0),
    }
    assert LoRALoader._apply_adapter_to_target(transformer, "proj", weights, scale=1.0, role=None)
    assert mx.allclose(transformer.proj(mx.ones((1, 4))), mx.array([[4.0, 4.0]]))


def test_lokr_linear_applies_dora_scale():
    linear = nn.Linear(2, 2, bias=False)
    linear.weight = mx.array([[3.0, 4.0], [0.0, 2.0]])
    lokr_linear = LoKrLinear.from_linear(
        linear,
        lokr_w1=mx.array([[1.0]]),
        lokr_w2=mx.eye(2),
        dora_scale=mx.array([[10.0], [3.0]]),
    )

    x = mx.array([[1.0, 2.0]])
    merged_weight = linear.weight + mx.eye(2)
    weight_norm = mx.linalg.norm(merged_weight, axis=1, keepdims=True) + mx.finfo(merged_weight.dtype).eps
    decomposed_weight = merged_weight * mx.array([[10.0], [3.0]]) / weight_norm
    expected = mx.matmul(x, decomposed_weight.T)

    assert mx.allclose(lokr_linear(x), expected)


def test_loader_replaces_linear_with_lokr_layer():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 2, bias=False)
            self.proj.weight = mx.zeros((2, 4))

    transformer = Transformer()
    applied = LoRALoader._apply_adapter_to_target(
        transformer,
        "proj",
        {
            "lokr_w1": mx.array([[1.0]]),
            "lokr_w2": mx.ones((2, 4)),
        },
        scale=1.0,
        role=None,
    )

    assert applied is True
    assert isinstance(transformer.proj, LoKrLinear)
    assert mx.allclose(transformer.proj(mx.ones((1, 4))), mx.array([[4.0, 4.0]]))


def test_loader_rejects_partial_lokr_group():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 2, bias=False)

    with pytest.raises(ValueError, match="Invalid LoKr matrices for proj"):
        LoRALoader._apply_adapter_to_target(
            Transformer(),
            "proj",
            {
                "lokr_w1": mx.array([[1.0]]),
            },
            scale=1.0,
            role=None,
        )


def test_rebuild_lokr_factors_from_w1_decomposition():
    w1_a = mx.array([[2.0, 0.0]])
    w1_b = mx.array([[1.0], [3.0]])
    w2 = mx.ones((2, 4))
    w1, rebuilt_w2, alpha_scale = rebuild_lokr_factors(
        {
            "lokr_w1_a": w1_a,
            "lokr_w1_b": w1_b,
            "lokr_w2": w2,
            "alpha": mx.array(6.0),
        }
    )

    assert mx.allclose(w1, mx.matmul(w1_a, w1_b))
    assert mx.allclose(rebuilt_w2, w2)
    assert alpha_scale == 3.0


def test_rebuild_lokr_factors_from_w2_decomposition():
    w1 = mx.array([[1.0]])
    w2_a = mx.array([[1.0, 0.0], [0.0, 1.0]])
    w2_b = mx.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
    w1, w2, alpha_scale = rebuild_lokr_factors(
        {
            "lokr_w1": w1,
            "lokr_w2_a": w2_a,
            "lokr_w2_b": w2_b,
            "alpha": mx.array(4.0),
        }
    )

    assert mx.allclose(w1, mx.array([[1.0]]))
    assert mx.allclose(w2, mx.matmul(w2_a, w2_b))
    assert alpha_scale == 2.0


def test_loader_applies_factorized_lokr_with_alpha():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 2, bias=False)
            self.proj.weight = mx.zeros((2, 4))

    transformer = Transformer()
    applied = LoRALoader._apply_adapter_to_target(
        transformer,
        "proj",
        {
            "lokr_w1_a": mx.array([[1.0]]),
            "lokr_w1_b": mx.array([[1.0]]),
            "lokr_w2": mx.ones((2, 4)),
            "alpha": mx.array(2.0),
        },
        scale=1.0,
        role=None,
    )

    assert applied is True
    assert isinstance(transformer.proj, LoKrLinear)
    assert mx.allclose(transformer.proj(mx.ones((1, 4))), mx.array([[8.0, 8.0]]))


def test_loader_applies_fully_factorized_lokr():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 2, bias=False)
            self.proj.weight = mx.zeros((2, 4))

    transformer = Transformer()
    applied = LoRALoader._apply_adapter_to_target(
        transformer,
        "proj",
        {
            "lokr_w1_a": mx.array([[1.0]]),
            "lokr_w1_b": mx.array([[1.0]]),
            "lokr_w2_a": mx.ones((2, 1)),
            "lokr_w2_b": mx.ones((1, 4)),
            "alpha": mx.array(2.0),
        },
        scale=0.5,
        role=None,
    )

    assert applied is True
    assert mx.allclose(transformer.proj(mx.ones((1, 4))), mx.array([[4.0, 4.0]]))


def test_loader_applies_lokr_with_t2_dense_fallback():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4, bias=False)
            self.proj.weight = mx.zeros((4, 4))

    transformer = Transformer()
    t2 = mx.arange(16, dtype=mx.float32).reshape((2, 2, 2, 2)) + 1
    applied = LoRALoader._apply_adapter_to_target(
        transformer,
        "proj",
        {
            "lokr_w1": mx.array([[1.0]]),
            "lokr_w2_a": mx.eye(2),
            "lokr_w2_b": mx.eye(2),
            "lokr_t2": t2,
        },
        scale=1.0,
        role=None,
    )

    x = mx.array([[1.0, 2.0, 3.0, 4.0]])
    expected = mx.matmul(x, transformer.proj.delta_weight().T)

    assert applied is True
    assert isinstance(transformer.proj, LoKrLinear)
    assert transformer.proj.lokr_w2.ndim == 4
    assert transformer.proj.can_use_factorized_matmul() is False
    assert mx.allclose(transformer.proj(x), expected)


def test_loader_rejects_malformed_lokr_shapes():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 2, bias=False)

    with pytest.raises(ValueError, match="Invalid LoKr matrices for proj"):
        LoRALoader._apply_adapter_to_target(
            Transformer(),
            "proj",
            {
                "lokr_w1": mx.ones((2, 2)),
                "lokr_w2": mx.ones((2, 2)),
            },
            scale=1.0,
            role=None,
        )


def test_loader_rejects_files_that_apply_no_layers(tmp_path):
    class Transformer(nn.Module):
        pass

    lora_path = tmp_path / "unmatched.safetensors"
    mx.save_safetensors(str(lora_path), {"unmatched.weight": mx.ones((1,))})

    with pytest.raises(ValueError, match="No LoRA layers were applied from unmatched.safetensors"):
        LoRALoader._apply_single_lora(
            Transformer(),
            str(lora_path),
            scale=1.0,
            lora_mapping=[],
            role=None,
        )


def test_flux2_double_block_qkv_lokr_splits_w2():
    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q = nn.Linear(2, 2, bias=False)
            self.to_k = nn.Linear(2, 2, bias=False)
            self.to_v = nn.Linear(2, 2, bias=False)
            self.to_q.weight = mx.zeros((2, 2))
            self.to_k.weight = mx.zeros((2, 2))
            self.to_v.weight = mx.zeros((2, 2))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attn()

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = [Block()]

    transformer = Transformer()
    mappings = LoRALoader._build_pattern_mappings(Flux2LoRAMapping.get_mapping())
    weights = {
        "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w1": mx.array([[1.0]]),
        "diffusion_model.double_blocks.0.img_attn.qkv.lokr_w2": mx.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, 2.0],
                [3.0, 0.0],
                [0.0, 3.0],
            ]
        ),
    }

    applied_count, matched_keys = LoRALoader._apply_lora_with_mapping(
        transformer, weights, scale=1.0, pattern_mappings=mappings, role=None
    )

    x = mx.ones((1, 2))
    assert applied_count == 3
    assert matched_keys == set(weights)
    assert mx.allclose(transformer.transformer_blocks[0].attn.to_q(x), mx.array([[1.0, 1.0]]))
    assert mx.allclose(transformer.transformer_blocks[0].attn.to_k(x), mx.array([[2.0, 2.0]]))
    assert mx.allclose(transformer.transformer_blocks[0].attn.to_v(x), mx.array([[3.0, 3.0]]))


def test_loader_fuses_lora_then_lokr_layer():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)
            self.proj.weight = mx.zeros((2, 2))

    transformer = Transformer()
    assert LoRALoader._apply_adapter_to_target(
        transformer,
        "proj",
        {
            "lora_A": mx.ones((2, 1)),
            "lora_B": mx.ones((1, 2)),
        },
        scale=1.0,
        role=None,
    )
    assert LoRALoader._apply_adapter_to_target(
        transformer,
        "proj",
        {
            "lokr_w1": mx.array([[1.0]]),
            "lokr_w2": mx.eye(2),
        },
        scale=1.0,
        role=None,
    )

    assert isinstance(transformer.proj, FusedLoRALinear)
    assert mx.allclose(transformer.proj(mx.ones((1, 2))), mx.array([[3.0, 3.0]]))


def test_loader_fuses_lora_then_lokr_t2_layer():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 4, bias=False)
            self.proj.weight = mx.zeros((4, 4))

    transformer = Transformer()
    assert LoRALoader._apply_adapter_to_target(
        transformer,
        "proj",
        {
            "lora_A": mx.ones((4, 1)),
            "lora_B": mx.ones((1, 4)),
        },
        scale=1.0,
        role=None,
    )

    t2 = mx.arange(16, dtype=mx.float32).reshape((2, 2, 2, 2)) + 1
    assert LoRALoader._apply_adapter_to_target(
        transformer,
        "proj",
        {
            "lokr_w1": mx.array([[1.0]]),
            "lokr_w2_a": mx.eye(2),
            "lokr_w2_b": mx.eye(2),
            "lokr_t2": t2,
        },
        scale=1.0,
        role=None,
    )

    x = mx.array([[1.0, 2.0, 3.0, 4.0]])
    lokr = transformer.proj.loras[-1]
    expected_lora = mx.matmul(mx.matmul(x, mx.ones((4, 1))), mx.ones((1, 4)))
    expected_lokr = mx.matmul(x, lokr.delta_weight().T)

    assert isinstance(transformer.proj, FusedLoRALinear)
    assert isinstance(lokr, LoKrLinear)
    assert lokr.can_use_factorized_matmul() is False
    assert mx.allclose(transformer.proj(x), expected_lora + expected_lokr)


def test_loader_fuses_lora_then_dora_lokr_at_weight_level():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)
            self.proj.weight = mx.zeros((2, 2))

    transformer = Transformer()
    assert LoRALoader._apply_adapter_to_target(
        transformer,
        "proj",
        {
            "lora_A": mx.ones((2, 1)),
            "lora_B": mx.ones((1, 2)),
        },
        scale=1.0,
        role=None,
    )
    assert LoRALoader._apply_adapter_to_target(
        transformer,
        "proj",
        {
            "lokr_w1": mx.array([[1.0]]),
            "lokr_w2": mx.eye(2),
            "dora_scale": mx.array([[10.0], [20.0]]),
        },
        scale=1.0,
        role=None,
    )

    current_weight = mx.ones((2, 2))
    merged_weight = current_weight + mx.eye(2)
    weight_norm = mx.linalg.norm(merged_weight, axis=1, keepdims=True) + mx.finfo(merged_weight.dtype).eps
    decomposed_weight = merged_weight * mx.array([[10.0], [20.0]]) / weight_norm
    expected = mx.matmul(mx.ones((1, 2)), decomposed_weight.T)

    assert isinstance(transformer.proj, FusedLoRALinear)
    assert mx.allclose(transformer.proj(mx.ones((1, 2))), expected)


def test_loader_fuses_lokr_then_lora_layer():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)
            self.proj.weight = mx.zeros((2, 2))

    transformer = Transformer()
    assert LoRALoader._apply_adapter_to_target(
        transformer,
        "proj",
        {
            "lokr_w1": mx.array([[1.0]]),
            "lokr_w2": mx.eye(2),
        },
        scale=1.0,
        role=None,
    )
    assert LoRALoader._apply_adapter_to_target(
        transformer,
        "proj",
        {
            "lora_A": mx.ones((2, 1)),
            "lora_B": mx.ones((1, 2)),
        },
        scale=1.0,
        role=None,
    )

    assert isinstance(transformer.proj, FusedLoRALinear)
    assert mx.allclose(transformer.proj(mx.ones((1, 2))), mx.array([[3.0, 3.0]]))


def test_saver_bakes_lokr_into_quantized_linear():
    linear = nn.Linear(64, 32, bias=False)
    linear.weight = mx.zeros((32, 64))
    quantized = linear.to_quantized(group_size=32, bits=8)
    lokr = LoKrLinear.from_linear(
        quantized,
        lokr_w1=mx.array([[1.0]]),
        lokr_w2=mx.ones((32, 64)),
        scale=1.0,
    )

    baked = LoRASaver._bake_lokr_into_linear(quantized, lokr)

    assert isinstance(baked, nn.QuantizedLinear)
    assert baked.bits == 8
    assert baked.group_size == 32
    x = mx.ones((1, 64))
    expected = lokr(x)
    assert mx.allclose(baked(x), expected, rtol=1e-4, atol=1e-3)


def test_load_and_apply_lora_bakes_lokr_by_default():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)
            self.proj.weight = mx.zeros((2, 2))

    transformer = Transformer()
    assert LoRALoader._apply_adapter_to_target(
        transformer,
        "proj",
        {
            "lokr_w1": mx.array([[1.0]]),
            "lokr_w2": mx.eye(2),
        },
        scale=1.0,
        role=None,
    )
    assert isinstance(transformer.proj, LoKrLinear)

    LoRASaver.bake_and_strip_lora(transformer)
    mx.eval(transformer.parameters())

    assert isinstance(transformer.proj, nn.Linear)
    assert mx.allclose(transformer.proj(mx.ones((1, 2))), mx.array([[1.0, 1.0]]))


def test_saver_bakes_lokr_layer():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)
            self.proj.weight = mx.zeros((2, 2))

    transformer = Transformer()
    assert LoRALoader._apply_adapter_to_target(
        transformer,
        "proj",
        {
            "lokr_w1": mx.array([[1.0]]),
            "lokr_w2": mx.eye(2),
        },
        scale=0.5,
        role=None,
    )

    LoRASaver.bake_and_strip_lora(transformer)

    assert isinstance(transformer.proj, nn.Linear)
    assert mx.allclose(transformer.proj.weight, 0.5 * mx.eye(2))
    assert mx.allclose(transformer.proj(mx.ones((1, 2))), mx.array([[0.5, 0.5]]))


def test_saver_bakes_fused_lora_and_lokr_layer():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)
            self.proj.weight = mx.zeros((2, 2))

    transformer = Transformer()
    assert LoRALoader._apply_adapter_to_target(
        transformer,
        "proj",
        {
            "lora_A": mx.ones((2, 1)),
            "lora_B": mx.ones((1, 2)),
        },
        scale=1.0,
        role=None,
    )
    assert LoRALoader._apply_adapter_to_target(
        transformer,
        "proj",
        {
            "lokr_w1": mx.array([[1.0]]),
            "lokr_w2": mx.eye(2),
        },
        scale=1.0,
        role=None,
    )

    LoRASaver.bake_and_strip_lora(transformer)

    assert isinstance(transformer.proj, nn.Linear)
    assert mx.allclose(transformer.proj(mx.ones((1, 2))), mx.array([[3.0, 3.0]]))


def test_saver_bakes_fused_lora_and_dora_lokr_at_weight_level():
    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2, bias=False)
            self.proj.weight = mx.zeros((2, 2))

    transformer = Transformer()
    assert LoRALoader._apply_adapter_to_target(
        transformer,
        "proj",
        {
            "lora_A": mx.ones((2, 1)),
            "lora_B": mx.ones((1, 2)),
        },
        scale=1.0,
        role=None,
    )
    assert LoRALoader._apply_adapter_to_target(
        transformer,
        "proj",
        {
            "lokr_w1": mx.array([[1.0]]),
            "lokr_w2": mx.eye(2),
            "dora_scale": mx.array([[10.0], [20.0]]),
        },
        scale=1.0,
        role=None,
    )

    current_weight = mx.ones((2, 2))
    merged_weight = current_weight + mx.eye(2)
    weight_norm = mx.linalg.norm(merged_weight, axis=1, keepdims=True) + mx.finfo(merged_weight.dtype).eps
    decomposed_weight = merged_weight * mx.array([[10.0], [20.0]]) / weight_norm

    LoRASaver.bake_and_strip_lora(transformer)

    assert isinstance(transformer.proj, nn.Linear)
    assert mx.allclose(transformer.proj.weight, decomposed_weight)
