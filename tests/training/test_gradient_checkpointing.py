import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.training.trainer import TrainingTrainer
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer

DIM, HEADS, HEAD_DIM = 128, 4, 32
IMG_TOKENS, TXT_TOKENS = 512, 16


class _TinyFlux2:
    @staticmethod
    def build(num_layers: int, num_single_layers: int) -> nn.Module:
        mx.random.seed(0)
        transformer = Flux2Transformer(
            in_channels=32,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=HEAD_DIM,
            num_attention_heads=HEADS,
            joint_attention_dim=64,
            axes_dims_rope=(8, 8, 8, 8),
        )
        model = nn.Module()
        model.transformer = transformer
        _TinyFlux2._inject_lora(transformer)
        model.freeze()
        for _, child in model.named_modules():
            if isinstance(child, LoRALinear):
                child.unfreeze(keys=["lora_A", "lora_B"], recurse=False)
        mx.eval(model.parameters())
        return model

    @staticmethod
    def inputs() -> dict:
        mx.random.seed(1)
        img_ids = mx.zeros((1, IMG_TOKENS, 4), dtype=mx.int32)
        txt_ids = mx.zeros((1, TXT_TOKENS, 4), dtype=mx.int32)
        inputs = {
            "hidden_states": mx.random.normal((1, IMG_TOKENS, 32)),
            "encoder_hidden_states": mx.random.normal((1, TXT_TOKENS, 64)),
            "timestep": mx.array([0.5]),
            "img_ids": img_ids,
            "txt_ids": txt_ids,
        }
        mx.eval(inputs)
        return inputs

    @staticmethod
    def loss(model: nn.Module, inputs: dict) -> mx.array:
        return model.transformer(**inputs).square().mean()

    @staticmethod
    def _inject_lora(module: nn.Module) -> None:
        for key, value in list(module.items()):
            if isinstance(value, nn.Linear) and not isinstance(value, LoRALinear):
                module[key] = LoRALinear.from_linear(value, r=4, scale=1.0)
            elif isinstance(value, nn.Module):
                _TinyFlux2._inject_lora(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, nn.Module):
                        _TinyFlux2._inject_lora(item)


def _run(model: nn.Module, inputs: dict, *, checkpointing: bool, ordered: bool) -> tuple[float, float, int]:
    model.transformer.gradient_checkpointing = checkpointing
    mx.reset_peak_memory()
    value_and_grad = nn.value_and_grad(model, lambda m, i: _TinyFlux2.loss(m, i))
    loss, grads = value_and_grad(model, inputs)
    if ordered:
        TrainingTrainer._evaluate_backward_in_block_order(loss, grads)
    else:
        mx.eval(loss, grads)
    flat = tree_flatten(grads)
    grad_norm = float(mx.sqrt(sum((g.astype(mx.float32) ** 2).sum() for _, g in flat)))
    return float(loss), grad_norm, mx.get_peak_memory()


@pytest.mark.fast
def test_checkpointing_matches_stock_loss_and_gradients():
    model = _TinyFlux2.build(num_layers=2, num_single_layers=6)
    inputs = _TinyFlux2.inputs()

    stock_loss, stock_norm, _ = _run(model, inputs, checkpointing=False, ordered=False)
    ckpt_loss, ckpt_norm, _ = _run(model, inputs, checkpointing=True, ordered=True)

    assert ckpt_loss == pytest.approx(stock_loss, rel=1e-4)
    assert ckpt_norm == pytest.approx(stock_norm, rel=1e-3)


@pytest.mark.fast
def test_ordered_backward_lowers_peak_memory_below_plain_checkpointing():
    model = _TinyFlux2.build(num_layers=2, num_single_layers=12)
    inputs = _TinyFlux2.inputs()

    _, _, stock_peak = _run(model, inputs, checkpointing=False, ordered=False)
    _, _, ckpt_peak = _run(model, inputs, checkpointing=True, ordered=False)
    _, _, ordered_peak = _run(model, inputs, checkpointing=True, ordered=True)

    # Plain checkpointing helps, but evaluating grads in tree order keeps most recomputed
    # blocks alive; the ordered backward is what brings the peak down to "a few blocks".
    assert ckpt_peak < stock_peak
    assert ordered_peak < 0.8 * ckpt_peak


@pytest.mark.fast
def test_block_prefix_groups_grads_by_block():
    prefix = TrainingTrainer._BLOCK_PREFIX
    assert (
        prefix.match("transformer.transformer_blocks.3.attn.to_q.lora_A").group(1) == "transformer.transformer_blocks.3"
    )
    assert prefix.match("transformer.single_transformer_blocks.19.attn.to_out.lora_B").group(1) == "transformer.single_transformer_blocks.19"  # fmt: off
    assert prefix.match("layers.7.attention.to_q.lora_A").group(1) == "layers.7"
    assert prefix.match("cap_embedder.1.lora_A").group(1) == "cap_embedder.1"
    assert prefix.match("transformer.proj_out.lora_A") is None
