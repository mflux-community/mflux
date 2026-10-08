from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mflux.models.common.config import ModelConfig
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import QwenTransformerBlock
from mflux.models.qwen.qwen_initializer import QwenImageInitializer
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

pytestmark = pytest.mark.fast

# Qwen-Image-Edit-2511 sets zero_cond_t in its transformer config: the tokens of the reference images are
# modulated with the embedding of timestep 0, the noisy latent with the current one. 2509 has no such split.
TINY = dict(num_layers=2, num_attention_heads=2, attention_head_dim=128, joint_attention_dim=64)
TARGET, REFERENCE = (1, 4, 6), (1, 4, 4)  # latent grids in patches: 24 target tokens, 16 reference tokens


class _Tiny:
    @staticmethod
    def model(zero_cond_t: bool) -> QwenTransformer:
        mx.random.seed(0)
        model = QwenTransformer(**TINY, zero_cond_t=zero_cond_t)
        # proj_out starts near zero in a fresh model; give every weight a value so the output moves.
        model.update({"proj_out": {"weight": mx.random.normal(model.proj_out.weight.shape) * 0.05}})
        mx.eval(model.parameters())
        return model

    @staticmethod
    def run(model: QwenTransformer, t: float, reference: bool, hidden=None) -> mx.array:
        tokens = TARGET[1] * TARGET[2] + (REFERENCE[1] * REFERENCE[2] if reference else 0)
        mx.random.seed(1)
        hidden = mx.random.normal((1, tokens, 64)) if hidden is None else hidden
        return model(
            t=t,
            config=SimpleNamespace(height=TARGET[1] * 16, width=TARGET[2] * 16),
            hidden_states=hidden,
            encoder_hidden_states=mx.random.normal((1, 5, 64)),
            encoder_hidden_states_mask=mx.ones((1, 5)),
            cond_image_grid=REFERENCE if reference else None,
        )


def test_modulate_gives_the_reference_tokens_the_other_modulation():
    x = mx.ones((1, 3, 2))
    at_t = mx.array([[1.0, 1.0, 0.0, 0.0, 5.0, 5.0]])  # shift 1, scale 0, gate 5
    at_zero = mx.array([[2.0, 2.0, 1.0, 1.0, 7.0, 7.0]])  # shift 2, scale 1, gate 7
    is_reference = mx.array([False, False, True])[None, :, None]

    modulated, gate = QwenTransformerBlock._modulate(x, at_t, at_zero, is_reference)

    assert modulated[0, :, 0].tolist() == [2.0, 2.0, 4.0]
    assert gate[0, :, 0].tolist() == [5.0, 5.0, 7.0]


def test_zero_cond_t_changes_an_edit_and_leaves_a_run_without_references_alone():
    plain, split = _Tiny.model(zero_cond_t=False), _Tiny.model(zero_cond_t=True)

    assert mx.array_equal(_Tiny.run(plain, 0.6, reference=False), _Tiny.run(split, 0.6, reference=False)).item()
    assert not mx.allclose(_Tiny.run(plain, 0.6, reference=True), _Tiny.run(split, 0.6, reference=True)).item()
    # At timestep 0 the two modulations are the same one, so the split changes nothing.
    assert mx.allclose(_Tiny.run(plain, 0.0, reference=True), _Tiny.run(split, 0.0, reference=True), atol=1e-5).item()


def test_the_2511_entry_builds_its_transformer_with_zero_cond_t(monkeypatch):
    built = {}

    def transformer(**kwargs):
        built.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("mflux.models.qwen.qwen_initializer.QwenTransformer", transformer)
    monkeypatch.setattr("mflux.models.qwen.qwen_initializer.QwenVAE", lambda: SimpleNamespace())
    encoder = SimpleNamespace(encoder=SimpleNamespace())
    monkeypatch.setattr("mflux.models.qwen.qwen_initializer.QwenTextEncoder", lambda: encoder)
    monkeypatch.setattr("mflux.models.qwen.qwen_initializer.VisionTransformer", lambda: SimpleNamespace())

    QwenImageInitializer._init_edit_models(SimpleNamespace(model_config=ModelConfig.qwen_image_edit_2511()))
    assert built == {"zero_cond_t": True}

    built.clear()
    QwenImageInitializer._init_edit_models(SimpleNamespace(model_config=ModelConfig.qwen_image_edit()))
    assert built == {}


def test_zero_cond_t_matches_diffusers(tmp_path):
    # Optional: needs a diffusers with zero_cond_t (0.36 or later). The tiny torch reference runs on CPU.
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers", reason="Install diffusers to compare with its Qwen-Image transformer")
    reference = diffusers.QwenImageTransformer2DModel(
        patch_size=2, in_channels=64, out_channels=16, axes_dims_rope=(16, 56, 56), zero_cond_t=True, **TINY
    ).eval()
    torch.manual_seed(0)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.05)
    reference.save_pretrained(tmp_path / "transformer")
    hidden, text = torch.randn(1, 40, 64), torch.randn(1, 5, 64)
    with torch.no_grad():
        expected = reference(
            hidden_states=hidden,
            encoder_hidden_states=text,
            encoder_hidden_states_mask=torch.ones(1, 5, dtype=torch.long),
            timestep=torch.tensor([0.6]),
            img_shapes=[[TARGET, REFERENCE]],
            return_dict=False,
        )[0].numpy()

    component = next(c for c in QwenWeightDefinition.get_components() if c.name == "transformer")
    outputs = {}
    for zero_cond_t in (True, False):
        model = QwenTransformer(**TINY, zero_cond_t=zero_cond_t)
        WeightApplier.apply_and_quantize(
            weights=WeightLoader.load_single_local(component, tmp_path),
            models={"transformer": model},
            quantize_arg=None,
            weight_definition=QwenWeightDefinition,
        )
        assert all(value.dtype == mx.float32 for _, value in tree_flatten(model.parameters()))
        outputs[zero_cond_t] = np.array(
            model(
                t=0.6,
                config=SimpleNamespace(height=TARGET[1] * 16, width=TARGET[2] * 16),
                hidden_states=mx.array(hidden.numpy()),
                encoder_hidden_states=mx.array(text.numpy()),
                encoder_hidden_states_mask=mx.ones((1, 5)),
                cond_image_grid=REFERENCE,
            ).astype(mx.float32)
        )

    assert np.abs(outputs[True] - expected).max() < 5e-3
    assert np.abs(outputs[False] - expected).max() > 0.1
