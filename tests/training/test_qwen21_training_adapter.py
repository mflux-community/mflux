import math
import random
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.training.dataset.batch import Batch, DataItem
from mflux.models.common.training.lora.path_util import expand_module_paths_from_targets
from mflux.models.common.training.lora.target_injector import inject_lora_targets
from mflux.models.common.training.state.training_spec import BlockRange, LoraLayersSpec, LoraTargetSpec
from mflux.models.common.training.trainer import TrainingTrainer
from mflux.models.common.training.utils import TrainingUtil
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.training_adapter import qwen21_training_adapter as adapter_module
from mflux.models.qwen21.weights.qwen21_lora_mapping import Qwen21LoRAMapping

pytestmark = pytest.mark.fast

Qwen21TrainingAdapter = adapter_module.Qwen21TrainingAdapter

# A tiny Qwen-Image-2.1 transformer: 2 blocks, 2 heads of 16, a 4x4 latent grid (64x64 pixels), 5 text tokens
RANK, ALPHA = 4, 8.0
SIZE = 64
LATENT_SHAPE = (1, (SIZE // 16) ** 2, 8)
TEXT_SHAPE = (1, 5, 24)
# ai-toolkit's targets
TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out")


class _TinyQwen21(nn.Module):
    # Stands in for QwenImage21: the adapter reaches the transformer, the VAE, the prompt cache and generate_image
    def __init__(self, *, quantize=None, model_config=None, model_path=None, bake_lora=False):  # noqa: ARG002
        super().__init__()
        mx.random.seed(0)
        self.transformer = Qwen21Transformer(
            in_channels=8,
            out_channels=8,
            num_layers=2,
            attention_head_dim=16,
            num_attention_heads=2,
            context_in_dim=24,
            mlp_ratio=2,
            axes_dims_rope=(4, 6, 6),
        )
        self.vae = nn.Module()
        self.text_encoder = nn.Module()
        self.prompt_cache = {}
        self.tiling_config = None
        self.encoded = []

    def _encode_prompt(self, prompt):
        self.encoded.append(prompt)
        value = float(len(self.encoded))
        self.prompt_cache[prompt] = (mx.full(TEXT_SHAPE, value), mx.ones(TEXT_SHAPE[:2], dtype=mx.int32))
        return self.prompt_cache[prompt]


class _Fake:
    @staticmethod
    def adapter(monkeypatch) -> Qwen21TrainingAdapter:
        monkeypatch.setattr(ModelConfig, "precision", mx.float32)
        monkeypatch.setattr(adapter_module, "QwenImage21", _TinyQwen21)
        return Qwen21TrainingAdapter(model_config=ModelConfig.qwen_image_21(), quantize=None)

    @staticmethod
    def with_lora(adapter: Qwen21TrainingAdapter) -> SimpleNamespace:
        targets = [
            LoraTargetSpec(
                module_path=f"transformer_blocks.{{block}}.{name}", blocks=BlockRange(0, 2), rank=RANK, alpha=ALPHA
            )  # fmt: off
            for name in TARGETS
        ]
        inject_lora_targets(adapter.transformer(), targets)
        adapter.freeze_base()
        TrainingTrainer._unfreeze_lora_layers(adapter.transformer())
        # A new adapter starts with B = 0, an identity; a trained one does not
        mx.random.seed(3)
        for _, module in adapter.transformer().named_modules():
            if isinstance(module, LoRALinear):
                module.lora_B = 0.05 * mx.random.normal(module.lora_B.shape)
        return SimpleNamespace(model="qwen-image-2.1", steps=4, guidance=None, lora_layers=LoraLayersSpec(targets=targets))  # fmt: off

    @staticmethod
    def item() -> DataItem:
        mx.random.seed(1)
        cond = {"embeds": mx.random.normal(TEXT_SHAPE), "mask": mx.ones(TEXT_SHAPE[:2], dtype=mx.int32)}
        return DataItem(data_id=0, prompt="p", image_path="p.png", clean_latents=mx.random.normal(LATENT_SHAPE), cond=cond, width=SIZE, height=SIZE)  # fmt: off

    @staticmethod
    def output(transformer: Qwen21Transformer) -> mx.array:
        mx.random.seed(2)
        latents, text = mx.random.normal(LATENT_SHAPE), mx.random.normal(TEXT_SHAPE)
        return transformer.forward_training(latents, text, mx.array([0.6]), SIZE, SIZE)


def test_the_saved_lora_has_ai_toolkit_keys_and_loads_through_the_qwen21_mapping(monkeypatch, tmp_path):
    adapter = _Fake.adapter(monkeypatch)
    spec = _Fake.with_lora(adapter)
    expected = _Fake.output(adapter.transformer())
    path = tmp_path / "adapter.safetensors"
    adapter.save_lora_adapter(path=path, training_spec=spec)

    weights = mx.load(str(path))
    # ai-toolkit's names: diffusion_model.*, the gate and up of the feed-forward in one gate_up of twice the rank
    assert weights["diffusion_model.transformer_blocks.1.attn.to_out.0.lora_A.weight"].shape == (RANK, 32)
    assert weights["diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_A.weight"].shape == (2 * RANK, 32)
    assert weights["diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight"].shape == (2 * 64, 2 * RANK)
    assert len(weights) == 2 * 2 * 6 and not any("gate_layer" in key or ".proj." in key for key in weights)

    # A fresh model (the same base weights) with the file loaded the way --lora reads it gives the trained output
    plain = _Fake.adapter(monkeypatch)
    assert not mx.allclose(_Fake.output(plain.transformer()), expected, atol=1e-4).item()
    LoRALoader.load_and_apply_lora(
        lora_mapping=Qwen21LoRAMapping.get_mapping(),
        transformer=plain.transformer(),
        lora_paths=[str(path)],
        lora_scales=[1.0],
        bake_lora=False,
    )
    assert mx.allclose(_Fake.output(plain.transformer()), expected, rtol=1e-5, atol=1e-5).item()

    # A resumed run gets each module back at its own rank and scale, which the saved optimizer state matches
    resumed = _Fake.adapter(monkeypatch)
    resumed.load_lora_adapter(path=path)
    assert mx.allclose(_Fake.output(resumed.transformer()), expected, rtol=1e-5, atol=1e-5).item()
    for module_path, _, _ in expand_module_paths_from_targets(spec.lora_layers.targets):
        trained = TrainingUtil.get_train_lora(adapter.transformer(), module_path)
        loaded = TrainingUtil.get_train_lora(resumed.transformer(), module_path)
        assert loaded.lora_A.shape == trained.lora_A.shape and loaded.scale == pytest.approx(trained.scale), module_path


def test_the_empty_caption_is_drawn_from_the_step_rng_and_never_in_validation(monkeypatch):
    adapter = _Fake.adapter(monkeypatch)
    spec = _Fake.with_lora(adapter)
    assert adapter.release_text_encoder(SimpleNamespace(monitoring=None))
    empty = adapter._empty
    assert adapter.model().encoded == [""] and adapter.model().text_encoder is None

    # ai-toolkit's caption_dropout_rate: 5% of the steps, the same ones again for the same rng state (a resume)
    rng = random.Random(0)
    saved = rng.getstate()
    drops = [adapter.training_condition(cond={}, rng=rng) is empty for _ in range(4000)]
    assert sum(drops) / len(drops) == pytest.approx(0.05, abs=0.01)
    rng.setstate(saved)
    assert [adapter.training_condition(cond={}, rng=rng) is empty for _ in range(4000)] == drops

    base_config = adapter.create_config(spec, width=SIZE, height=SIZE)
    item = _Fake.item()

    def loss(**kwargs):
        return float(TrainingTrainer.compute_loss(adapter, spec, base_config, Batch(data=[item], rng=random.Random(7)), **kwargs))  # fmt: off

    clean = loss()
    monkeypatch.setattr(Qwen21TrainingAdapter, "CAPTION_DROPOUT", 1.0)
    assert loss(training=False) == clean
    assert loss() != clean


def test_gradient_checkpointing_gives_the_same_loss_and_lora_only_gradients(monkeypatch):
    adapter = _Fake.adapter(monkeypatch)
    spec = _Fake.with_lora(adapter)
    base_config = adapter.create_config(spec, width=SIZE, height=SIZE)
    item = _Fake.item()

    def loss(_model):
        return TrainingTrainer.compute_loss(adapter, spec, base_config, Batch(data=[item], rng=random.Random(7)))

    results = []
    for enabled in (False, True):
        adapter.set_gradient_checkpointing(enabled)
        value, grads = nn.value_and_grad(adapter.model(), loss)(adapter.model())
        if enabled:
            TrainingTrainer._evaluate_backward_in_block_order(value, grads)
        results.append((float(value), dict(tree_flatten(grads))))
    (stock_loss, stock_grads), (ckpt_loss, ckpt_grads) = results
    assert len(stock_grads) == 2 * 2 * len(TARGETS) and all(k.endswith(("lora_A", "lora_B")) for k in stock_grads)
    assert ckpt_loss == pytest.approx(stock_loss, rel=1e-5)
    for name, grad in stock_grads.items():
        assert float(mx.abs(grad).max()) > 0, name
        assert mx.allclose(ckpt_grads[name], grad, rtol=1e-4, atol=1e-6).item(), name


def test_the_noise_level_is_logit_normal_around_sigma_mu():
    adapter = Qwen21TrainingAdapter.__new__(Qwen21TrainingAdapter)
    rng = random.Random(0)
    sigmas = sorted(adapter.sample_sigma(rng=rng, image_seq_len=1024) for _ in range(4001))
    assert all(0.0 < s < 1.0 for s in sigmas)
    median = 1.0 / (1.0 + math.exp(-Qwen21TrainingAdapter.SIGMA_MU))
    assert sigmas[2000] == pytest.approx(median, abs=0.02)


def test_gate_layer_without_proj_is_rejected_before_training():
    from mflux.models.qwen21.training_adapter.qwen21_comfy_lora import Qwen21ComfyLoRA

    Qwen21ComfyLoRA.check_targets(["transformer_blocks.0.img_mlp.gate_layer", "transformer_blocks.0.img_mlp.proj"])
    with pytest.raises(ValueError, match="train together"):
        Qwen21ComfyLoRA.check_targets(["transformer_blocks.0.img_mlp.gate_layer", "transformer_blocks.1.img_mlp.proj"])


def test_modulation_is_saved_under_its_checkpoint_name_and_split_back():
    from mflux.models.qwen21.training_adapter.qwen21_comfy_lora import Qwen21ComfyLoRA

    down, up = mx.ones((4, 8)), mx.full((6, 4), 2.0)
    saved = Qwen21ComfyLoRA.weights({"modulation.layers.1": (down, up, 0.5)})
    assert set(saved) == {"diffusion_model.modulation.1.lora_A.weight", "diffusion_model.modulation.1.lora_B.weight"}
    split = Qwen21ComfyLoRA.split(saved, {"modulation.layers.1": 4}, {"modulation.layers.1": 0.5})
    assert mx.array_equal(split["transformer.modulation.1.lora_B.weight"], up)
    assert split["transformer.modulation.1.alpha"].item() == 2.0
