import math
import random
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn
from mlx.optimizers import AdamW
from mlx.utils import tree_flatten

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.training.dataset.batch import Batch, DataItem
from mflux.models.common.training.lora.target_injector import inject_lora_targets
from mflux.models.common.training.state.training_spec import LoraLayersSpec, LoraTargetSpec
from mflux.models.common.training.trainer import TrainingTrainer
from mflux.models.fibo2.model.fibo2_text_encoder.prompt_encoder import Fibo2PromptEncoder
from mflux.models.fibo2.model.fibo2_transformer.fibo2_perceiver import Fibo2Perceiver
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer import Fibo2Transformer
from mflux.models.fibo2.training_adapter import fibo2_training_adapter as adapter_module

pytestmark = pytest.mark.fast

Fibo2TrainingAdapter = adapter_module.Fibo2TrainingAdapter

# A tiny FIBO 2 transformer: 4 layers with cross-attention in layers 1 and 3, so 3 text bundles, and 4 to 8 gist tokens
RANK, ALPHA = 4, 8.0
LATENT_SHAPE = (1, 4, 8, 8)
TEXT_SHAPE = (1, 3, 6, 48)


class _TinyFibo2(nn.Module):
    # Stands in for Fibo2: the adapter reaches the transformer, the prompt cache and generate_image
    def __init__(self, *, quantize=None, model_config=None, model_path=None):  # noqa: ARG002
        super().__init__()
        self.transformer = _TinyFibo2.transformer_with_weights(seed=0)
        self.vae = nn.Module()
        self.text_encoder = nn.Module()
        self.prompt_cache = {}
        self.generate_kwargs = None

    def generate_image(self, **kwargs):
        self.generate_kwargs = kwargs
        self.prompt_cache["preview"] = mx.zeros((2,))
        return SimpleNamespace(image="preview-image")

    @staticmethod
    def transformer_with_weights(seed: int) -> Fibo2Transformer:
        transformer = Fibo2Transformer(
            in_channels=4,
            dim=64,
            n_layers=4,
            n_refiner_layers=2,
            n_heads=2,
            cap_feat_dim=48,
            injection_layer_ids=(1, 3),
            perceiver_num_layers=2,
            max_num_gist_tokens=8,
            axes_dims=(8, 12, 12),
            axes_lens=(16, 8, 8),
        )
        transformer.perceiver = Fibo2Perceiver(64, 2, 48, 2, 1e-5, min_gist=4, max_gist=8, gist_step=2, min_text_len=4, max_text_len=12)  # fmt: off
        # Random everywhere, the zero cross-attention gates and the latent queries too, so every layer moves the output
        mx.random.seed(seed)
        weights = []
        for name, value in tree_flatten(transformer.parameters()):
            base = 1.0 if value.ndim == 1 and "norm" in name and "adaLN" not in name else 0.0
            weights.append((name, base + 0.2 * mx.random.normal(value.shape)))
        transformer.load_weights(weights, strict=True)
        return transformer


class _Fake:
    @staticmethod
    def adapter(monkeypatch) -> Fibo2TrainingAdapter:
        monkeypatch.setattr(ModelConfig, "precision", mx.float32)
        monkeypatch.setattr(adapter_module, "Fibo2", _TinyFibo2)
        # The caption drop is random per step; tests that don't look at it keep every step on its caption
        monkeypatch.setattr(Fibo2TrainingAdapter, "TEXT_DROP", 0.0)
        return Fibo2TrainingAdapter(model_config=ModelConfig.fibo2(), quantize=None)

    @staticmethod
    def targets(transformer: nn.Module) -> list[LoraTargetSpec]:
        # Every linear layer, as the reference trainer adapts them
        paths = [name for name, module in transformer.named_modules() if isinstance(module, nn.Linear)]
        return [LoraTargetSpec(module_path=path, rank=RANK, alpha=ALPHA) for path in paths]

    @staticmethod
    def with_lora(adapter: Fibo2TrainingAdapter) -> list[LoraTargetSpec]:
        targets = _Fake.targets(adapter.transformer())
        inject_lora_targets(adapter.transformer(), targets)
        adapter.freeze_base()
        TrainingTrainer._unfreeze_lora_layers(adapter.transformer())
        return targets

    @staticmethod
    def randomize_lora_b(transformer: nn.Module) -> None:
        # A new adapter starts with B = 0, an identity; a trained one does not
        mx.random.seed(3)
        for _, module in transformer.named_modules():
            if isinstance(module, LoRALinear):
                module.lora_B = 0.05 * mx.random.normal(module.lora_B.shape)

    @staticmethod
    def training_spec(targets: list[LoraTargetSpec]) -> SimpleNamespace:
        return SimpleNamespace(
            model="fibo2",
            steps=4,
            guidance=5.0,
            training_loop=SimpleNamespace(timestep_low=0, timestep_high=None, timestep_type=None),
            lora_layers=LoraLayersSpec(targets=targets),
        )

    @staticmethod
    def item() -> DataItem:
        mx.random.seed(1)
        cond = {"embeds": mx.random.normal(TEXT_SHAPE)}
        latents = mx.random.normal(LATENT_SHAPE)
        return DataItem(data_id=0, prompt="p", image_path="p.png", clean_latents=latents, cond=cond, width=64, height=64)  # fmt: off

    @staticmethod
    def output(transformer: nn.Module) -> mx.array:
        mx.random.seed(2)
        latents, text = mx.random.normal(LATENT_SHAPE), mx.random.normal(TEXT_SHAPE)
        return transformer(latents, mx.array([0.6]), text)


class _ScriptedRandom:
    def __init__(self, uniforms: list[float], normals: list[float]):
        self.uniforms, self.normals = list(uniforms), list(normals)

    def random(self) -> float:
        return self.uniforms.pop(0)

    def gauss(self, mu: float, sigma: float) -> float:
        return mu + sigma * self.normals.pop(0)


def test_noise_levels_follow_the_reference_trainer(monkeypatch):
    adapter = _Fake.adapter(monkeypatch)
    # 90%: sigmoid(N(shift, 1)), the shift 0.5 at 256 image tokens and 1.15 at 4096
    assert adapter.sample_sigma(rng=_ScriptedRandom([0.5], [0.0]), image_seq_len=4096) == pytest.approx(1 / (1 + math.exp(-1.15)))  # fmt: off
    assert adapter.sample_sigma(rng=_ScriptedRandom([0.5], [1.0]), image_seq_len=256) == pytest.approx(1 / (1 + math.exp(-1.5)))  # fmt: off
    # 10%: uniform
    assert adapter.sample_sigma(rng=_ScriptedRandom([0.05, 0.3], []), image_seq_len=4096) == 0.3

    rng = random.Random(0)
    sigmas = [adapter.sample_sigma(rng=rng, image_seq_len=1024) for _ in range(20000)]
    assert all(0.0 <= sigma <= 1.0 for sigma in sigmas)
    # At 1024 tokens (512x512) the shift is 0.63. The share of draws below x is 0.9 * P(N(0.63, 1) < logit(x)) + 0.1 * x
    for x in (0.2, 0.5, 0.65, 0.8, 0.95):
        logit_normal = 0.5 * (1 + math.erf((math.log(x / (1 - x)) - 0.63) / math.sqrt(2)))
        assert sum(sigma < x for sigma in sigmas) / len(sigmas) == pytest.approx(0.9 * logit_normal + 0.1 * x, abs=0.01)


def test_the_caption_without_its_text_images_conditions_noise_levels_above_0_9(monkeypatch):
    adapter = _Fake.adapter(monkeypatch)
    seen = []
    monkeypatch.setattr(adapter._fibo2, "transformer", lambda latents, sigma, embeds: seen.append((sigma, embeds)) or latents * 2)  # fmt: off
    latents = mx.ones(LATENT_SHAPE)
    with_images, without = mx.zeros(TEXT_SHAPE), mx.ones(TEXT_SHAPE)
    cond = {"embeds": with_images, "base_embeds": without}

    velocity = adapter.predict_noise(t=0, latents_t=latents, sigmas=mx.array([0.95]), cond=cond, config=None)
    adapter.predict_noise(t=1, latents_t=latents, sigmas=mx.array([0.95, 0.5]), cond=cond, config=None)
    adapter.predict_noise(t=0, latents_t=latents, sigmas=mx.array([0.95]), cond={"embeds": with_images}, config=None)

    assert [float(sigma[0]) for sigma, _ in seen] == pytest.approx([0.95, 0.5, 0.95])
    assert [embeds is without for _, embeds in seen] == [True, False, False]
    assert mx.array_equal(velocity, latents * 2).item()


def test_the_null_caption_conditions_a_tenth_of_the_steps(monkeypatch):
    # Bria's trainer drops the caption for 10% of the steps (--text-drop), to the null caption FIBO 2 was trained with.
    # Without it, at 300 single-image steps the LoRA learned the line work on every run but the background color on
    # one run in three
    adapter = _Fake.adapter(monkeypatch)
    monkeypatch.setattr(Fibo2TrainingAdapter, "TEXT_DROP", 0.1)
    seen = []
    monkeypatch.setattr(adapter._fibo2, "transformer", lambda latents, sigma, embeds: seen.append(embeds) or latents)
    full, null = mx.zeros(TEXT_SHAPE), mx.full(TEXT_SHAPE, 2.0)
    adapter._null = null
    mx.random.seed(0)
    for _ in range(1000):
        adapter.predict_noise(t=0, latents_t=mx.ones(LATENT_SHAPE), sigmas=mx.array([0.5]), cond={"embeds": full}, config=None)  # fmt: off
    assert sum(embeds is null for embeds in seen) / len(seen) == pytest.approx(0.1, abs=0.03)
    assert all(embeds is full or embeds is null for embeds in seen)


def test_the_null_caption_is_encoded_before_the_text_encoder_goes(monkeypatch):
    adapter = _Fake.adapter(monkeypatch)
    model = _with_prompt_encoding(adapter)
    spec = SimpleNamespace(guidance=5.0, monitoring=SimpleNamespace(preview_prompts=[]))
    assert adapter.release_text_encoder(spec)
    assert Fibo2PromptEncoder.NULL_CAPTION in model.encoded and adapter._null is not None and model.prompt_cache == {}


def test_a_few_training_steps_lower_the_loss_of_a_tiny_fibo2(monkeypatch):
    adapter = _Fake.adapter(monkeypatch)
    targets = _Fake.with_lora(adapter)
    spec = _Fake.training_spec(targets)
    base_config = adapter.create_config(spec, width=64, height=64)
    item = _Fake.item()

    def loss(_model):
        # The same noise level and noise each time, so the loss can only fall through the LoRA
        return TrainingTrainer.compute_loss(adapter, spec, base_config, Batch(data=[item], rng=random.Random(7)))

    value_and_grad = nn.value_and_grad(adapter.model(), loss)
    optimizer = AdamW(learning_rate=1e-3)
    losses = []
    for _ in range(4):
        value, grads = value_and_grad(adapter.model())
        flat = dict(tree_flatten(grads))
        # Only the LoRA matrices train, and every one of them gets a gradient
        assert flat and all(name.endswith(("lora_A", "lora_B")) for name in flat)
        assert len(flat) == 2 * len(targets)
        optimizer.update(adapter.model(), grads)
        mx.eval(adapter.model().parameters(), optimizer.state)
        losses.append(float(value))
    assert all(math.isfinite(value) for value in losses)
    assert losses[-1] < losses[0]


def test_the_saved_lora_has_comfyui_keys_and_loads_back_at_its_strength(monkeypatch, tmp_path):
    adapter = _Fake.adapter(monkeypatch)
    targets = _Fake.with_lora(adapter)
    _Fake.randomize_lora_b(adapter.transformer())
    expected = _Fake.output(adapter.transformer())
    spec = _Fake.training_spec(targets)
    path = tmp_path / "adapter.safetensors"
    adapter.save_lora_adapter(path=path, training_spec=spec)

    weights = mx.load(str(path))
    # ComfyUI's names and fused layers: q/k/v stacked into attention.qkv (rank 3 x RANK), the timestep MLP and final
    # modulation under their Sequential indices, to_out without its .0
    assert weights["diffusion_model.t_embedder.mlp.0.lora_A.weight"].shape == (RANK, 256)
    assert weights["diffusion_model.final_layer.adaLN_modulation.1.lora_B.weight"].shape == (64, RANK)
    assert weights["diffusion_model.layers.0.attention.qkv.lora_A.weight"].shape[0] == 3 * RANK
    assert "diffusion_model.layers.1.cross_attention.to_out.lora_A.weight" in weights
    assert not any(".to_q.lora" in key and ".attention.to_q" in key and "cross" not in key for key in weights)

    # Loaded on a fresh model with the same base weights, the adapter gives the same output, alpha / rank included
    resumed = Fibo2TrainingAdapter(model_config=ModelConfig.fibo2(), quantize=None)
    assert not mx.allclose(_Fake.output(resumed.transformer()), expected, atol=1e-4).item()
    resumed.load_lora_adapter(path=path)
    assert mx.allclose(_Fake.output(resumed.transformer()), expected, rtol=1e-5, atol=1e-4).item()

    # Through the fused mapping, as --lora and ComfyUI read the file, without the per-module split of a resume
    fused = Fibo2TrainingAdapter(model_config=ModelConfig.fibo2(), quantize=None)
    fused.load_training_adapter(path=path)
    assert mx.allclose(_Fake.output(fused.transformer()), expected, rtol=1e-5, atol=1e-4).item()

    # A resumed run saves the same strength again
    again = tmp_path / "again.safetensors"
    resumed.save_lora_adapter(path=again, training_spec=spec)
    reloaded = Fibo2TrainingAdapter(model_config=ModelConfig.fibo2(), quantize=None)
    reloaded.load_lora_adapter(path=again)
    assert mx.allclose(_Fake.output(reloaded.transformer()), expected, rtol=1e-5, atol=1e-4).item()


def test_a_resumed_run_gets_each_module_back_at_its_own_rank_and_scale(monkeypatch, tmp_path):
    # The saved optimizer state has one entry per trained module at its rank: resuming from the fused ComfyUI file must
    # give q, k, v, w1 and w3 their own down and up weights again, not the stacked ones, and alpha / rank back as the
    # layer's scale rather than folded into up, which would change the weights the optimizer state belongs to
    from mflux.models.common.training.lora.path_util import expand_module_paths_from_targets
    from mflux.models.common.training.utils import TrainingUtil

    adapter = _Fake.adapter(monkeypatch)
    targets = _Fake.with_lora(adapter)
    _Fake.randomize_lora_b(adapter.transformer())
    spec = _Fake.training_spec(targets)
    path = tmp_path / "adapter.safetensors"
    adapter.save_lora_adapter(path=path, training_spec=spec)

    resumed = Fibo2TrainingAdapter(model_config=ModelConfig.fibo2(), quantize=None)
    resumed.load_lora_adapter(path=path)
    for module_path, _, _ in expand_module_paths_from_targets(spec.lora_layers.targets):
        trained = TrainingUtil.get_train_lora(adapter.transformer(), module_path)
        loaded = TrainingUtil.get_train_lora(resumed.transformer(), module_path)
        assert loaded.lora_A.shape == trained.lora_A.shape, module_path
        assert mx.allclose(loaded.lora_A, trained.lora_A).item(), module_path
        assert loaded.scale == pytest.approx(trained.scale) and trained.scale != 1.0, module_path
        assert mx.allclose(loaded.lora_B, trained.lora_B, atol=1e-6).item(), module_path


def test_the_feed_forward_is_padded_like_the_comfyui_repack():
    # Bria's 9536-wide feed-forward is stored 9728 wide in Comfy-Org's repack; the saved LoRA matches it
    from mflux.models.fibo2.training_adapter.fibo2_comfy_lora import Fibo2ComfyLoRA
    from mflux.models.fibo2.weights.fibo2_lora_mapping import Fibo2LoRAMapping

    rank, dim, hidden = 2, 8, Fibo2LoRAMapping.FFN_HIDDEN
    loras = {
        "layers.0.feed_forward.w1": (mx.ones((rank, dim)), mx.ones((hidden, rank)), 1.0),
        "layers.0.feed_forward.w3": (mx.ones((rank, dim)), 2 * mx.ones((hidden, rank)), 1.0),
        "layers.0.feed_forward.w2": (mx.ones((rank, hidden)), mx.ones((dim, rank)), 0.5),
    }
    weights = Fibo2ComfyLoRA.weights(loras)
    up = weights["diffusion_model.layers.0.feed_forward.w13.lora_B.weight"]
    assert up.shape == (2 * 9728, 2 * rank)
    assert float(up[:hidden, :rank].min()) == 1.0 and float(up[9728 : 9728 + hidden, rank:].min()) == 2.0
    assert float(mx.abs(up[hidden:9728]).max()) == 0.0
    assert weights["diffusion_model.layers.0.feed_forward.w2.lora_A.weight"].shape == (rank, 9728)
    assert float(weights["diffusion_model.layers.0.feed_forward.w2.lora_B.weight"].max()) == 0.5


def test_gradient_checkpointing_gives_the_same_loss_and_gradients(monkeypatch):
    adapter = _Fake.adapter(monkeypatch)
    _Fake.with_lora(adapter)
    _Fake.randomize_lora_b(adapter.transformer())

    def loss(model):
        return _Fake.output(model.transformer).square().mean()

    results = []
    for enabled in (False, True):
        adapter.set_gradient_checkpointing(enabled)
        value, grads = nn.value_and_grad(adapter.model(), loss)(adapter.model())
        if enabled:
            TrainingTrainer._evaluate_backward_in_block_order(value, grads)
        results.append((float(value), dict(tree_flatten(grads))))
    (stock_loss, stock_grads), (ckpt_loss, ckpt_grads) = results
    assert ckpt_loss == pytest.approx(stock_loss, rel=1e-5)
    assert ckpt_grads.keys() == stock_grads.keys()
    for name, grad in stock_grads.items():
        assert mx.allclose(ckpt_grads[name], grad, rtol=1e-4, atol=1e-6).item(), name


def test_previews_use_the_training_guidance_and_leave_no_cached_prompts(monkeypatch):
    adapter = _Fake.adapter(monkeypatch)
    adapter.create_config(SimpleNamespace(steps=8, guidance=5.0), width=64, height=64)

    image = adapter.generate_preview_image(seed=7, prompt="a teapot", width=64, height=64, steps=6)

    assert image == "preview-image"
    assert adapter._fibo2.generate_kwargs["guidance"] == 5.0
    assert adapter._fibo2.generate_kwargs["num_inference_steps"] == 6
    # Arrays in the model's prompt cache would be parameters the optimizer trains and decays
    assert adapter._fibo2.prompt_cache == {}


def _with_prompt_encoding(adapter):
    # The parts of Fibo2 the release uses: a caption with text_render draws text images, the rest is cached by text
    model = adapter.model()
    model.encoded = []

    def encode(prompt, images=None):
        model.encoded.append(prompt)
        model.prompt_cache.setdefault(prompt, mx.full((2,), float(len(model.encoded))))
        return model.prompt_cache[prompt]

    model._encode_prompt = encode
    model._encode_prompts = lambda prompt, images, text_encoder_images, render_text_images: (encode(prompt), None)
    model._text_images = lambda caption: ["text"] if isinstance(caption, dict) and caption.get("text_render") else []
    return model


def test_the_text_encoder_is_released_and_previews_use_the_kept_encodings(monkeypatch):
    adapter = _Fake.adapter(monkeypatch)
    model = _with_prompt_encoding(adapter)
    spec = SimpleNamespace(guidance=5.0, monitoring=SimpleNamespace(preview_prompts=["a bluxyz drawing of a lighthouse"]))  # fmt: off

    assert adapter.release_text_encoder(spec)

    assert model.text_encoder is None
    assert model.prompt_cache == {}  # nothing left in the module, which training would treat as parameters
    seen = {}
    model.generate_image = lambda **kwargs: seen.update(cache=dict(model.prompt_cache)) or SimpleNamespace(image="img")
    adapter.generate_preview_image(seed=1, prompt="a bluxyz drawing of a lighthouse", width=64, height=64, steps=2)
    assert len(seen["cache"]) == 2  # the preview prompt and the default negative, encoded before the release
    assert model.prompt_cache == {}


def test_a_preview_that_draws_text_keeps_the_text_encoder(monkeypatch):
    adapter = _Fake.adapter(monkeypatch)
    model = _with_prompt_encoding(adapter)
    caption = '{"short_description": "a sign", "text_render": [{"text": "OPEN"}]}'
    spec = SimpleNamespace(guidance=5.0, monitoring=SimpleNamespace(preview_prompts=[caption]))

    assert not adapter.release_text_encoder(spec)
    assert model.text_encoder is not None
