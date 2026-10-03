import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import PIL.Image
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.cli.capabilities import build_capabilities
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.flux2.cli import flux2_edit_generate, flux2_generate
from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.flux2.model.flux2_transformer.attention import Flux2Attention
from mflux.models.flux2.model.flux2_transformer.feed_forward import Flux2FeedForward
from mflux.models.flux2.model.flux2_transformer.flux2_kv_cache import Flux2KVCache
from mflux.models.flux2.model.flux2_transformer.parallel_self_attention import Flux2ParallelSelfAttention
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit
from mflux.models.qwen21 import qwen_image21_initializer
from mflux.models.qwen21.cli import qwen21_edit_generate, qwen21_generate
from mflux.models.qwen21.model.qwen21_transformer.qwen21_attention import Qwen21Attention
from mflux.models.qwen21.model.qwen21_transformer.qwen21_feed_forward import Qwen21SwiGLUFeedForward
from mflux.models.qwen21.model.qwen21_transformer.qwen21_layout import QwenImage21Layout
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_transformer.qwen_image21_transformer import QwenImage21Transformer
from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
from mflux.models.qwen21.variants.edit.qwen_image_21_edit import QwenImage21Edit
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
from mflux.models.z_image.cli import z_image_generate, z_image_turbo_generate
from mflux.models.z_image.model.z_image_transformer.attention import ZImageAttention
from mflux.models.z_image.model.z_image_transformer.feed_forward import FeedForward
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.variants.z_image import ZImage
from mflux.models.z_image.z_image_initializer import ZImageInitializer
from mflux.utils.generated_image import GeneratedImage
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals

pytestmark = pytest.mark.fast

FLOAT16 = ComputePrecision(mx.float16)
FLUX2_MODULES = (Flux2Attention, Flux2FeedForward, Flux2ParallelSelfAttention)
Z_IMAGE_MODULES = (ZImageAttention, FeedForward)
QWEN21_MODULES = (Qwen21Attention, Qwen21SwiGLUFeedForward)
QWEN21_EDIT_CONFIG = dict(
    num_layers=1,
    num_attention_heads=1,
    attention_head_dim=64,
    axes_dims_rope=(16, 24, 24),
    context_in_dim=64,
    in_channels=4,
    out_channels=4,
    mlp_ratio=2,
    eps=1e-6,
    causal_condition=True,
)


class Tiny:
    @staticmethod
    def flux2() -> Flux2Transformer:
        mx.random.seed(0)
        return Flux2Transformer(
            in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=32,
            num_attention_heads=2,
            joint_attention_dim=64,
            timestep_guidance_channels=64,
            axes_dims_rope=(8, 8, 8, 8),
        )

    @staticmethod
    def z_image() -> ZImageTransformer:
        mx.random.seed(0)
        return ZImageTransformer(
            in_channels=4,
            dim=192,
            n_layers=1,
            n_refiner_layers=1,
            n_heads=2,
            cap_feat_dim=64,
            axes_dims=[32, 32, 32],
            axes_lens=[64, 32, 32],
        )

    @staticmethod
    def qwen21() -> Qwen21Transformer:
        mx.random.seed(0)
        return Qwen21Transformer(
            in_channels=64,
            out_channels=64,
            num_layers=2,
            attention_head_dim=64,
            num_attention_heads=2,
            context_in_dim=64,
            mlp_ratio=1,
            axes_dims_rope=(8, 28, 28),
            eps=1e-6,
        )

    @staticmethod
    def qwen21_edit() -> QwenImage21Transformer:
        mx.random.seed(0)
        return QwenImage21Transformer(QWEN21_EDIT_CONFIG)

    @staticmethod
    def as_loaded(model: nn.Module, bits: int | None) -> nn.Module:
        # What the weight loader hands over: bfloat16, optionally quantized with bfloat16 scales.
        model.set_dtype(mx.bfloat16)
        if bits:
            nn.quantize(model, group_size=64, bits=bits, class_predicate=Tiny._quantizable)
        mx.eval(model.parameters())
        return model

    @staticmethod
    def compute_dtypes(root: nn.Module, module_types: tuple) -> set[mx.Dtype]:
        return {value.dtype for value in Tiny._inside(root, module_types)}

    @staticmethod
    def stream_dtypes(root: nn.Module, module_types: tuple) -> set[mx.Dtype]:
        inside = {id(value) for value in Tiny._inside(root, module_types)}
        return {value.dtype for value in Tiny._floats(root) if id(value) not in inside}

    @staticmethod
    def _inside(root: nn.Module, module_types: tuple) -> list[mx.array]:
        modules = [module for _, module in root.named_modules() if isinstance(module, module_types)]
        return [value for module in modules for value in Tiny._floats(module)]

    @staticmethod
    def _floats(module: nn.Module) -> list[mx.array]:
        return [value for _, value in tree_flatten(module.parameters()) if mx.issubdtype(value.dtype, mx.floating)]

    @staticmethod
    def _quantizable(path: str, module: nn.Module) -> bool:
        return isinstance(module, nn.Linear) and module.weight.shape[-1] % 64 == 0


class Probe(nn.Module):
    # Wraps a layer inside an attention or feed-forward module and records the dtypes its matmul ran in.
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.dtypes = set()

    def __call__(self, x: mx.array) -> mx.array:
        y = self.inner(x)
        self.dtypes.add((x.dtype, y.dtype))
        return y

    @staticmethod
    def install(targets: list[tuple[object, str | int]]) -> list["Probe"]:
        probes = []
        for parent, key in targets:
            if isinstance(key, int):
                parent[key] = Probe(parent[key])
                probes.append(parent[key])
            else:
                setattr(parent, key, Probe(getattr(parent, key)))
                probes.append(getattr(parent, key))
        return probes

    @staticmethod
    def assert_float16(probes: list["Probe"]) -> None:
        for probe in probes:
            assert probe.dtypes == {(mx.float16, mx.float16)}


class Run:
    @staticmethod
    def flux2(transformer: Flux2Transformer) -> dict[str, mx.array]:
        mx.random.seed(1)
        hidden = mx.random.normal((1, 16, 16)).astype(mx.bfloat16)
        context = mx.random.normal((1, 8, 64)).astype(mx.bfloat16)
        img_ids = mx.random.randint(0, 8, (16, 4))
        txt_ids = mx.random.randint(0, 8, (8, 4))
        timestep = mx.array([0.5])
        outputs = {"plain": transformer(hidden, context, timestep, img_ids, txt_ids)}
        cache = Flux2KVCache(num_double_layers=1, num_single_layers=1)
        cache.configure(mode="extract", num_ref_tokens=4)
        outputs["extract"] = transformer(hidden, context, timestep, img_ids, txt_ids, kv_cache=cache)
        cache.configure(mode="cached", num_ref_tokens=4)
        outputs["cached"] = transformer(hidden[:, :12], context, timestep, img_ids[:12], txt_ids, kv_cache=cache)
        return outputs

    @staticmethod
    def z_image(transformer: ZImageTransformer) -> dict[str, mx.array]:
        mx.random.seed(2)
        latents = mx.random.normal((4, 1, 8, 8)).astype(mx.bfloat16)
        captions = mx.random.normal((7, 64)).astype(mx.bfloat16)
        return {"plain": transformer(latents, mx.array(0.3), sigmas=mx.array([1.0, 0.5, 0.0]), cap_feats=captions)}

    @staticmethod
    def qwen21(transformer: Qwen21Transformer) -> dict[str, mx.array]:
        config = Config(
            width=64,
            height=64,
            guidance=1.0,
            scheduler="linear",
            model_config=ModelConfig.qwen_image_21(),
            num_inference_steps=40,
        )
        mx.random.seed(3)
        embedding = mx.random.normal((1, 8, 64)).astype(mx.bfloat16)
        latents = mx.random.normal((1, 16, 64)).astype(mx.bfloat16)

        def step(mask: mx.array) -> mx.array:
            # The geometry cache is keyed by size only; a padded prompt of the same length must rebuild it.
            transformer._geometry_cache.clear()
            return transformer(
                t=5,
                config=config,
                hidden_states=latents,
                encoder_hidden_states=embedding,
                encoder_hidden_states_mask=mask,
            )

        outputs = {"text_cache": step(mx.ones((1, 8)))}
        transformer.use_text_cache = False
        outputs["joint"] = step(mx.ones((1, 8)))
        # A padded prompt takes the masked path, whose additive mask is built in the bfloat16 stream dtype.
        outputs["padded"] = step(mx.array([[1, 1, 1, 1, 1, 1, 0, 0]]))
        transformer.use_text_cache = True
        return outputs

    @staticmethod
    def qwen21_edit(transformer: QwenImage21Transformer) -> dict[str, mx.array]:
        layout = QwenImage21Layout.create(mx.array([False, True, False]), [(1, 2, 2)] * 2, (16, 24, 24))
        hidden = (mx.arange(32, dtype=mx.float32).reshape(1, 8, 4) / 32).astype(mx.bfloat16)
        text = (mx.arange(192, dtype=mx.float32).reshape(1, 3, 64) / 192).astype(mx.bfloat16)
        cache = []
        outputs = {"plain": transformer(hidden, text, mx.array([0.6]), layout)}
        outputs["extract"] = transformer(hidden, text, mx.array([0.6]), layout, cache)
        outputs["cached"] = transformer(hidden, text, mx.array([0.4]), layout, cache)
        return outputs

    @staticmethod
    def assert_close(outputs: dict[str, mx.array], reference: dict[str, mx.array]) -> None:
        for name, expected in reference.items():
            assert outputs[name].dtype == expected.dtype, name
            np.testing.assert_allclose(
                np.array(outputs[name].astype(mx.float32)),
                np.array(expected.astype(mx.float32)),
                atol=0.05,
                err_msg=name,
            )


class SimulatedLoRA:
    # Runs where the initializers apply LoRA. A runtime adapter keeps float32 factors, and a bake that
    # re-quantizes from a float32 weight leaves float32 scales: both must end up in float16 too.
    @staticmethod
    def flux2(model, *args) -> None:
        block = model.transformer.transformer_blocks[0]
        SimulatedLoRA._adapt(model, (block.attn, "to_q"), (block.ff, "linear_out"))

    @staticmethod
    def z_image(model, *args) -> None:
        layer = model.transformer.layers[0]
        SimulatedLoRA._adapt(model, (layer.attention, "to_q"), (layer.feed_forward, "w2"))

    @staticmethod
    def qwen21(model, *args) -> None:
        block = model.transformer.transformer_blocks[0]
        SimulatedLoRA._adapt(model, (block.attn, "to_q"), (block.img_mlp, "out"))

    @staticmethod
    def _adapt(model, adapted: tuple[nn.Module, str], baked: tuple[nn.Module, str]) -> None:
        parent, name = adapted
        setattr(parent, name, LoRALinear.from_linear(getattr(parent, name), r=4))
        parent, name = baked
        output_dims, input_dims = getattr(parent, name).weight.shape
        input_dims *= 32 // getattr(parent, name).bits
        dense = nn.Linear(input_dims, output_dims, bias=False)
        setattr(parent, name, nn.QuantizedLinear.from_linear(dense, group_size=64, bits=8))
        model.lora_paths, model.lora_scales = [], []


class TinyInit:
    # Stubs out tokenizers and weight files; the initializers' own step order is what runs.
    @staticmethod
    def flux2(monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(Flux2Initializer, "_load_weights", lambda path: None)
        monkeypatch.setattr(Flux2Initializer, "_init_tokenizers", lambda model, path: None)
        monkeypatch.setattr(Flux2Initializer, "_init_models", lambda model: TinyInit._attach(model, Tiny.flux2()))
        monkeypatch.setattr(Flux2Initializer, "_apply_weights", TinyInit._load)
        monkeypatch.setattr(Flux2Initializer, "_apply_lora", SimulatedLoRA.flux2)

    @staticmethod
    def z_image(monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(ZImageInitializer, "_load_weights", lambda path: None)
        monkeypatch.setattr(ZImageInitializer, "_init_tokenizers", lambda model, path: None)
        monkeypatch.setattr(ZImageInitializer, "_init_models", lambda model: TinyInit._attach(model, Tiny.z_image()))
        monkeypatch.setattr(ZImageInitializer, "_apply_weights", TinyInit._load)
        monkeypatch.setattr(ZImageInitializer, "_apply_lora", SimulatedLoRA.z_image)

    @staticmethod
    def qwen21(monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(PathResolution, "resolve", lambda *args, **kwargs: tmp_path)
        monkeypatch.setattr(Qwen21Initializer, "_init_tokenizers", lambda model, path: None)
        monkeypatch.setattr(Qwen21Initializer, "_init_models", lambda model: TinyInit._attach(model, Tiny.qwen21()))
        monkeypatch.setattr(Qwen21Initializer, "load_components", TinyInit._load_components)
        monkeypatch.setattr(Qwen21Initializer, "apply_lora", SimulatedLoRA.qwen21)

    @staticmethod
    def qwen21_edit(monkeypatch, tmp_path) -> None:
        for name, config in (("vae", {}), ("transformer", QWEN21_EDIT_CONFIG), ("text_encoder", {})):
            (tmp_path / name).mkdir()
            (tmp_path / name / "config.json").write_text(json.dumps(config))
        tokenizers = {"qwen21": SimpleNamespace(tokenizer=None)}
        monkeypatch.setattr(PathResolution, "resolve", lambda *args, **kwargs: tmp_path)
        monkeypatch.setattr(qwen_image21_initializer.TokenizerLoader, "load_all", lambda *args: tokenizers)
        monkeypatch.setattr(qwen_image21_initializer, "QwenImage21Processor", lambda *args: None)
        monkeypatch.setattr(qwen_image21_initializer, "QwenImage21VAE", lambda config: None)
        monkeypatch.setattr(qwen_image21_initializer, "QwenImage21TextEncoder", lambda config: None)
        monkeypatch.setattr(Qwen21Initializer, "load_components", TinyInit._load_components)
        monkeypatch.setattr(Qwen21Initializer, "apply_lora", SimulatedLoRA.qwen21)

    @staticmethod
    def fail(*args, **kwargs):
        raise AssertionError("weights were loaded before the precision was validated")

    @staticmethod
    def _attach(model, transformer: nn.Module) -> None:
        model.transformer = transformer

    @staticmethod
    def _load(model, weights, quantize: int | None) -> None:
        Tiny.as_loaded(model.transformer, quantize)
        model.bits = quantize

    @staticmethod
    def _load_components(model, root, weight_definition, quantize: int | None, **kwargs) -> None:
        TinyInit._load(model, None, quantize)


class FakeTokenizer:
    def tokenize(self, prompt, max_length=None):
        input_ids = mx.arange(4, dtype=mx.int32)[None, :]
        return SimpleNamespace(input_ids=input_ids, attention_mask=mx.ones_like(input_ids))


class FakeTextEncoder:
    def __call__(self, input_ids, attention_mask):
        return mx.zeros((1, input_ids.shape[1], 8))

    def get_prompt_embeds(self, input_ids, attention_mask, hidden_state_layers):
        return mx.zeros((1, input_ids.shape[1], 8))


class FakeTransformer:
    # A zero prediction: the denoise loop, the decode and the metadata all still run.
    time_text_embed = None

    def __call__(self, **kwargs):
        return mx.zeros_like(kwargs["x"] if "x" in kwargs else kwargs["hidden_states"])

    def clear_text_cache(self) -> None:
        pass


class FakeQwen21EditTransformer:
    axes = (4, 6, 6)

    def __call__(self, hidden, text, timestep, layout, cache=None, step_cache=None):
        return mx.zeros((1, layout.target_tokens, hidden.shape[-1]))


class FakeVAE:
    bn = SimpleNamespace(running_mean=mx.zeros((128,)), running_var=mx.ones((128,)), eps=1e-4)

    def __init__(self, scale: int = 8, channels: int = 32, frames: bool = False):
        self.scale = scale
        self.channels = channels
        self.frames = frames

    def encode(self, pixels: mx.array) -> mx.array:
        size = (pixels.shape[2] // self.scale, pixels.shape[3] // self.scale)
        return mx.zeros((1, self.channels, 1, *size) if self.frames else (1, self.channels, *size))

    def decode(self, latents: mx.array) -> mx.array:
        return mx.zeros((1, 3, *latents.shape[2:-2], latents.shape[-2] * self.scale, latents.shape[-1] * self.scale))

    def decode_packed_latents(self, packed_latents: mx.array, tiling_config=None) -> mx.array:
        return mx.zeros((1, 3, packed_latents.shape[2] * 8, packed_latents.shape[3] * 8))


class Generate:
    # Runs the real generate_image of each model on fakes instead of weights, down to the GeneratedImage it returns.
    @staticmethod
    def flux2_klein(precision: ComputePrecision, reference: Path) -> GeneratedImage:
        model = Generate._model(Flux2Klein, ModelConfig.flux2_klein_4b(), precision)
        return model.generate_image(seed=1, prompt="x", num_inference_steps=1, height=64, width=64)

    @staticmethod
    def flux2_klein_edit(precision: ComputePrecision, reference: Path) -> GeneratedImage:
        model = Generate._model(Flux2KleinEdit, ModelConfig.flux2_klein_4b(), precision)
        return model.generate_image(
            seed=1, prompt="x", num_inference_steps=1, height=64, width=64, image_paths=[reference]
        )

    @staticmethod
    def z_image(precision: ComputePrecision, reference: Path) -> GeneratedImage:
        model = Generate._model(ZImage, ModelConfig.z_image_turbo(), precision)
        return model.generate_image(seed=1, prompt="x", num_inference_steps=1, height=64, width=64)

    @staticmethod
    def qwen21(precision: ComputePrecision, reference: Path) -> GeneratedImage:
        model = Generate._model(QwenImage21, ModelConfig.qwen_image_21(), precision)
        model.prompt_cache = {"x": (mx.zeros((1, 4, 64), dtype=ModelConfig.precision), mx.ones((1, 4)))}
        return model.generate_image(seed=1, prompt="x", num_inference_steps=1, height=64, width=64)

    @staticmethod
    def qwen21_edit(precision: ComputePrecision, reference: Path) -> GeneratedImage:
        model = Generate._model(QwenImage21Edit, ModelConfig.qwen_image_21(), precision)
        model.transformer = FakeQwen21EditTransformer()
        model.vae = FakeVAE(scale=16, channels=64, frames=True)
        model.processor = SimpleNamespace(
            image_processor=SimpleNamespace(size={"shortest_edge": 0, "longest_edge": 1e12})
        )
        model._encode_prompt = lambda prompt, images: (
            mx.zeros((1, 6, 8)),
            mx.array([False, True, True, True, True, False]),
        )
        return model.generate_image(
            seed=1, prompt="x", num_inference_steps=4, image_paths=[reference], output_resolution=64
        )

    @staticmethod
    def _model(model_class: type, model_config: ModelConfig, precision: ComputePrecision):
        model = model_class.__new__(model_class)
        model.__dict__.update(
            model_config=model_config,
            tokenizers={"qwen3": FakeTokenizer(), "z_image": FakeTokenizer(), "qwen21": FakeTokenizer()},
            text_encoder=FakeTextEncoder(),
            transformer=FakeTransformer(),
            vae=FakeVAE(),
            prompt_cache={},
            callbacks=CallbackRegistry(),
            tiling_config=None,
            bits=None,
            lora_paths=None,
            lora_scales=None,
            compute_precision=precision,
        )
        return model


class FakeFlux2Klein(FakeModel):
    real = Flux2Klein


class FakeFlux2KleinEdit(FakeModel):
    real = Flux2KleinEdit


class FakeQwenImage21(FakeModel):
    real = QwenImage21


class FakeQwenImage21Edit(FakeModel):
    real = QwenImage21Edit


class TestComputePrecision:
    def test_rejects_a_precision_it_does_not_support(self):
        with pytest.raises(ValueError, match="Unsupported compute precision"):
            ComputePrecision(mx.bfloat16)

    def test_off_hands_every_array_back_untouched(self):
        precision = ComputePrecision()
        x = mx.ones((2, 64), dtype=mx.bfloat16)
        linear = nn.Linear(64, 64).apply(lambda value: value.astype(mx.bfloat16))
        before = dict(tree_flatten(linear.parameters()))

        precision.apply(linear, (nn.Linear,))

        assert precision.to_compute(x) is x
        assert precision.shrink(x, 32.0) is x
        assert precision.to_stream(x, mx.float32, 32.0) is x
        assert all(value is before[key] for key, value in tree_flatten(linear.parameters()))
        assert not hasattr(linear, "compute_precision")

    def test_to_stream_clips_an_overflow_instead_of_passing_inf_on(self):
        overflowed = mx.array([float("inf"), -float("inf"), 2.0], dtype=mx.float16)
        out = FLOAT16.to_stream(overflowed, mx.float32)
        assert out.dtype == mx.float32
        assert out.tolist() == [65504.0, -65504.0, 2.0]

    def test_to_compute_leaves_non_float_arrays_alone(self):
        mask = mx.array([True, False])
        assert FLOAT16.to_compute(mask) is mask

    def test_metadata_names_the_option_only_when_it_is_on(self):
        assert FLOAT16.generation_parameters() == {"compute_precision": "float16"}
        assert ComputePrecision().generation_parameters() == {}


class TestDefaultPath:
    # Without the option nothing is cast and every module returns the dtype it was given.
    @pytest.mark.parametrize("bits", [None, 4])
    @pytest.mark.parametrize(
        ("build", "run", "module_types"),
        [
            (Tiny.flux2, Run.flux2, FLUX2_MODULES),
            (Tiny.z_image, Run.z_image, Z_IMAGE_MODULES),
            (Tiny.qwen21, Run.qwen21, QWEN21_MODULES),
            (Tiny.qwen21_edit, Run.qwen21_edit, QWEN21_MODULES),
        ],
    )
    def test_no_precision_changes_nothing(self, build, run, module_types, bits):
        reference = run(Tiny.as_loaded(build(), bits))
        transformer = Tiny.as_loaded(build(), bits)

        transformer.apply_compute_precision(ComputePrecision())

        assert Tiny.compute_dtypes(transformer, module_types) == {mx.bfloat16}
        outputs = run(transformer)
        for name, expected in reference.items():
            assert outputs[name].dtype == expected.dtype
            assert mx.array_equal(outputs[name], expected), name

    @pytest.mark.parametrize("stream", [mx.float32, mx.bfloat16])
    def test_z_image_feed_forward_off_is_the_plain_swiglu(self, stream):
        mx.random.seed(4)
        ff = FeedForward(dim=64, hidden_dim=128).apply(lambda value: value.astype(stream))
        x = mx.random.normal((1, 4, 64)).astype(stream)
        assert mx.array_equal(ff(x), ff.w2(nn.silu(ff.w1(x)) * ff.w3(x)))


class TestFloat16Internals:
    @pytest.mark.parametrize("bits", [None, 4])
    def test_flux2_runs_attention_and_ffn_in_float16(self, bits):
        transformer = Tiny.as_loaded(Tiny.flux2(), bits)
        reference = Run.flux2(transformer)

        transformer.apply_compute_precision(FLOAT16)
        double, single = transformer.transformer_blocks[0], transformer.single_transformer_blocks[0]
        probes = Probe.install(
            [
                (double.attn, "to_out"),
                (double.attn, "to_add_out"),
                (double.ff, "linear_out"),
                (double.ff_context, "linear_out"),
                (single.attn, "to_out"),
            ]
        )
        outputs = Run.flux2(transformer)

        Probe.assert_float16(probes)
        assert Tiny.compute_dtypes(transformer, FLUX2_MODULES) == {mx.float16}
        assert Tiny.stream_dtypes(transformer, FLUX2_MODULES) == {mx.bfloat16}
        Run.assert_close(outputs, reference)

    @pytest.mark.parametrize("bits", [None, 4])
    def test_z_image_runs_attention_and_ffn_in_float16(self, bits):
        transformer = Tiny.as_loaded(Tiny.z_image(), bits)
        reference = Run.z_image(transformer)

        transformer.apply_compute_precision(FLOAT16)
        probes = Probe.install(
            [
                (transformer.layers[0].attention.to_out, 0),
                (transformer.layers[0].feed_forward, "w2"),
                (transformer.context_refiner[0].attention.to_out, 0),
                (transformer.noise_refiner[0].feed_forward, "w2"),
            ]
        )
        outputs = Run.z_image(transformer)

        Probe.assert_float16(probes)
        assert Tiny.compute_dtypes(transformer, Z_IMAGE_MODULES) == {mx.float16}
        assert Tiny.stream_dtypes(transformer, Z_IMAGE_MODULES) == {mx.bfloat16}
        Run.assert_close(outputs, reference)

    @pytest.mark.parametrize("bits", [None, 4])
    def test_qwen21_runs_attention_and_ffn_in_float16_on_every_path(self, bits):
        transformer = Tiny.as_loaded(Tiny.qwen21(), bits)
        reference = Run.qwen21(transformer)

        transformer.apply_compute_precision(FLOAT16)
        probes = Probe.install(
            [
                (transformer.transformer_blocks[0].attn.to_out, 0),
                (transformer.transformer_blocks[1].img_mlp, "out"),
            ]
        )
        outputs = Run.qwen21(transformer)

        Probe.assert_float16(probes)
        assert Tiny.compute_dtypes(transformer, QWEN21_MODULES) == {mx.float16}
        assert Tiny.stream_dtypes(transformer, QWEN21_MODULES) == {mx.bfloat16}
        Run.assert_close(outputs, reference)
        (cache,) = transformer._text_caches.values()
        assert {array.dtype for kv in cache["text_kvs"] for array in kv} == {mx.float16}

    @pytest.mark.parametrize("bits", [None, 4])
    def test_qwen21_edit_reference_path_runs_in_float16(self, bits):
        transformer = Tiny.as_loaded(Tiny.qwen21_edit(), bits)
        reference = Run.qwen21_edit(transformer)

        transformer.apply_compute_precision(FLOAT16)
        probes = Probe.install(
            [
                (transformer.transformer_blocks[0].attn.to_out, 0),
                (transformer.transformer_blocks[0].img_mlp, "out"),
            ]
        )
        outputs = Run.qwen21_edit(transformer)

        Probe.assert_float16(probes)
        Run.assert_close(outputs, reference)

    def test_qwen21_drops_what_an_earlier_run_built_at_the_old_precision(self):
        transformer = Tiny.as_loaded(Tiny.qwen21(), None)
        Run.qwen21(transformer)
        assert transformer._text_caches and transformer._image_step_fn is not None

        transformer.apply_compute_precision(FLOAT16)

        assert transformer._text_caches == {}
        assert transformer._step_fn is None and transformer._image_step_fn is None


class TestZImageHeadroom:
    # Z-Image's FFN output (~1e6) and attention output (~1.4e5) overflow float16. The weights here make
    # the same happen at toy size; the float32 module is the reference.
    @pytest.mark.parametrize("stream", [mx.float32, mx.bfloat16])
    def test_feed_forward_stays_finite_where_plain_float16_overflows(self, stream):
        ff = FeedForward(dim=64, hidden_dim=128)
        ff.w1.weight = mx.full((128, 64), 9.375)
        ff.w3.weight = mx.full((128, 64), 9.375)
        ff.w2.weight = mx.full((64, 128), 0.01)
        x = mx.ones((1, 4, 64))
        expected = ff(x)  # 600 * 600 per hidden unit, then 128 * 0.01 of it: 460800

        ff16 = FeedForward(dim=64, hidden_dim=128)
        ff16.update(ff.parameters())
        FLOAT16.apply(ff16, (FeedForward,))
        x16 = x.astype(mx.float16)
        plain = ff16.w2(nn.silu(ff16.w1(x16)) * ff16.w3(x16))
        out = ff16(x.astype(stream))

        assert not mx.isfinite(plain).all()
        assert out.dtype == stream
        assert mx.allclose(out.astype(mx.float32), expected, rtol=1e-2)

    @pytest.mark.parametrize("stream", [mx.float32, mx.bfloat16])
    def test_attention_stays_finite_where_plain_float16_overflows(self, stream):
        mx.random.seed(5)
        attention = ZImageAttention(dim=64, n_heads=2)
        attention.to_v.weight = mx.full((64, 64), 31.25)
        attention.to_out[0].weight = mx.ones((64, 64))
        x = mx.ones((1, 8, 64))
        rope = mx.stack([mx.ones((8, 16)), mx.zeros((8, 16))], axis=-1)
        mask = mx.ones((1, 8), dtype=mx.bool_)
        expected = attention(x, attention_mask=mask, freqs_cis=rope)  # V = 2000, then 64 * 2000: 128000

        attention16 = ZImageAttention(dim=64, n_heads=2)
        attention16.update(attention.parameters())
        FLOAT16.apply(attention16, (ZImageAttention,))
        plain = attention16.to_out[0](attention16.to_v(x.astype(mx.float16)))
        out = attention16(x.astype(stream), attention_mask=mask, freqs_cis=rope)

        assert not mx.isfinite(plain).all()
        assert out.dtype == stream
        assert mx.allclose(out.astype(mx.float32), expected, rtol=1e-2)


class TestInitializerOrder:
    FAMILIES = [
        (Flux2Klein, TinyInit.flux2, FLUX2_MODULES),
        (Flux2KleinEdit, TinyInit.flux2, FLUX2_MODULES),
        (ZImage, TinyInit.z_image, Z_IMAGE_MODULES),
        (QwenImage21, TinyInit.qwen21, QWEN21_MODULES),
        (QwenImage21Edit, TinyInit.qwen21_edit, QWEN21_MODULES),
    ]

    @pytest.mark.parametrize(("model_class", "stub", "module_types"), FAMILIES)
    def test_cast_runs_after_quantization_and_lora(self, monkeypatch, tmp_path, model_class, stub, module_types):
        stub(monkeypatch, tmp_path)

        model = model_class(quantize=4, compute_precision=mx.float16)

        assert Tiny.compute_dtypes(model.transformer, module_types) == {mx.float16}
        assert Tiny.stream_dtypes(model.transformer, module_types) == {mx.bfloat16}

    @pytest.mark.parametrize(("model_class", "stub", "module_types"), FAMILIES)
    def test_without_the_option_nothing_is_cast(self, monkeypatch, tmp_path, model_class, stub, module_types):
        stub(monkeypatch, tmp_path)

        model = model_class(quantize=4)

        assert Tiny.compute_dtypes(model.transformer, module_types) == {mx.bfloat16, mx.float32}

    @pytest.mark.parametrize(("model_class", "stub", "module_types"), FAMILIES)
    def test_an_unsupported_precision_fails_before_any_weight_loads(
        self, monkeypatch, tmp_path, model_class, stub, module_types
    ):
        stub(monkeypatch, tmp_path)
        monkeypatch.setattr(PathResolution, "resolve", TinyInit.fail)
        monkeypatch.setattr(Flux2Initializer, "_load_weights", TinyInit.fail)
        monkeypatch.setattr(ZImageInitializer, "_load_weights", TinyInit.fail)

        with pytest.raises(ValueError, match="Unsupported compute precision"):
            model_class(compute_precision=mx.bfloat16)

    @pytest.mark.parametrize(("precision", "expected"), [(mx.float16, {"compute_precision": "float16"}), (None, {})])
    @pytest.mark.parametrize(("model_class", "stub", "module_types"), FAMILIES)
    def test_the_model_keeps_the_setting_for_its_image_metadata(
        self, monkeypatch, tmp_path, model_class, stub, module_types, precision, expected
    ):
        stub(monkeypatch, tmp_path)

        model = model_class(quantize=4, compute_precision=precision)

        assert model.compute_precision.generation_parameters() == expected


class TestImageMetadata:
    # An image made with the option records it, and --config-from-conf on that image's sidecar restores it.
    CASES = [
        (Generate.flux2_klein, flux2_generate),
        (Generate.flux2_klein_edit, flux2_edit_generate),
        (Generate.z_image, z_image_turbo_generate),
        (Generate.qwen21, qwen21_generate),
        (Generate.qwen21_edit, qwen21_edit_generate),
    ]

    @pytest.mark.parametrize(("precision", "expected"), [(FLOAT16, "float16"), (ComputePrecision(), None)])
    @pytest.mark.parametrize(("generate", "command"), CASES)
    def test_a_saved_image_replays_with_the_precision_it_was_made_with(
        self, monkeypatch, tmp_path, generate, command, precision, expected
    ):
        reset_cli_globals(monkeypatch)
        reference = tmp_path / "reference.png"
        PIL.Image.new("RGB", (64, 64), (128, 128, 128)).save(reference)

        generate(precision, reference).save(tmp_path / "image.png", export_json_metadata=True)
        sidecar = tmp_path / "image.metadata.json"
        monkeypatch.setattr(sys, "argv", ["mflux", "--config-from-conf", str(sidecar)])

        assert json.loads(sidecar.read_text()).get("compute_precision") == expected
        assert command.build_parser().parse_args().compute_precision == expected


class TestSaving:
    SAVABLE = [
        (Flux2Klein, TinyInit.flux2),
        (ZImage, TinyInit.z_image),
        (QwenImage21, TinyInit.qwen21),
        (QwenImage21Edit, TinyInit.qwen21_edit),
    ]

    @pytest.mark.parametrize(("model_class", "stub"), SAVABLE)
    def test_a_float16_compute_model_refuses_to_save_before_writing_anything(
        self, monkeypatch, tmp_path, model_class, stub
    ):
        stub(monkeypatch, tmp_path)
        model = model_class(quantize=4, compute_precision=mx.float16)

        with pytest.raises(ValueError, match="Build the model without compute_precision to save it"):
            model.save_model(str(tmp_path / "saved"))

        assert not (tmp_path / "saved").exists()

    # The edit model also writes its processor, which the stub does not have.
    @pytest.mark.parametrize(("model_class", "stub"), SAVABLE[:3])
    def test_without_the_option_the_model_still_saves(self, monkeypatch, tmp_path, model_class, stub):
        stub(monkeypatch, tmp_path)
        model = model_class(quantize=4)

        model.save_model(str(tmp_path / "saved"))

        assert list((tmp_path / "saved" / "transformer").glob("*.safetensors"))


class TestCommandLine:
    COMMANDS = [
        ("mflux-generate-flux2", flux2_generate, "Flux2Klein", FakeFlux2Klein),
        ("mflux-generate-flux2-edit", flux2_edit_generate, "Flux2KleinEdit", FakeFlux2KleinEdit),
        ("mflux-generate-qwen-2.1", qwen21_generate, "QwenImage21", FakeQwenImage21),
        ("mflux-generate-qwen-2.1-edit", qwen21_edit_generate, "QwenImage21Edit", FakeQwenImage21Edit),
    ]

    @pytest.mark.parametrize(("flag", "expected"), [([], None), (["--compute-precision", "float16"], mx.float16)])
    @pytest.mark.parametrize(("command", "module", "attribute", "fake"), COMMANDS)
    def test_the_flag_reaches_the_model(self, monkeypatch, tmp_path, command, module, attribute, fake, flag, expected):
        fake.instances.clear()
        monkeypatch.setattr(module, attribute, fake)
        reset_cli_globals(monkeypatch)
        PIL.Image.new("RGB", (64, 64)).save(tmp_path / "ref.png")
        images = ["--image-paths", str(tmp_path / "ref.png")] if command == "mflux-generate-flux2-edit" else []
        # A missing prompt file stops main() right after the model is built.
        argv = [command, "--prompt-file", str(tmp_path / "missing.txt"), *images, *flag]
        monkeypatch.setattr(sys, "argv", argv)

        module.main()

        assert len(fake.instances) == 1
        assert fake.instances[0].init_kwargs["compute_precision"] is expected

    @pytest.mark.parametrize(
        "module",
        [
            flux2_generate,
            flux2_edit_generate,
            z_image_generate,
            z_image_turbo_generate,
            qwen21_generate,
            qwen21_edit_generate,
        ],
    )
    def test_config_from_conf_restores_the_option(self, monkeypatch, tmp_path, module):
        TestCommandLine._replay(monkeypatch, tmp_path, compute_precision="float16")
        assert module.build_parser().parse_args().compute_precision == "float16"

    def test_the_command_line_wins_over_the_sidecar(self, monkeypatch, tmp_path):
        # With the flag given, the sidecar's value is never read, so even a bad one goes unnoticed.
        TestCommandLine._replay(monkeypatch, tmp_path, "--compute-precision", "float16", compute_precision="bfloat16")
        assert flux2_generate.build_parser().parse_args().compute_precision == "float16"

    @pytest.mark.parametrize("recorded", ["bfloat16", 16, ["float16"]])
    def test_a_bad_value_in_the_sidecar_fails_while_parsing(self, monkeypatch, tmp_path, recorded):
        TestCommandLine._replay(monkeypatch, tmp_path, compute_precision=recorded)
        with pytest.raises(SystemExit) as exc:
            flux2_generate.build_parser().parse_args()
        assert exc.value.code == 2

    def test_an_unsupported_value_is_rejected_by_the_parser(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["mflux-generate-flux2", "--prompt", "x", "--compute-precision", "bfloat16"])
        with pytest.raises(SystemExit):
            flux2_generate.build_parser().parse_args()

    def test_only_the_supporting_commands_take_the_flag(self):
        flags = {
            command["command"]: {option["flag"]: option["status"] for option in command.get("options", [])}
            for command in build_capabilities()["commands"]
        }
        supporting = {name for name, options in flags.items() if "--compute-precision" in options}
        assert supporting == {
            "mflux-generate-flux2",
            "mflux-generate-flux2-edit",
            "mflux-generate-z-image",
            "mflux-generate-z-image-turbo",
            "mflux-generate-qwen-2.1",
            "mflux-generate-qwen-2.1-edit",
        }
        assert {flags[name]["--compute-precision"] for name in supporting} == {"honored"}

    @staticmethod
    def _replay(monkeypatch, tmp_path, *flags: str, **recorded) -> None:
        reset_cli_globals(monkeypatch)
        PIL.Image.new("RGB", (64, 64)).save(tmp_path / "ref.png")
        sidecar = tmp_path / "image.metadata.json"
        sidecar.write_text(
            json.dumps({"prompt": "x", "seed": 1, "image_paths": [str(tmp_path / "ref.png")], **recorded})
        )
        monkeypatch.setattr(sys, "argv", ["mflux", "--config-from-conf", str(sidecar), *flags])
