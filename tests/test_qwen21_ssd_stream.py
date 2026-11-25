import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.ssd_stream.streamer import StreamedBlock
from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_text_encoder import Qwen21TextEncoder
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

pytestmark = pytest.mark.fast


class Qwen21StreamSupport:
    TRANSFORMER = dict(
        in_channels=4,
        out_channels=4,
        context_in_dim=16,
        num_layers=3,
        num_attention_heads=1,
        attention_head_dim=64,
        mlp_ratio=1,
        axes_dims_rope=(16, 24, 24),
    )
    TEXT_ENCODER = dict(
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=64,
        head_dim=32,
        mrope_section=[6, 5, 5],
    )

    @staticmethod
    def randomized(module, seed: int):
        rng = np.random.default_rng(seed)
        weights = []
        for name, value in sorted(tree_flatten(module.parameters())):
            data = rng.standard_normal(value.shape).astype(np.float32) * 0.1
            if "norm" in name:
                data += 1
            weights.append((name, mx.array(data)))
        module.load_weights(weights)
        mx.eval(module.parameters())
        return module

    @staticmethod
    def transformer() -> Qwen21Transformer:
        return Qwen21StreamSupport.randomized(Qwen21Transformer(**Qwen21StreamSupport.TRANSFORMER), 741)

    @staticmethod
    def text_encoder() -> Qwen21TextEncoder:
        return Qwen21StreamSupport.randomized(Qwen21TextEncoder(**Qwen21StreamSupport.TEXT_ENCODER), 742)

    @staticmethod
    def save_sharded(module, folder) -> None:
        # The layout of the original transformer: two diffusers shards and their index, under the module's names.
        folder.mkdir(parents=True)
        weights = dict(tree_flatten(module.parameters()))
        names = sorted(weights)
        shards = {"diffusion_pytorch_model-00001-of-00002.safetensors": names[::2]}
        shards["diffusion_pytorch_model-00002-of-00002.safetensors"] = names[1::2]
        weight_map = {}
        for shard, keys in shards.items():
            mx.save_safetensors(str(folder / shard), {key: weights[key] for key in keys})
            weight_map.update(dict.fromkeys(keys, shard))
        index = folder / "diffusion_pytorch_model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": weight_map}))

    @staticmethod
    def save_original_text_encoder(module, folder) -> None:
        # The layout of the original text encoder: one file without an index, names under model.language_model.
        folder.mkdir(parents=True)
        weights = {f"model.language_model.{name}": value for name, value in tree_flatten(module.parameters())}
        mx.save_safetensors(str(folder / "model.safetensors"), weights)

    @staticmethod
    def stream(root, transformer, text_encoder=None) -> None:
        model = SimpleNamespace(
            transformer=transformer, text_encoder=text_encoder, small_text_encoder=text_encoder is None
        )
        Qwen21Initializer._apply_ssd_stream(model, root)

    @staticmethod
    def denoise(model: Qwen21Transformer, use_text_cache: bool) -> list[mx.array]:
        rng = np.random.default_rng(21)
        latents = mx.array(rng.standard_normal((1, 4, 4)).astype(np.float32))
        text = mx.array(rng.standard_normal((1, 3, 16)).astype(np.float32))
        config = Config(
            width=32,
            height=32,
            guidance=1.0,
            scheduler="linear",
            model_config=ModelConfig.qwen_image_21(),
            num_inference_steps=40,
        )
        model.use_text_cache = use_text_cache
        outputs = []
        with model.inference_projections():
            for timestep in (1, 2):
                outputs.append(model(timestep, config, latents, text))
                mx.eval(outputs[-1])
        return outputs

    @staticmethod
    def installed(blocks) -> bool:
        return any(value.size > 1 for block in blocks for _, value in tree_flatten(block.module.parameters()))

    @staticmethod
    def lora(module, seed: int) -> None:
        rng = np.random.default_rng(seed)
        for block in module.transformer_blocks:
            adapter = LoRALinear.from_linear(block.attn.to_q, r=2)
            adapter.lora_A = mx.array(rng.standard_normal(adapter.lora_A.shape).astype(np.float32) * 0.2)
            adapter.lora_B = mx.array(rng.standard_normal(adapter.lora_B.shape).astype(np.float32) * 0.2)
            block.attn.to_q = adapter


@pytest.mark.parametrize("use_text_cache", [True, False])
def test_streamed_transformer_matches_the_resident_one(tmp_path, use_text_cache):
    # With the text cache the blocks run through text_forward and image_forward; without it through __call__.
    resident = Qwen21StreamSupport.transformer()
    Qwen21StreamSupport.save_sharded(resident, tmp_path / "transformer")
    expected = Qwen21StreamSupport.denoise(resident, use_text_cache)
    streamed = Qwen21StreamSupport.transformer()
    Qwen21StreamSupport.stream(tmp_path, streamed)
    assert all(isinstance(block, StreamedBlock) for block in streamed.transformer_blocks)
    assert not Qwen21StreamSupport.installed(streamed.transformer_blocks)
    actual = Qwen21StreamSupport.denoise(streamed, use_text_cache)
    assert not Qwen21StreamSupport.installed(streamed.transformer_blocks)
    for result, reference in zip(actual, expected):
        # The resident run packs the projections and compiles the step; the streamed run does neither.
        np.testing.assert_allclose(np.array(result), np.array(reference), atol=2e-5, rtol=2e-5)


def test_streamed_text_encoder_reads_the_original_one_file_layout(tmp_path):
    resident = Qwen21StreamSupport.text_encoder()
    Qwen21StreamSupport.save_original_text_encoder(resident, tmp_path / "text_encoder")
    Qwen21StreamSupport.save_sharded(Qwen21StreamSupport.transformer(), tmp_path / "transformer")
    streamed = Qwen21StreamSupport.text_encoder()
    Qwen21StreamSupport.stream(tmp_path, Qwen21StreamSupport.transformer(), streamed)
    assert streamed.layers[0].prefix == "model.language_model.layers.0."
    tokens = mx.array([[1, 5, 9, 2]])
    np.testing.assert_array_equal(np.array(streamed(tokens)), np.array(resident(tokens)))
    assert not Qwen21StreamSupport.installed(streamed.layers)


def test_a_runtime_lora_stays_in_memory_while_the_base_weights_stream(tmp_path):
    base = Qwen21StreamSupport.transformer()
    Qwen21StreamSupport.save_sharded(base, tmp_path / "transformer")
    resident = Qwen21StreamSupport.transformer()
    Qwen21StreamSupport.lora(resident, 5)
    expected = Qwen21StreamSupport.denoise(resident, use_text_cache=True)
    streamed = Qwen21StreamSupport.transformer()
    Qwen21StreamSupport.lora(streamed, 5)
    Qwen21StreamSupport.stream(tmp_path, streamed)
    block = streamed.transformer_blocks[0]
    assert block.module_key("attn.to_q.weight") == "attn.to_q.linear.weight"
    actual = Qwen21StreamSupport.denoise(streamed, use_text_cache=True)
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(np.array(result), np.array(reference), atol=2e-5, rtol=2e-5)
    # The adapter is resident; the base weight under it is the placeholder again.
    assert block.module.attn.to_q.lora_A.size > 1
    assert block.module.attn.to_q.linear.weight.size == 1


def test_streamed_weights_take_the_compute_precision_of_the_layer(tmp_path):
    resident = Qwen21StreamSupport.transformer()
    Qwen21StreamSupport.save_sharded(resident, tmp_path / "transformer")
    resident.apply_compute_precision(ComputePrecision(mx.float16))
    expected = Qwen21StreamSupport.denoise(resident, use_text_cache=True)
    streamed = Qwen21StreamSupport.transformer()
    streamed.apply_compute_precision(ComputePrecision(mx.float16))
    Qwen21StreamSupport.stream(tmp_path, streamed)
    actual = Qwen21StreamSupport.denoise(streamed, use_text_cache=True)
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(np.array(result), np.array(reference), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(quantize=8), "-q/--quantize"),
        (dict(lora_paths=["a.safetensors"], bake_lora=True), "baked LoRA"),
    ],
)
def test_combinations_that_would_drop_weights_are_refused_before_any_read(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Qwen21Initializer.init(
            model=object(),
            **{"quantize": None, **kwargs},
            model_path=None,
            model_config=None,
            ssd_stream=True,
        )


def test_the_w8a8_dit_is_refused(tmp_path, monkeypatch):
    checkpoint = tmp_path / "dit.safetensors"
    checkpoint.write_bytes(b"")
    monkeypatch.setenv("MFLUX_QWEN21_W8A8", str(checkpoint))
    with pytest.raises(ValueError, match="MFLUX_QWEN21_W8A8"):
        Qwen21Initializer.init(model=object(), quantize=None, model_path=None, model_config=None, ssd_stream=True)


def test_a_streamed_model_cannot_be_saved(tmp_path):
    model = QwenImage21.__new__(QwenImage21)
    model.ssd_stream = True
    with pytest.raises(ValueError, match="cannot be saved"):
        model.save_model(str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


class _Built(Exception):
    pass


@pytest.mark.parametrize(
    ("flags", "bake_lora"),
    [
        ([], False),
        (["--bake-lora"], True),
        (["--no-bake-lora"], False),
    ],
)
def test_the_cli_keeps_a_lora_as_a_runtime_layer_with_ssd_stream(tmp_path, monkeypatch, flags, bake_lora):
    from mflux.models.qwen21.cli import qwen21_generate

    lora = tmp_path / "a.safetensors"
    lora.write_bytes(b"")

    def build(**kwargs):
        raise _Built(kwargs)

    monkeypatch.setattr(qwen21_generate, "QwenImage21", build)
    argv = ["mflux-generate-qwen-2.1", "--prompt", "p", "--ssd-stream", "--lora", str(lora), *flags]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(_Built) as built:
        qwen21_generate.main()
    assert built.value.args[0]["ssd_stream"] is True
    assert built.value.args[0]["bake_lora"] is bake_lora


def test_the_cli_bakes_a_lora_by_default_without_ssd_stream(tmp_path, monkeypatch):
    from mflux.models.qwen21.cli import qwen21_generate

    lora = tmp_path / "a.safetensors"
    lora.write_bytes(b"")

    def build(**kwargs):
        raise _Built(kwargs)

    monkeypatch.setattr(qwen21_generate, "QwenImage21", build)
    monkeypatch.setattr("sys.argv", ["mflux-generate-qwen-2.1", "--prompt", "p", "--lora", str(lora)])
    with pytest.raises(_Built) as built:
        qwen21_generate.main()
    assert built.value.args[0]["ssd_stream"] is False
    assert built.value.args[0]["bake_lora"] is True
