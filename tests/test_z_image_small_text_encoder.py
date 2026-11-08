import json
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx import nn
from mlx.utils import tree_flatten
from torch import nn as tnn

from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.model_revisions import ModelRevisions
from mflux.models.z_image import z_image_initializer
from mflux.models.z_image.cli import z_image_turbo_generate
from mflux.models.z_image.model.z_image_text_encoder import small_text_encoder as small_module
from mflux.models.z_image.model.z_image_text_encoder.small_text_encoder import (
    SMALL_ENCODER_REPO,
    ZImageSmallTextEncoder,
)
from mflux.models.z_image.model.z_image_text_encoder.student_adapter import ZImageStudentAdapter
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
from mflux.models.z_image.variants.z_image import ZImage
from mflux.models.z_image.weights.z_image_weight_definition import (
    ZImageNoTextEncoderWeightDefinition,
    ZImageWeightDefinition,
)
from mflux.models.z_image.z_image_initializer import ZImageInitializer
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals

S_DIM, T_DIM, DIM, HEADS, BLOCKS, FF_MULT = 24, 20, 16, 2, 2, 4

PUBLISHED_CONFIG = {
    "adapter_dim": 1024,
    "adapter_heads": 8,
    "adapter_blocks": 2,
    "adapter_ff_mult": 4,
    "hs_tap_index": -2,
    "teacher_hidden_size": 2560,
    "student_hidden_size": 2048,
    "student_config_dict": {
        "hidden_size": 2048,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 6144,
        "head_dim": 128,
        "rope_theta": 10000.0,
    },
}


class TorchXAttnBlock(tnn.Module):
    def __init__(self, dim, heads, ff_mult):
        super().__init__()
        self.norm_q = tnn.LayerNorm(dim)
        self.norm_kv = tnn.LayerNorm(dim)
        self.attn = tnn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_ff = tnn.LayerNorm(dim)
        self.ff = tnn.Sequential(tnn.Linear(dim, ff_mult * dim), tnn.GELU(), tnn.Dropout(0.0), tnn.Linear(ff_mult * dim, dim), tnn.Dropout(0.0))  # fmt: skip

    def forward(self, q, kv, key_padding_mask):
        kvn = self.norm_kv(kv)
        q = q + self.attn(self.norm_q(q), kvn, kvn, key_padding_mask=key_padding_mask)[0]
        return q + self.ff(self.norm_ff(q))


class TorchAdapter(tnn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = tnn.Linear(S_DIM, DIM)
        self.kv_proj = tnn.Linear(S_DIM, DIM)
        self.blocks = tnn.ModuleList([TorchXAttnBlock(DIM, HEADS, FF_MULT) for _ in range(BLOCKS)])
        self.proj_out = tnn.Linear(DIM, T_DIM)

    def forward(self, student_hs, mask):
        q = self.q_proj(student_hs)
        kv = self.kv_proj(student_hs)
        for block in self.blocks:
            q = block(q, kv, key_padding_mask=~mask.bool())
        return self.proj_out(q).masked_fill(~mask[..., None].bool(), 0)


def random_torch_adapter() -> TorchAdapter:
    torch.manual_seed(0)
    adapter = TorchAdapter().eval()
    with torch.no_grad():
        for param in adapter.parameters():
            param.copy_(param + 0.3 * torch.randn_like(param))
    return adapter


def raw_adapter_state(adapter: TorchAdapter, prefix: str = "") -> dict[str, mx.array]:
    return {prefix + k: mx.array(v.detach().numpy()) for k, v in adapter.state_dict().items()}


def tiny_adapter() -> ZImageStudentAdapter:
    return ZImageStudentAdapter(
        student_dim=S_DIM, teacher_dim=T_DIM, dim=DIM, heads=HEADS, blocks=BLOCKS, ff_mult=FF_MULT
    )


def tiny_student() -> TextEncoder:
    return TextEncoder(
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=128,
        head_dim=32,
        rope_theta=10000.0,
    )


def tiny_adapter_for_student() -> ZImageStudentAdapter:
    return ZImageStudentAdapter(student_dim=64, teacher_dim=48, dim=64, heads=2, blocks=1, ff_mult=2)


def student_raw(student: TextEncoder) -> dict[str, mx.array]:
    return {f"student.model.{k}": v for k, v in tree_flatten(student.parameters()) if not k.endswith("inv_freq")}


@pytest.mark.fast
class TestAdapter:
    @pytest.mark.parametrize("lengths", [(8, 5), (8, 8)])
    def test_matches_the_torch_reference_and_zeroes_padded_rows(self, lengths):
        reference = random_torch_adapter()
        adapter = tiny_adapter()
        ZImageSmallTextEncoder.load_adapter_weights(adapter, raw_adapter_state(reference, "adapter."))
        x = torch.randn(2, 8, S_DIM, generator=torch.Generator().manual_seed(1)) * 2
        mask = torch.tensor([[1] * n + [0] * (8 - n) for n in lengths])
        with torch.no_grad():
            expected = reference(x, mask).numpy()
        actual = adapter(mx.array(x.numpy()), mx.array(mask.numpy()))
        assert actual.dtype == mx.float32
        np.testing.assert_allclose(np.array(actual), expected, atol=1e-5, rtol=1e-5)
        for row, n in enumerate(lengths):
            assert np.all(np.array(actual[row, n:]) == 0)
        assert np.abs(np.array(actual[1, : lengths[1]])).max() > 0

    def test_gelu_is_the_erf_form(self):
        x = mx.array([0.5, 1.0, -1.5])
        erf = 0.5 * x * (1 + mx.erf(x / mx.sqrt(mx.array(2.0))))
        tanh = nn.gelu_approx(x)
        assert float(mx.abs(erf - tanh).max()) > 1e-4
        reference = random_torch_adapter()
        block_in = mx.array(np.linspace(-3, 3, 24, dtype=np.float32)[None])
        torch_gelu = tnn.GELU()(torch.from_numpy(np.array(block_in))).numpy()
        np.testing.assert_allclose(np.array(nn.gelu(block_in)), torch_gelu, atol=1e-6)
        assert reference is not None


@pytest.mark.fast
class TestKeyMapping:
    def test_in_proj_is_split_into_q_k_v_in_order(self):
        weight = mx.arange(12 * 4, dtype=mx.float32).reshape(12, 4)
        bias = mx.arange(12, dtype=mx.float32)
        mapped = ZImageSmallTextEncoder.map_adapter_keys(
            {"blocks.0.attn.in_proj_weight": weight, "blocks.0.attn.in_proj_bias": bias}
        )
        assert mx.array_equal(mapped["blocks.0.attn.query_proj.weight"], weight[:4])
        assert mx.array_equal(mapped["blocks.0.attn.key_proj.weight"], weight[4:8])
        assert mx.array_equal(mapped["blocks.0.attn.value_proj.weight"], weight[8:])
        assert mx.array_equal(mapped["blocks.0.attn.value_proj.bias"], bias[8:])

    def test_ff_indices_map(self):
        mapped = ZImageSmallTextEncoder.map_adapter_keys(
            {"blocks.1.ff.0.weight": mx.zeros((1,)), "blocks.1.ff.3.bias": mx.zeros((1,))}
        )
        assert set(mapped) == {"blocks.1.ff_in.weight", "blocks.1.ff_out.bias"}

    def test_a_missing_key_is_an_error(self):
        raw = raw_adapter_state(random_torch_adapter(), "adapter.")
        del raw["adapter.proj_out.bias"]
        with pytest.raises(ValueError, match="proj_out.bias"):
            ZImageSmallTextEncoder.load_adapter_weights(tiny_adapter(), raw)

    def test_an_unexpected_key_is_an_error(self):
        raw = raw_adapter_state(random_torch_adapter(), "adapter.")
        raw["adapter.extra.weight"] = mx.zeros((2,))
        with pytest.raises(ValueError, match="extra.weight"):
            ZImageSmallTextEncoder.load_adapter_weights(tiny_adapter(), raw)

    def test_a_wrong_shape_is_an_error(self):
        raw = raw_adapter_state(random_torch_adapter(), "adapter.")
        raw["adapter.proj_out.weight"] = mx.zeros((T_DIM + 1, DIM))
        with pytest.raises(ValueError, match="proj_out.weight"):
            ZImageSmallTextEncoder.load_adapter_weights(tiny_adapter(), raw)

    def test_student_keys_load_with_the_student_model_prefix(self):
        source = tiny_student()
        mx.eval(source.parameters())
        target = tiny_student()
        ZImageSmallTextEncoder.load_student_weights(target, student_raw(source))
        ids = mx.arange(6, dtype=mx.int32)[None]
        # weights load as bf16, the checkpoint's dtype
        assert mx.array_equal(target.layers[1].self_attn.q_norm.weight, source.layers[1].self_attn.q_norm.weight.astype(mx.bfloat16))  # fmt: skip
        assert mx.allclose(source(ids).astype(mx.float32), target(ids).astype(mx.float32), atol=5e-2)

    def test_student_missing_unexpected_and_wrong_shape_are_errors(self):
        raw = student_raw(tiny_student())
        missing = {k: v for k, v in raw.items() if k != "student.model.norm.weight"}
        with pytest.raises(ValueError, match=r"norm\.weight"):
            ZImageSmallTextEncoder.load_student_weights(tiny_student(), missing)
        with pytest.raises(ValueError, match=r"lm_head\.weight"):
            ZImageSmallTextEncoder.load_student_weights(
                tiny_student(), {**raw, "student.model.lm_head.weight": mx.zeros((2,))}
            )
        with pytest.raises(ValueError, match=r"embed_tokens\.weight"):
            ZImageSmallTextEncoder.load_student_weights(
                tiny_student(), {**raw, "student.model.embed_tokens.weight": mx.zeros((3, 3))}
            )


@pytest.mark.fast
class TestConfigAndBuild:
    def config(self, **changes):
        return {**PUBLISHED_CONFIG, **changes}

    def test_accepts_the_published_config(self):
        ZImageSmallTextEncoder.validate_config(PUBLISHED_CONFIG)

    @pytest.mark.parametrize("field", sorted(ZImageSmallTextEncoder.EXPECTED_CONFIG))
    def test_rejects_a_mismatched_field_naming_it(self, field):
        with pytest.raises(ValueError, match=field):
            ZImageSmallTextEncoder.validate_config(self.config(**{field: 7}))

    @pytest.mark.parametrize("field", sorted(ZImageSmallTextEncoder.EXPECTED_STUDENT_CONFIG))
    def test_rejects_a_mismatched_student_field_naming_it(self, field):
        student = {**PUBLISHED_CONFIG["student_config_dict"], field: 7}
        with pytest.raises(ValueError, match=field):
            ZImageSmallTextEncoder.validate_config(self.config(student_config_dict=student))

    def test_the_student_has_the_measured_dims_and_rope_base(self):
        student = ZImageSmallTextEncoder.build_student()
        assert student.rotary_emb.inv_freq.shape == (64,)
        # inv_freq[1] = base ** (-2/128)
        assert float(student.rotary_emb.inv_freq[1]) == pytest.approx(10000.0 ** (-2 / 128), rel=1e-6)
        assert len(student.layers) == 28
        assert student.embed_tokens.weight.shape == (151936, 2048)
        attn = student.layers[0].self_attn
        assert attn.q_proj.weight.shape == (16 * 128, 2048)
        assert attn.k_proj.weight.shape == (8 * 128, 2048)
        assert student.layers[0].mlp.gate_proj.weight.shape == (6144, 2048)
        assert student.norm.eps == 1e-6

    def test_the_adapter_has_the_published_dims(self):
        adapter = ZImageSmallTextEncoder.build_adapter()
        assert adapter.q_proj.weight.shape == (1024, 2048)
        assert adapter.proj_out.weight.shape == (2560, 1024)
        assert len(adapter.blocks) == 2
        assert adapter.blocks[0].ff_in.weight.shape == (4096, 1024)
        assert adapter.blocks[0].norm_q.eps == 1e-5

    def test_the_repo_is_pinned_to_the_published_commit(self):
        assert ModelRevisions.get(SMALL_ENCODER_REPO) == "b545401c153e04b85d3383f3a2ac789aec44b61a"


def write_tiny_checkpoint(tmp_path):
    student = tiny_student()
    adapter = tiny_adapter_for_student()
    torch_like = {f"adapter.{k}": v for k, v in tree_flatten(adapter.parameters())}
    # undo the MLX-side renames so the file looks like the published one
    renamed = {}
    for key, value in torch_like.items():
        if ".attn.query_proj." in key:
            continue
        renamed[key.replace(".ff_in.", ".ff.0.").replace(".ff_out.", ".ff.3.")] = value
    for i in range(len(adapter.blocks)):
        for suffix in ("weight", "bias"):
            parts = [
                torch_like[f"adapter.blocks.{i}.attn.{p}.{suffix}"] for p in ("query_proj", "key_proj", "value_proj")
            ]
            for p in ("query_proj", "key_proj", "value_proj"):
                renamed.pop(f"adapter.blocks.{i}.attn.{p}.{suffix}", None)
            renamed[f"adapter.blocks.{i}.attn.in_proj_{suffix}"] = mx.concatenate(parts, axis=0)
    path = tmp_path / "model.safetensors"
    mx.save_safetensors(str(path), {**student_raw(student), **renamed})
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(PUBLISHED_CONFIG))
    return config_path, path


@pytest.mark.fast
class TestLoad:
    def patch(self, monkeypatch, tmp_path):
        config_path, weights_path = write_tiny_checkpoint(tmp_path)
        calls = []

        def fake_download(repo, filename, **kwargs):
            calls.append((repo, filename, kwargs))
            return str(config_path if filename.endswith("config.json") else weights_path)

        monkeypatch.setattr(small_module, "hf_hub_download", fake_download)
        monkeypatch.setattr(ZImageSmallTextEncoder, "build_student", staticmethod(tiny_student))
        monkeypatch.setattr(ZImageSmallTextEncoder, "build_adapter", staticmethod(tiny_adapter_for_student))
        return calls

    def test_load_asks_for_exactly_the_two_files_at_the_pinned_revision(self, monkeypatch, tmp_path):
        calls = self.patch(monkeypatch, tmp_path)
        encoder = ZImageSmallTextEncoder.load()
        assert calls == [
            (SMALL_ENCODER_REPO, "text_encoder/config.json", {"revision": "b545401c153e04b85d3383f3a2ac789aec44b61a"}),
            (
                SMALL_ENCODER_REPO,
                "text_encoder/model.safetensors",
                {"revision": "b545401c153e04b85d3383f3a2ac789aec44b61a"},
            ),
        ]
        ids = mx.arange(6, dtype=mx.int32)[None]
        mask = mx.array([[1, 1, 1, 1, 0, 0]])
        out = encoder(ids, mask)
        assert out.shape == (1, 6, 48) and out.dtype == ModelConfig.precision
        assert np.all(np.array(out[0, 4:].astype(mx.float32)) == 0)

    def test_quantizing_touches_the_student_and_never_the_adapter(self, monkeypatch, tmp_path):
        self.patch(monkeypatch, tmp_path)
        encoder = ZImageSmallTextEncoder.load(bits=8)
        assert isinstance(encoder.student.layers[0].self_attn.q_proj, nn.QuantizedLinear)
        assert type(encoder.adapter.q_proj) is nn.Linear
        assert type(encoder.adapter.blocks[0].attn.query_proj) is nn.Linear
        assert encoder.adapter.q_proj.weight.dtype == mx.float32

    def test_a_download_error_is_friendly(self, monkeypatch):
        def fail(*args, **kwargs):
            raise OSError("offline")

        monkeypatch.setattr(small_module, "hf_hub_download", fail)
        with pytest.raises(RuntimeError, match="3.6 GB"):
            ZImageSmallTextEncoder.load()


@pytest.mark.fast
class TestWeightDefinition:
    def test_the_filtered_definition_skips_the_text_encoder(self):
        names = [c.name for c in ZImageNoTextEncoderWeightDefinition.get_components()]
        patterns = ZImageNoTextEncoderWeightDefinition.get_download_patterns()
        assert names == ["vae", "transformer"]
        assert not [p for p in patterns if p.startswith("text_encoder")]
        assert "tokenizer/*" in patterns and "transformer/*.safetensors" in patterns

    def test_the_default_definition_is_unchanged(self):
        assert [c.name for c in ZImageWeightDefinition.get_components()] == ["vae", "transformer", "text_encoder"]
        patterns = ZImageWeightDefinition.get_download_patterns()
        assert "text_encoder/*.safetensors" in patterns and "text_encoder/*.json" in patterns


@pytest.mark.fast
class TestInitializer:
    def test_only_turbo_is_supported(self):
        assert ZImageInitializer.supports_small_text_encoder(ModelConfig.z_image_turbo())
        assert not ZImageInitializer.supports_small_text_encoder(ModelConfig.z_image())
        assert not ZImageInitializer.supports_small_text_encoder(ModelConfig.z_image_turbo_controlnet_union_2_1())

    @pytest.mark.parametrize("config", [ModelConfig.z_image(), ModelConfig.z_image_turbo_controlnet_union_2_1()])
    def test_other_configs_are_rejected_before_any_weight_loads(self, monkeypatch, config):
        def fail(*args, **kwargs):
            raise AssertionError("weights must not load")

        monkeypatch.setattr(ZImageInitializer, "_load_weights", fail)
        monkeypatch.setattr(ZImageInitializer, "_load_weights_without_text_encoder", fail)
        monkeypatch.setattr(ZImageSmallTextEncoder, "load", fail)
        with pytest.raises(ValueError, match="only works with Z-Image-Turbo"):
            ZImage(model_config=config, small_text_encoder=True)

    def test_the_small_path_never_builds_the_native_encoder_or_quantizes_the_adapter(self, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("the native text encoder must not be built")

        monkeypatch.setattr(z_image_initializer, "TextEncoder", fail)
        monkeypatch.setattr(z_image_initializer, "VAE", lambda: SimpleNamespace())
        monkeypatch.setattr(z_image_initializer, "ZImageTransformer", lambda: "transformer")
        model = SimpleNamespace(small_text_encoder=True)
        ZImageInitializer._init_models(model)
        assert model.text_encoder is None
        assert ZImageInitializer._weighted_models(model) == {"vae": model.vae, "transformer": "transformer"}

    def test_the_small_path_loads_the_filtered_weights_then_the_small_encoder(self, monkeypatch):
        events = []
        loaded = SimpleNamespace(dense_component=lambda name: None)
        monkeypatch.setattr(
            ZImageInitializer,
            "_load_weights_without_text_encoder",
            staticmethod(lambda path: events.append("weights") or loaded),
        )
        monkeypatch.setattr(ZImageInitializer, "_init_tokenizers", staticmethod(lambda model, path: None))
        monkeypatch.setattr(ZImageInitializer, "_init_models", staticmethod(lambda model: None))
        monkeypatch.setattr(ZImageInitializer, "_apply_lora", staticmethod(lambda *args: None))

        def apply(model, weights, quantize):
            model.bits = quantize
            events.append(("apply", quantize))

        monkeypatch.setattr(ZImageInitializer, "_apply_weights", staticmethod(apply))
        monkeypatch.setattr(
            ZImageSmallTextEncoder, "load", staticmethod(lambda bits=None: events.append(("small", bits)) or "small")
        )
        model = SimpleNamespace(transformer=SimpleNamespace(set_float32=lambda f: None))
        ZImageInitializer.init(model, ModelConfig.z_image_turbo(), quantize=4, small_text_encoder=True)
        assert events == ["weights", ("apply", 4), ("small", 4)]
        assert model.text_encoder == "small" and model.small_text_encoder is True

    def test_the_weight_loader_picks_the_filtered_definition(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            z_image_initializer.WeightLoader,
            "load",
            lambda weight_definition, model_path: seen.append(weight_definition),
        )
        ZImageInitializer._load_weights_without_text_encoder("p")
        ZImageInitializer._load_weights("p")
        assert seen == [ZImageNoTextEncoderWeightDefinition, ZImageWeightDefinition]


@pytest.mark.fast
class TestMetadataAndSaving:
    @pytest.mark.parametrize(("flag", "expected"), [(True, {"small_text_encoder": True}), (False, {})])
    def test_generation_parameters_record_the_flag_only_when_on(self, flag, expected):
        model = SimpleNamespace(
            float32=False, compute_precision=ComputePrecision(None), small_text_encoder=flag, vae=None
        )
        parameters = ZImage._generation_parameters(model)
        assert {k: v for k, v in parameters.items() if k == "small_text_encoder"} == expected

    def test_it_merges_with_compute_precision(self):
        model = SimpleNamespace(
            float32=False, compute_precision=ComputePrecision(mx.float16), small_text_encoder=True, vae=None
        )
        parameters = ZImage._generation_parameters(model)
        assert parameters["compute_precision"] == "float16" and parameters["small_text_encoder"] is True

    def test_save_model_refuses(self, tmp_path):
        model = SimpleNamespace(small_text_encoder=True, bits=None)
        with pytest.raises(ValueError, match="small_text_encoder"):
            ZImage.save_model(model, str(tmp_path / "saved"))
        assert not (tmp_path / "saved").exists()


class FakeZImage(FakeModel):
    real = ZImage


@pytest.mark.fast
class TestCommandLine:
    @pytest.fixture(autouse=True)
    def isolated(self, monkeypatch):
        FakeZImage.instances.clear()
        monkeypatch.setattr(z_image_turbo_generate, "ZImage", FakeZImage)
        reset_cli_globals(monkeypatch)
        yield
        FakeZImage.instances.clear()

    @pytest.mark.parametrize(("flag", "expected"), [([], False), (["--small-text-encoder"], True)])
    def test_the_flag_reaches_the_model(self, monkeypatch, tmp_path, flag, expected):
        argv = ["mflux-generate-z-image-turbo", "--prompt-file", str(tmp_path / "missing.txt"), *flag]
        monkeypatch.setattr(sys, "argv", argv)
        z_image_turbo_generate.main()
        assert FakeZImage.instances[0].init_kwargs["small_text_encoder"] is expected

    @pytest.mark.parametrize(
        ("recorded", "argv_flag", "expected"),
        [
            (True, [], True),
            (True, ["--no-small-text-encoder"], False),
            (False, ["--small-text-encoder"], True),
            (None, [], False),
        ],
    )
    def test_a_sidecar_replays_the_flag(self, monkeypatch, tmp_path, recorded, argv_flag, expected):
        sidecar = tmp_path / "image.metadata.json"
        extra = {} if recorded is None else {"small_text_encoder": recorded}
        sidecar.write_text(json.dumps({"prompt": "a fox", "seed": 1, "steps": 9} | extra))
        monkeypatch.setattr(sys, "argv", ["cmd", "--config-from-metadata", str(sidecar), *argv_flag])
        assert z_image_turbo_generate.build_parser().parse_args().small_text_encoder is expected

    def test_the_help_names_the_repo(self):
        parser = z_image_turbo_generate.build_parser()
        action = next(a for a in parser._actions if "--small-text-encoder" in a.option_strings)
        assert "[EXPERIMENTAL]" not in action.help
        assert "SearchingMan/Z-Image-Turbo-student-adapter" in action.help
