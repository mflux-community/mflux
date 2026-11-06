import json
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import PIL.Image
import pytest
import torch
from mlx import nn
from torch import nn as tnn

from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config.model_config import AVAILABLE_MODELS, ModelConfig
from mflux.models.common.resolution.model_revisions import ModelRevisions
from mflux.models.flux2.cli import flux2_edit_generate, flux2_generate
from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mflux.models.flux2.model.flux2_text_encoder.qwen3_small_text_adapter import Qwen3SmallTextAdapter
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_text_encoder.small_text_encoder import (
    ADAPTER_REPO,
    SMALL_ENCODER_REPO,
    Flux2SmallTextEncoder,
)
from mflux.models.flux2.variants import Flux2Klein, Flux2KleinEdit
from mflux.models.flux2.weights.flux2_weight_definition import (
    Flux2KleinNoTextEncoderWeightDefinition,
    Flux2KleinWeightDefinition,
    Qwen3SmallTextEncoderWeightDefinition,
)
from tests.cli.helpers.command_fakes import reset_cli_globals
from tests.test_compute_precision import FakeFlux2Klein, FakeFlux2KleinEdit

N_SLICES, SLICE, HIDDEN, OUT, ATTN_DIM, HEADS, BLOCKS, MAX_LEN = 3, 16, 32, 24, 16, 2, 2, 12


class TorchPerLayerNorm(tnn.Module):
    def __init__(self, in_dim, n_slices):
        super().__init__()
        self.n, self.d = n_slices, in_dim // n_slices
        self.per = tnn.LayerNorm(self.d, elementwise_affine=False)
        self.all = tnn.LayerNorm(in_dim)

    def forward(self, x):
        b, n, _ = x.shape
        return self.all(self.per(x.view(b, n, self.n, self.d)).reshape(b, n, -1))


class TorchAttnBlock(tnn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n1 = tnn.LayerNorm(d_model)
        self.attn = tnn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.n2 = tnn.LayerNorm(d_model)
        self.ffn = tnn.Sequential(tnn.Linear(d_model, d_model * 4), tnn.GELU(approximate="tanh"), tnn.Linear(d_model * 4, d_model))  # fmt: skip

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ffn(self.n2(x))


class TorchAttnMixer(tnn.Module):
    def __init__(self, in_dim, out_dim, d_model, n_heads, blocks, max_len):
        super().__init__()
        self.norm = tnn.LayerNorm(in_dim)
        self.inp = tnn.Linear(in_dim, d_model)
        self.pos = tnn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = tnn.ModuleList([TorchAttnBlock(d_model, n_heads) for _ in range(blocks)])
        self.out = tnn.Linear(d_model, out_dim)

    def forward(self, x):
        h = self.inp(self.norm(x)) + self.pos[:, : x.shape[1]]
        for b in self.blocks:
            h = b(h)
        return self.out(h)


class TorchAdapter(tnn.Module):
    def __init__(self):
        super().__init__()
        in_dim = N_SLICES * SLICE
        self.mlp = tnn.Sequential(
            TorchPerLayerNorm(in_dim, N_SLICES),
            tnn.Linear(in_dim, HIDDEN),
            tnn.GELU(approximate="tanh"),
            tnn.Linear(HIDDEN, HIDDEN),
            tnn.GELU(approximate="tanh"),
            tnn.Linear(HIDDEN, OUT),
        )
        self.attn = TorchAttnMixer(in_dim, OUT, ATTN_DIM, HEADS, BLOCKS, MAX_LEN)

    def forward(self, x):
        return self.mlp(x) + self.attn(x)


def tiny_adapter() -> Qwen3SmallTextAdapter:
    return Qwen3SmallTextAdapter(
        in_dim=N_SLICES * SLICE,
        out_dim=OUT,
        hidden=HIDDEN,
        n_slices=N_SLICES,
        attn_dim=ATTN_DIM,
        attn_heads=HEADS,
        attn_blocks=BLOCKS,
        max_len=MAX_LEN,
    )


def random_torch_adapter() -> TorchAdapter:
    torch.manual_seed(0)
    adapter = TorchAdapter().eval()
    with torch.no_grad():
        for name, param in adapter.named_parameters():
            if name.endswith("norm.weight") or ".n1." in name or ".n2." in name or "mlp.0.all" in name:
                param.copy_(1 + 0.3 * torch.randn_like(param))
            elif name.endswith("bias"):
                param.copy_(0.2 * torch.randn_like(param))
            elif name == "attn.pos" or name.startswith("attn.out"):
                param.copy_(0.5 * torch.randn_like(param))
    return adapter


def raw_state(adapter: TorchAdapter) -> dict[str, mx.array]:
    return {k: mx.array(v.detach().numpy()) for k, v in adapter.state_dict().items()}


@pytest.mark.fast
class TestAdapter:
    def test_gelu_tanh_is_the_tanh_formula(self):
        x = mx.linspace(-5, 5, 101)
        expected = 0.5 * x * (1 + mx.tanh(mx.sqrt(2 / mx.pi) * (x + 0.044715 * x**3)))
        assert mx.allclose(nn.GELU(approx="tanh")(x), expected, atol=1e-6)

    @pytest.mark.parametrize("length", [MAX_LEN, 7])
    def test_matches_the_torch_reference(self, length):
        reference = random_torch_adapter()
        adapter = tiny_adapter()
        Flux2SmallTextEncoder.load_adapter_weights(adapter, raw_state(reference))
        x = torch.randn(2, length, N_SLICES * SLICE, generator=torch.Generator().manual_seed(1)) * 3
        with torch.no_grad():
            expected = reference(x).numpy()
            expected_mlp = reference.mlp(x).numpy()
        actual = adapter(mx.array(x.numpy()))
        assert actual.dtype == mx.float32
        np.testing.assert_allclose(np.array(actual), expected, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(np.array(adapter.apply_mlp(mx.array(x.numpy()))), expected_mlp, atol=1e-4, rtol=1e-4)

    def test_the_attention_branch_contributes(self):
        reference = random_torch_adapter()
        x = torch.randn(1, 5, N_SLICES * SLICE)
        with torch.no_grad():
            assert float((reference(x) - reference.mlp(x)).abs().max()) > 1e-2


@pytest.mark.fast
class TestCheckpointValidation:
    def metadata(self, **changes):
        return {**Flux2SmallTextEncoder.EXPECTED_METADATA, **changes}

    def test_accepts_the_published_metadata(self):
        Flux2SmallTextEncoder.validate_metadata(self.metadata())

    @pytest.mark.parametrize("field", sorted(Flux2SmallTextEncoder.EXPECTED_METADATA))
    def test_rejects_a_mismatched_field_naming_it(self, field):
        with pytest.raises(ValueError, match=field):
            Flux2SmallTextEncoder.validate_metadata(self.metadata(**{field: "bogus"}))

    def test_rejects_a_missing_field(self):
        metadata = self.metadata()
        del metadata["drop_first"]
        with pytest.raises(ValueError, match="drop_first"):
            Flux2SmallTextEncoder.validate_metadata(metadata)

    def test_a_missing_key_is_an_error(self):
        raw = raw_state(random_torch_adapter())
        del raw["attn.out.bias"]
        with pytest.raises(ValueError, match="attn.out.bias"):
            Flux2SmallTextEncoder.load_adapter_weights(tiny_adapter(), raw)

    def test_an_unexpected_key_is_an_error(self):
        raw = raw_state(random_torch_adapter())
        raw["attn.extra.weight"] = mx.zeros((2,))
        with pytest.raises(ValueError, match="attn.extra.weight"):
            Flux2SmallTextEncoder.load_adapter_weights(tiny_adapter(), raw)

    def test_a_wrong_shape_is_an_error(self):
        raw = raw_state(random_torch_adapter())
        raw["attn.pos"] = mx.zeros((1, MAX_LEN + 1, ATTN_DIM))
        with pytest.raises(ValueError, match="attn.pos"):
            Flux2SmallTextEncoder.load_adapter_weights(tiny_adapter(), raw)

    def test_in_proj_is_split_into_q_k_v_in_order(self):
        weight = mx.arange(12 * 4, dtype=mx.float32).reshape(12, 4)
        mapped = Flux2SmallTextEncoder.map_adapter_keys({"attn.blocks.0.attn.in_proj_weight": weight})
        assert mx.array_equal(mapped["attn.blocks.0.attn.query_proj.weight"], weight[:4])
        assert mx.array_equal(mapped["attn.blocks.0.attn.key_proj.weight"], weight[4:8])
        assert mx.array_equal(mapped["attn.blocks.0.attn.value_proj.weight"], weight[8:])

    def test_both_repos_are_pinned_to_the_published_commits(self):
        assert ModelRevisions.get(ADAPTER_REPO) == "26a3064438fb32f5d51b482cfb1f7fa642ff0b0d"
        assert ModelRevisions.get(SMALL_ENCODER_REPO) == "c1899de289a04d12100db370d81485cdf75e47ca"


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def tokenize(self, prompt, max_length=None, **kwargs):
        self.calls.append({"max_length": max_length, **kwargs})
        ids = mx.arange(max_length, dtype=mx.int32)[None] % 50
        mask = (mx.arange(max_length)[None] < 9).astype(mx.int32)
        return SimpleNamespace(input_ids=ids, attention_mask=mask)


def tiny_qwen() -> Qwen3TextEncoder:
    return Qwen3TextEncoder(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        head_dim=8,
    )


def tiny_small_encoder(length: int = 256) -> Flux2SmallTextEncoder:
    mx.random.seed(0)
    adapter = Qwen3SmallTextAdapter(
        in_dim=32, out_dim=24, hidden=16, n_slices=2, attn_dim=8, attn_heads=2, attn_blocks=1, max_len=length
    )
    metadata = {"student_layers": "1,3", "drop_first": "5", "len": str(length)}
    return Flux2SmallTextEncoder(encoder=tiny_qwen(), adapter=adapter, metadata=metadata)


@pytest.mark.fast
class TestPromptEncoding:
    def test_small_encoder_tokenizes_at_256_and_drops_the_first_five(self):
        tokenizer = RecordingTokenizer()
        embeds, text_ids = Flux2PromptEncoder.encode_prompt(
            prompt="a fox",
            tokenizer=tokenizer,
            text_encoder=tiny_small_encoder(),
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )
        assert tokenizer.calls == [{"max_length": 256, "keep_empty": True}]
        assert embeds.shape == (1, 251, 24)
        assert embeds.dtype == mx.bfloat16
        assert text_ids.shape == (1, 251, 4)
        assert text_ids[0, :, 3].tolist() == list(range(5, 256))

    def test_ids_are_repeated_per_image(self):
        embeds, text_ids = Flux2PromptEncoder.encode_prompt(
            prompt="a fox",
            tokenizer=RecordingTokenizer(),
            text_encoder=tiny_small_encoder(),
            num_images_per_prompt=2,
        )
        assert embeds.shape[0] == 2 and text_ids.shape[0] == 2

    def test_native_path_still_uses_the_callers_length_and_all_tokens(self):
        tokenizer = RecordingTokenizer()
        embeds, text_ids = Flux2PromptEncoder.encode_prompt(
            prompt="a fox",
            tokenizer=tokenizer,
            text_encoder=tiny_qwen(),
            max_sequence_length=40,
            text_encoder_out_layers=(1, 2, 3),
        )
        assert tokenizer.calls == [{"max_length": 40}]
        assert embeds.shape == (1, 40, 48)
        assert text_ids[0, :, 3].tolist() == list(range(40))

    def test_an_empty_prompt_keeps_its_padded_length(self):
        from mflux.models.common.tokenizer import LanguageTokenizer

        class Hf:
            def apply_chat_template(self, messages, **kwargs):
                return "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>"

            def __call__(self, prompts, max_length, **kwargs):
                return {"input_ids": np.zeros((1, max_length), dtype=np.int32), "attention_mask": np.ones((1, max_length), dtype=np.int32)}  # fmt: skip

        tokenizer = LanguageTokenizer(Hf(), use_chat_template=True)
        assert tokenizer.tokenize("", max_length=256).input_ids.shape == (1, 0)
        assert tokenizer.tokenize("", max_length=256, keep_empty=True).input_ids.shape == (1, 256)


@pytest.mark.fast
class TestInitializer:
    def test_an_unsupported_model_is_rejected_before_any_weight_loads(self, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("weights must not load")

        monkeypatch.setattr(Flux2Initializer, "_load_weights", fail)
        monkeypatch.setattr(Flux2Initializer, "_load_weights_without_text_encoder", fail)
        for config in (AVAILABLE_MODELS["iris-3b"], AVAILABLE_MODELS["z-image-turbo"]):
            with pytest.raises(ValueError, match="only works with FLUX.2-klein-4B"):
                Flux2Klein(model_config=config, small_text_encoder=True, small_decoder=False)
            with pytest.raises(ValueError, match="only works with FLUX.2-klein-4B"):
                Flux2KleinEdit(model_config=config, small_text_encoder=True, small_decoder=False)

    def test_4b_configs_are_supported(self):
        assert Flux2Initializer.supports_small_text_encoder(ModelConfig.flux2_klein_4b())
        assert Flux2Initializer.supports_small_text_encoder(ModelConfig.flux2_klein_base_4b())
        assert not Flux2Initializer.supports_small_text_encoder(AVAILABLE_MODELS["iris-3b"])

    def test_the_filtered_definition_skips_the_klein_text_encoder(self):
        names = [c.name for c in Flux2KleinNoTextEncoderWeightDefinition.get_components()]
        patterns = Flux2KleinNoTextEncoderWeightDefinition.get_download_patterns()
        assert names == ["vae", "transformer"]
        assert not [p for p in patterns if p.startswith("text_encoder")]
        assert "tokenizer/**" in patterns and "transformer/*.safetensors" in patterns

    def test_the_default_definition_is_unchanged(self):
        assert [c.name for c in Flux2KleinWeightDefinition.get_components()] == ["vae", "transformer", "text_encoder"]
        patterns = Flux2KleinWeightDefinition.get_download_patterns()
        assert "text_encoder/*.safetensors" in patterns and "text_encoder/*.json" in patterns

    def test_the_small_encoder_definition_names_the_pinned_files(self):
        assert Qwen3SmallTextEncoderWeightDefinition.get_download_patterns() == ["model.safetensors", "config.json"]
        (component,) = Qwen3SmallTextEncoderWeightDefinition.get_components()
        assert component.hf_subdir == "" and component.num_layers == 28

    def test_the_small_path_never_builds_the_native_encoder_or_quantizes_the_adapter(self):
        model = SimpleNamespace(
            small_text_encoder=True, vae="vae", transformer="transformer", model_config=ModelConfig.flux2_klein_4b()
        )
        assert Flux2Initializer._weighted_models(model) == {"vae": "vae", "transformer": "transformer"}


@pytest.mark.fast
class TestMetadataAndSaving:
    @pytest.mark.parametrize("cls", [Flux2Klein, Flux2KleinEdit])
    @pytest.mark.parametrize(("flag", "expected"), [(True, {"small_text_encoder": True}), (False, {})])
    def test_generation_parameters_record_the_flag_only_when_on(self, cls, flag, expected):
        model = SimpleNamespace(compute_precision=ComputePrecision(None), small_text_encoder=flag, small_decoder=False)
        assert cls._generation_parameters(model) == expected

    def test_it_merges_with_compute_precision(self):
        model = SimpleNamespace(
            compute_precision=ComputePrecision(mx.float16), small_text_encoder=True, small_decoder=False
        )
        assert Flux2Klein._generation_parameters(model) == {"compute_precision": "float16", "small_text_encoder": True}

    def test_save_model_refuses(self, tmp_path):
        model = SimpleNamespace(small_text_encoder=True, bits=None)
        with pytest.raises(ValueError, match="small_text_encoder"):
            Flux2Klein.save_model(model, str(tmp_path / "saved"))
        assert not (tmp_path / "saved").exists()


@pytest.mark.fast
class TestCommandLine:
    COMMANDS = [
        ("mflux-generate-flux2", flux2_generate, "Flux2Klein", FakeFlux2Klein, []),
        (
            "mflux-generate-flux2-edit",
            flux2_edit_generate,
            "Flux2KleinEdit",
            FakeFlux2KleinEdit,
            ["--image-paths", "ref.png"],
        ),  # fmt: skip
    ]

    @pytest.mark.parametrize(("command", "module", "attribute", "fake", "extra"), COMMANDS)
    @pytest.mark.parametrize(("flag", "expected"), [([], False), (["--small-text-encoder"], True)])
    def test_the_flag_reaches_the_model(self, monkeypatch, tmp_path, command, module, attribute, fake, extra, flag, expected):  # fmt: skip
        fake.instances.clear()
        monkeypatch.setattr(module, attribute, fake)
        reset_cli_globals(monkeypatch)
        PIL.Image.new("RGB", (64, 64)).save(tmp_path / "ref.png")
        extra = [str(tmp_path / "ref.png") if e == "ref.png" else e for e in extra]
        argv = [command, "--prompt-file", str(tmp_path / "missing.txt"), *extra, *flag]
        monkeypatch.setattr(sys, "argv", argv)
        module.main()
        assert fake.instances[0].init_kwargs["small_text_encoder"] is expected

    @pytest.mark.parametrize(("command", "module", "attribute", "fake", "extra"), COMMANDS)
    def test_an_unsupported_model_config_is_rejected_by_the_parser(self, monkeypatch, capsys, command, module, attribute, fake, extra):  # fmt: skip
        fake.instances.clear()
        monkeypatch.setattr(module, attribute, fake)
        reset_cli_globals(monkeypatch)
        monkeypatch.setattr(Flux2Initializer, "supports_small_text_encoder", staticmethod(lambda config: False))
        argv = [command, "--model", "flux2-klein-9b", "--prompt", "x", *extra, "--small-text-encoder"]
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            module.main()
        assert "only works with FLUX.2-klein-4B" in capsys.readouterr().err
        assert fake.instances == []

    @pytest.mark.parametrize(("module", "extra"), [(flux2_generate, []), (flux2_edit_generate, ["--image-paths", "ref.png"])])  # fmt: skip
    @pytest.mark.parametrize(
        ("recorded", "argv_flag", "expected"),
        [
            (True, [], True),
            (True, ["--no-small-text-encoder"], False),
            (False, ["--small-text-encoder"], True),
            (None, [], False),
        ],  # fmt: skip
    )
    def test_a_sidecar_replays_the_flag(self, monkeypatch, tmp_path, module, extra, recorded, argv_flag, expected):
        sidecar = tmp_path / "image.metadata.json"
        sidecar.write_text(json.dumps({"prompt": "a fox", "seed": 1, "steps": 4} | ({} if recorded is None else {"small_text_encoder": recorded})))  # fmt: skip
        monkeypatch.setattr(sys, "argv", ["cmd", "--config-from-metadata", str(sidecar), *extra, *argv_flag])
        assert module.build_parser().parse_args().small_text_encoder is expected

    def test_a_sidecar_with_null_replays_as_off(self, monkeypatch, tmp_path):
        sidecar = tmp_path / "image.metadata.json"
        sidecar.write_text(json.dumps({"prompt": "a fox", "seed": 1, "steps": 4, "small_text_encoder": None}))
        monkeypatch.setattr(sys, "argv", ["cmd", "--config-from-metadata", str(sidecar)])
        assert flux2_generate.build_parser().parse_args().small_text_encoder is False

    @pytest.mark.parametrize("recorded", ["false", "true", 1])
    def test_a_recorded_value_that_is_not_a_boolean_stops_the_replay(self, monkeypatch, tmp_path, capsys, recorded):
        # bool("false") is True: a hand-edited sidecar must not switch the text encoder.
        sidecar = tmp_path / "image.metadata.json"
        sidecar.write_text(json.dumps({"prompt": "a fox", "seed": 1, "steps": 4, "small_text_encoder": recorded}))
        monkeypatch.setattr(sys, "argv", ["cmd", "--config-from-metadata", str(sidecar)])
        with pytest.raises(SystemExit) as exit_info:
            flux2_generate.build_parser().parse_args()
        assert exit_info.value.code == 2
        assert "small_text_encoder in --config-from-metadata must be true or false" in capsys.readouterr().err
