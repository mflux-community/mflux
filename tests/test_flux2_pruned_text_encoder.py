import re
import sys
from types import SimpleNamespace

import httpx
import mlx.core as mx
import pytest
from huggingface_hub.errors import RepositoryNotFoundError
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.resolution.model_revisions import ModelRevisions
from mflux.models.flux2.cli import flux2_edit_generate, flux2_generate
from mflux.models.flux2.flux2_initializer import SMALL_TEXT_ENCODER_UNSUPPORTED, Flux2Initializer
from mflux.models.flux2.model.flux2_text_encoder import pruned_text_encoder as pruned_module
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mflux.models.flux2.model.flux2_text_encoder.pruned_text_encoder import PRUNED_ENCODER_REPO, Flux2PrunedTextEncoder
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_text_encoder.small_text_encoder import Flux2SmallTextEncoder
from mflux.models.flux2.variants import Flux2Klein
from tests.cli.helpers.command_fakes import reset_cli_globals

REVISION = "89ff813c44e7ea58564dc79f58c3d2128d007a50"

# config.json and pruning_metadata.json of SearchingMan/FLUX.2-klein-9B-Text-Encoder-Pruned-5.1B at REVISION.
REAL_CONFIG = {
    "architectures": ["Qwen3Model"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "dtype": "bfloat16",
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
    "layer_types": None,
    "max_position_embeddings": 40960,
    "max_window_layers": 36,
    "model_type": "qwen3",
    "num_attention_heads": 32,
    "num_hidden_layers": 26,
    "num_key_value_heads": 8,
    "pad_token_id": None,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
    "sliding_window": None,
    "tie_word_embeddings": False,
    "transformers_version": "5.7.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}
REAL_CONFIG["layer_types"] = ["full_attention"] * 26
REAL_METADATA = {
    "base_model": "black-forest-labs/FLUX.2-klein-9B",
    "artifact_source": "downloaded_remote_runs/tail_plus_merge_ffn_gqa_remove_28-29-30-31-32-33-34-35_20260703_114826/artifact_pruned_recovered_text_encoder",
    "text_encoder_out_layers": [9, 17, 25],
    "exported_num_hidden_layers": 26,
    "kept_original_layers": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
    ],
    "deleted_original_layers": [10, 19, 28, 29, 30, 31, 32, 33, 34, 35],
    "lm_head_dropped": True,
    "final_norm_identity": False,
    "ffn_keep_by_exported_layer": {"17": 8192, "18": 8192, "19": 8192, "20": 8192, "21": 8192, "22": 8192, "23": 8192},
    "head_groups_by_exported_layer": {
        "9": 4,
        "10": 6,
        "11": 5,
        "12": 6,
        "13": 6,
        "14": 6,
        "15": 6,
        "17": 5,
        "18": 4,
        "19": 4,
        "21": 6,
        "22": 6,
        "23": 4,
    },
    "ffn_importance_method": "activation_wanda_l1",
    "head_source": "probe_pinned_kept_groups",
    "total_params": 5097875968,
    "dtype": "bfloat16",
    "max_sequence_length": 512,
}

TINY_CONFIG = {
    "model_type": "qwen3",
    "hidden_size": 64,
    "head_dim": 16,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "intermediate_size": 128,
    "vocab_size": 64,
    "rms_norm_eps": 1e-6,
    "num_hidden_layers": 4,
    "attention_bias": False,
    "rope_parameters.rope_theta": 1000000.0,
}
# Original hidden states (9, 18, 27) are the outputs of original layers 8, 17, 26; exported layer 3 feeds no tap.
TINY_METADATA = {
    "text_encoder_out_layers": [1, 2, 3],
    "exported_num_hidden_layers": 4,
    "final_norm_identity": False,
    "kept_original_layers": [8, 17, 26, 27],
    "ffn_keep_by_exported_layer": {"1": 64, "3": 64},
    "head_groups_by_exported_layer": {"2": 1, "3": 1},
}
SPEC_KEYS = {
    "embed_tokens.weight",
    "norm.weight",
    "layers.N.input_layernorm.weight",
    "layers.N.post_attention_layernorm.weight",
    "layers.N.self_attn.q_proj.weight",
    "layers.N.self_attn.k_proj.weight",
    "layers.N.self_attn.v_proj.weight",
    "layers.N.self_attn.o_proj.weight",
    "layers.N.self_attn.q_norm.weight",
    "layers.N.self_attn.k_norm.weight",
    "layers.N.mlp.gate_proj.weight",
    "layers.N.mlp.up_proj.weight",
    "layers.N.mlp.down_proj.weight",
}


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr(Flux2PrunedTextEncoder, "EXPECTED_CONFIG", TINY_CONFIG)
    monkeypatch.setattr(
        Flux2PrunedTextEncoder,
        "EXPECTED_METADATA",
        {k: TINY_METADATA[k] for k in ("text_encoder_out_layers", "exported_num_hidden_layers", "final_norm_identity")},
    )


def tiny_raw(encoder: Qwen3TextEncoder) -> dict[str, mx.array]:
    raw = {k: v for k, v in tree_flatten(encoder.parameters()) if not k.endswith("inv_freq")}
    raw["layers.3.mlp.up_proj.weight"] = mx.zeros((64, 64))
    raw["layers.3.self_attn.q_norm.weight"] = mx.zeros((16,))
    return raw


def tiny_config() -> dict:
    config = {k: v for k, v in TINY_CONFIG.items() if "." not in k}
    config["rope_parameters"] = {"rope_theta": 1000000, "rope_type": "default"}
    return config


@pytest.mark.fast
class TestValidation:
    def test_accepts_the_published_config_and_metadata(self):
        Flux2PrunedTextEncoder.validate_config(REAL_CONFIG)
        Flux2PrunedTextEncoder.validate_metadata(REAL_METADATA)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("model_type", "llama"),
            ("hidden_size", 2560),
            ("head_dim", 64),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 4),
            ("intermediate_size", 9728),
            ("vocab_size", 1000),
            ("rms_norm_eps", 1e-5),
            ("num_hidden_layers", 36),
            ("attention_bias", True),
        ],
    )
    def test_a_config_mismatch_names_the_field_and_the_repo(self, field, value):
        with pytest.raises(ValueError, match=f"config field '{field}'.*{PRUNED_ENCODER_REPO}"):
            Flux2PrunedTextEncoder.validate_config({**REAL_CONFIG, field: value})

    @pytest.mark.parametrize("rope", [{"rope_theta": 10000}, {}, None])
    def test_the_rope_theta_must_be_in_rope_parameters(self, rope):
        config = {**REAL_CONFIG, "rope_parameters": rope, "rope_theta": 1000000}
        with pytest.raises(ValueError, match="rope_parameters.rope_theta"):
            Flux2PrunedTextEncoder.validate_config(config)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("text_encoder_out_layers", [9, 18, 27]),
            ("exported_num_hidden_layers", 36),
            ("final_norm_identity", True),
            ("kept_original_layers", list(range(25))),
            (
                "kept_original_layers",
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
            ),  # fmt: skip
            ("ffn_keep_by_exported_layer", {"17": 0}),
            ("ffn_keep_by_exported_layer", {"17": 13000}),
            ("ffn_keep_by_exported_layer", {"26": 8192}),
            ("head_groups_by_exported_layer", {"9": 0}),
            ("head_groups_by_exported_layer", {"9": 9}),
            ("head_groups_by_exported_layer", {"30": 4}),
            ("head_groups_by_exported_layer", None),
        ],
    )
    def test_a_metadata_mismatch_names_the_field_and_the_repo(self, field, value):
        with pytest.raises(ValueError, match=f"metadata field '{field}.*{PRUNED_ENCODER_REPO}"):
            Flux2PrunedTextEncoder.validate_metadata({**REAL_METADATA, field: value})

    def test_a_missing_metadata_field_is_rejected(self):
        with pytest.raises(ValueError, match="final_norm_identity"):
            Flux2PrunedTextEncoder.validate_metadata({k: v for k, v in REAL_METADATA.items() if k != "final_norm_identity"})  # fmt: skip


@pytest.mark.fast
class TestTapMapAndShapes:
    def test_the_taps_map_to_the_exported_hidden_states(self):
        assert Flux2PrunedTextEncoder.tap_map(REAL_METADATA) == (9, 17, 25)
        assert list(Flux2PrunedTextEncoder.tap_map(REAL_METADATA)) == REAL_METADATA["text_encoder_out_layers"]

    @pytest.mark.parametrize("taps", [(9, 11, 27), (9, 20, 27), (29, 30, 31), (0, 9, 18)])
    def test_a_tap_on_a_deleted_layer_is_refused(self, taps):
        with pytest.raises(ValueError, match="was deleted"):
            Flux2PrunedTextEncoder.map_taps(REAL_METADATA["kept_original_layers"], taps)

    def test_the_encoder_builds_only_the_layers_the_taps_need(self):
        assert Flux2PrunedTextEncoder.num_built_layers(REAL_METADATA) == 25

    def test_layer_shapes_from_the_published_metadata(self):
        shapes = Flux2PrunedTextEncoder.layer_shapes(REAL_METADATA, 25)
        assert len(shapes) == 25
        assert shapes[9] == {"num_key_value_heads": 4, "num_attention_heads": 16}
        assert shapes[17] == {"intermediate_size": 8192, "num_key_value_heads": 5, "num_attention_heads": 20}
        assert shapes[20] == {"intermediate_size": 8192}
        assert shapes[16] == {}
        assert shapes[23] == {"intermediate_size": 8192, "num_key_value_heads": 4, "num_attention_heads": 16}
        assert shapes[24] == {}

    def test_layer_shapes_resolve_to_the_full_dimensions_for_untouched_layers(self, tiny):
        encoder = Flux2PrunedTextEncoder.build_encoder(TINY_METADATA)
        assert len(encoder.layers) == 3
        assert encoder.layers[0].mlp.gate_proj.weight.shape == (128, 64)
        assert encoder.layers[1].mlp.gate_proj.weight.shape == (64, 64)
        assert encoder.layers[2].self_attn.q_proj.weight.shape == (4 * 16, 64)
        assert encoder.layers[2].self_attn.k_proj.weight.shape == (16, 64)
        assert encoder.layers[0].self_attn.q_proj.weight.shape == (8 * 16, 64)

    def test_the_parameter_names_match_the_checkpoint_layout(self, tiny):
        encoder = Flux2PrunedTextEncoder.build_encoder(TINY_METADATA)
        names = {re.sub(r"\d+", "N", k) for k, _ in tree_flatten(encoder.parameters()) if not k.endswith("inv_freq")}
        assert names == SPEC_KEYS


@pytest.mark.fast
class TestQwen3LayerShapes:
    def kwargs(self):
        return dict(vocab_size=32, hidden_size=16, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, intermediate_size=32, head_dim=8)  # fmt: skip

    def shapes(self, encoder):
        return {k: v.shape for k, v in tree_flatten(encoder.parameters())}

    def test_none_and_empty_overrides_leave_the_encoder_unchanged(self):
        default = self.shapes(Qwen3TextEncoder(**self.kwargs()))
        assert self.shapes(Qwen3TextEncoder(**self.kwargs(), layer_shapes=None)) == default
        assert self.shapes(Qwen3TextEncoder(**self.kwargs(), layer_shapes=[{}, {}, {}])) == default
        assert default["layers.0.self_attn.q_proj.weight"] == (32, 16)
        assert default["layers.2.mlp.up_proj.weight"] == (32, 16)

    def test_an_override_changes_only_its_layer(self):
        shapes = [{}, {"num_attention_heads": 2, "num_key_value_heads": 1, "intermediate_size": 16}, {}]
        encoder = Qwen3TextEncoder(**self.kwargs(), layer_shapes=shapes)
        got = self.shapes(encoder)
        assert got["layers.1.self_attn.q_proj.weight"] == (16, 16)
        assert got["layers.1.self_attn.k_proj.weight"] == (8, 16)
        assert got["layers.1.mlp.down_proj.weight"] == (16, 16)
        assert got["layers.0.self_attn.q_proj.weight"] == (32, 16)

    def test_a_wrong_length_is_an_error(self):
        with pytest.raises(ValueError, match="layer_shapes"):
            Qwen3TextEncoder(**self.kwargs(), layer_shapes=[{}])


@pytest.mark.fast
class TestWeights:
    def test_layers_past_the_last_tap_are_dropped_and_the_rest_load(self, tiny):
        mx.random.seed(0)
        source = Flux2PrunedTextEncoder.build_encoder(TINY_METADATA)
        raw = tiny_raw(source)
        target = Flux2PrunedTextEncoder.build_encoder(TINY_METADATA)
        Flux2PrunedTextEncoder.load_weights(target, raw)
        for key, value in tree_flatten(target.parameters()):
            if not key.endswith("inv_freq"):
                assert mx.array_equal(value, raw[key].astype(value.dtype)), key
        assert target.layers[0].self_attn.q_proj.weight.dtype == mx.bfloat16

    def test_a_missing_key_is_an_error(self, tiny):
        encoder = Flux2PrunedTextEncoder.build_encoder(TINY_METADATA)
        raw = tiny_raw(encoder)
        del raw["layers.2.mlp.down_proj.weight"]
        with pytest.raises(ValueError, match="1 missing.*layers.2.mlp.down_proj.weight"):
            Flux2PrunedTextEncoder.load_weights(encoder, raw)

    def test_an_unexpected_key_is_an_error(self, tiny):
        encoder = Flux2PrunedTextEncoder.build_encoder(TINY_METADATA)
        raw = tiny_raw(encoder)
        raw["lm_head.weight"] = mx.zeros((4, 4))
        with pytest.raises(ValueError, match="1 unexpected.*lm_head.weight"):
            Flux2PrunedTextEncoder.load_weights(encoder, raw)

    def test_a_wrong_shape_is_an_error(self, tiny):
        encoder = Flux2PrunedTextEncoder.build_encoder(TINY_METADATA)
        raw = tiny_raw(encoder)
        raw["layers.1.mlp.up_proj.weight"] = mx.zeros((128, 64))  # the full-size shape for a pruned layer
        with pytest.raises(ValueError, match="wrong shape.*layers.1.mlp.up_proj.weight"):
            Flux2PrunedTextEncoder.load_weights(encoder, raw)


@pytest.mark.fast
class TestPromptEmbeds:
    def encoder(self):
        mx.random.seed(1)
        return Flux2PrunedTextEncoder(Flux2PrunedTextEncoder.build_encoder(TINY_METADATA), TINY_METADATA)

    def test_the_default_taps_are_remapped_to_the_inner_encoder(self, tiny):
        model = self.encoder()
        ids = mx.arange(10, dtype=mx.int32)[None] % 64
        mask = mx.array([[1] * 7 + [0] * 3])
        embeds = model.get_prompt_embeds(ids, mask, hidden_state_layers=(9, 18, 27))
        expected = model.encoder.get_prompt_embeds(ids, mask, hidden_state_layers=(1, 2, 3))
        assert embeds.shape == (1, 10, 3 * 64)
        assert mx.array_equal(embeds, expected)
        assert mx.array_equal(model.get_prompt_embeds(ids, mask), expected)

    def test_a_deleted_tap_is_refused(self, tiny):
        with pytest.raises(ValueError, match="was deleted"):
            self.encoder().get_prompt_embeds(mx.zeros((1, 4), dtype=mx.int32), hidden_state_layers=(9, 11, 27))

    def test_the_prompt_encoder_uses_the_native_path_with_the_callers_length(self, tiny):
        calls = []

        class Tokenizer:
            def tokenize(self, prompt, max_length=None, **kwargs):
                calls.append((max_length, kwargs))
                return SimpleNamespace(
                    input_ids=mx.arange(max_length, dtype=mx.int32)[None] % 64,
                    attention_mask=mx.ones((1, max_length), dtype=mx.int32),
                )

        embeds, text_ids = Flux2PromptEncoder.encode_prompt(
            prompt="a fox",
            tokenizer=Tokenizer(),
            text_encoder=self.encoder(),
            max_sequence_length=512,
            num_images_per_prompt=2,
        )
        assert calls == [(512, {})]
        assert embeds.shape == (2, 512, 3 * 64) and text_ids.shape == (2, 512, 4)


@pytest.mark.fast
class TestLoad:
    def patch(self, monkeypatch, tmp_path):
        encoder = Flux2PrunedTextEncoder.build_encoder(TINY_METADATA)
        mx.save_safetensors(str(tmp_path / "model.safetensors"), {k: v.astype(mx.bfloat16) for k, v in tiny_raw(encoder).items()})  # fmt: skip
        import json

        (tmp_path / "config.json").write_text(json.dumps(tiny_config()))
        (tmp_path / "pruning_metadata.json").write_text(json.dumps(TINY_METADATA))
        calls = []

        def fake_download(repo, filename, **kwargs):
            calls.append((repo, filename, kwargs))
            return str(tmp_path / filename)

        monkeypatch.setattr(pruned_module, "hf_hub_download", fake_download)
        return calls

    def test_load_asks_for_exactly_the_three_files_at_the_pinned_revision(self, tiny, monkeypatch, tmp_path):
        calls = self.patch(monkeypatch, tmp_path)
        model = Flux2PrunedTextEncoder.load()
        assert calls == [
            (PRUNED_ENCODER_REPO, name, {"revision": REVISION})
            for name in ("config.json", "pruning_metadata.json", "model.safetensors")
        ]
        out = model.get_prompt_embeds(mx.arange(6, dtype=mx.int32)[None])
        assert out.shape == (1, 6, 3 * 64) and out.dtype == mx.bfloat16

    def test_the_repo_is_pinned(self):
        assert ModelRevisions.get(PRUNED_ENCODER_REPO) == REVISION
        assert PRUNED_ENCODER_REPO == "SearchingMan/FLUX.2-klein-9B-Text-Encoder-Pruned-5.1B"

    def test_quantizing_runs(self, tiny, monkeypatch, tmp_path):
        self.patch(monkeypatch, tmp_path)
        model = Flux2PrunedTextEncoder.load(bits=4)
        assert isinstance(model.encoder.layers[0].self_attn.q_proj, nn.QuantizedLinear)
        assert isinstance(model.encoder.embed_tokens, nn.QuantizedEmbedding)
        out = model.get_prompt_embeds(mx.arange(6, dtype=mx.int32)[None])
        assert out.shape == (1, 6, 3 * 64)

    def test_a_mismatching_config_stops_before_the_weights_load(self, tiny, monkeypatch, tmp_path):
        import json

        self.patch(monkeypatch, tmp_path)
        (tmp_path / "config.json").write_text(json.dumps({**tiny_config(), "hidden_size": 128}))
        monkeypatch.setattr(mx, "load", lambda *a, **k: pytest.fail("weights must not load"))
        with pytest.raises(ValueError, match="hidden_size"):
            Flux2PrunedTextEncoder.load()

    def test_an_offline_download_error_is_friendly(self, monkeypatch):
        def fail(*args, **kwargs):
            raise OSError("offline")

        monkeypatch.setattr(pruned_module, "hf_hub_download", fail)
        with pytest.raises(RuntimeError, match=rf"{PRUNED_ENCODER_REPO}.*10.2 GB.*HF_HUB_OFFLINE"):
            Flux2PrunedTextEncoder.load()

    def test_a_missing_repo_is_friendly(self, monkeypatch):
        def fail(*args, **kwargs):
            raise RepositoryNotFoundError(
                "gone", response=httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co"))
            )

        monkeypatch.setattr(pruned_module, "hf_hub_download", fail)
        with pytest.raises(RuntimeError, match=rf"{PRUNED_ENCODER_REPO}.*not found"):
            Flux2PrunedTextEncoder.load()


@pytest.mark.fast
class TestInitializer:
    NARROW = ("flux2-klein-4b", "flux2-klein-base-4b")
    WIDE = ("flux2-klein-9b", "flux2-klein-9b-kv", "flux2-klein-base-9b")

    @pytest.mark.parametrize("key", NARROW + WIDE)
    def test_klein_models_support_the_flag(self, key):
        assert Flux2Initializer.supports_small_text_encoder(AVAILABLE_MODELS[key])
        Flux2Initializer.validate_small_text_encoder(AVAILABLE_MODELS[key])

    @pytest.mark.parametrize("key", ["iris-3b", "z-image-turbo", "flux3-image-klein"])
    def test_other_models_are_refused_naming_both_groups(self, key):
        assert not Flux2Initializer.supports_small_text_encoder(AVAILABLE_MODELS[key])
        with pytest.raises(ValueError, match="klein-4B.*klein-9B"):
            Flux2Initializer.validate_small_text_encoder(AVAILABLE_MODELS[key])

    def test_the_dispatch_picks_the_encoder_by_conditioning_width(self, monkeypatch):
        monkeypatch.setattr(Flux2SmallTextEncoder, "load", staticmethod(lambda bits=None: ("small", bits)))
        monkeypatch.setattr(Flux2PrunedTextEncoder, "load", staticmethod(lambda bits=None: ("pruned", bits)))
        for key in self.NARROW:
            assert Flux2Initializer.load_small_text_encoder(AVAILABLE_MODELS[key], 8) == ("small", 8)
        for key in self.WIDE:
            assert Flux2Initializer.load_small_text_encoder(AVAILABLE_MODELS[key], None) == ("pruned", None)

    def test_the_9b_init_loads_the_pruned_encoder_without_the_klein_one(self, monkeypatch):
        loaded = []
        monkeypatch.setattr(Flux2PrunedTextEncoder, "load", staticmethod(lambda bits=None: loaded.append(bits) or "pruned"))  # fmt: skip
        monkeypatch.setattr(Flux2Initializer, "_load_weights_without_text_encoder", staticmethod(lambda path: "weights"))  # fmt: skip
        monkeypatch.setattr(Flux2Initializer, "_load_weights", staticmethod(lambda path: pytest.fail("native weights")))
        monkeypatch.setattr(Flux2Initializer, "_init_tokenizers", staticmethod(lambda model, path: None))
        monkeypatch.setattr(Flux2Initializer, "_init_models", staticmethod(lambda model: None))
        monkeypatch.setattr(Flux2Initializer, "_apply_weights", staticmethod(lambda model, weights, q: setattr(model, "bits", q)))  # fmt: skip
        monkeypatch.setattr(Flux2Initializer, "_apply_lora", staticmethod(lambda *a: (None, None)))
        model = SimpleNamespace()
        Flux2Initializer.init(model, AVAILABLE_MODELS["flux2-klein-9b"], quantize=4, small_text_encoder=True)
        assert model.text_encoder == "pruned" and loaded == [4]

    def test_save_model_refuses_with_a_generic_message(self, tmp_path):
        model = SimpleNamespace(small_text_encoder=True, bits=None)
        with pytest.raises(ValueError, match="small_text_encoder") as error:
            Flux2Klein.save_model(model, str(tmp_path / "saved"))
        assert "Qwen3-0.6B" not in str(error.value) and "Qwen3-4B" not in str(error.value)


@pytest.mark.fast
class TestCommandLine:
    COMMANDS = [
        ("mflux-generate-flux2", flux2_generate, []),
        ("mflux-generate-flux2-edit", flux2_edit_generate, ["--image-paths", "ref.png"]),
    ]

    @pytest.mark.parametrize(("command", "module", "extra"), COMMANDS)
    @pytest.mark.parametrize("model", ["flux2-klein-9b", "flux2-klein-9b-kv", "flux2-klein-base-9b", "flux2-klein-4b"])
    def test_validate_accepts_the_flag_for_klein_models(self, monkeypatch, command, module, extra, model):
        monkeypatch.setattr(sys, "argv", [command, "--model", model, "--prompt", "x", *extra, "--small-text-encoder"])
        reset_cli_globals(monkeypatch)
        args = module.build_parser().parse_args()
        command_class = flux2_generate.Flux2Command if module is flux2_generate else flux2_edit_generate.Flux2EditCommand  # fmt: skip
        assert command_class.validate(args).model_name == AVAILABLE_MODELS[model].model_name

    @pytest.mark.parametrize("module", [flux2_generate, flux2_edit_generate])
    def test_the_error_message_names_both_groups(self, module):
        assert module.SMALL_TEXT_ENCODER_UNSUPPORTED == SMALL_TEXT_ENCODER_UNSUPPORTED
        assert "klein-4B" in SMALL_TEXT_ENCODER_UNSUPPORTED and "klein-9B" in SMALL_TEXT_ENCODER_UNSUPPORTED

    @pytest.mark.parametrize("module", [flux2_generate, flux2_edit_generate])
    def test_the_capabilities_condition_covers_9b(self, module):
        option = module.CONDITIONAL_OPTIONS["--small-text-encoder"]
        assert "klein-4B" in option["condition"] and "klein-9B" in option["condition"]
        assert "4B and 9B" in option["reason"]

    @pytest.mark.parametrize("module", [flux2_generate, flux2_edit_generate])
    def test_the_help_describes_both_encoders(self, module):
        action = next(a for a in module.build_parser()._actions if "--small-text-encoder" in a.option_strings)
        assert "AiArtLab/qwen3-0.6b-4b-adapter" in action.help
        assert "SearchingMan/FLUX.2-klein-9B-Text-Encoder-Pruned-5.1B" in action.help
        assert "~10.2 GB instead of ~16.4 GB" in action.help
        assert '"' not in action.help
