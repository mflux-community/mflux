import json

import mlx.core as mx
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.common.config import ModelConfig
from mflux.models.common.resolution.model_revisions import ModelRevisions
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.weights.flux2_weight_definition import Qwen3SmallTextEncoderWeightDefinition

PRUNED_ENCODER_REPO = "SearchingMan/FLUX.2-klein-9B-Text-Encoder-Pruned-5.1B"
CONFIG_FILE = "config.json"
METADATA_FILE = "pruning_metadata.json"
WEIGHTS_FILE = "model.safetensors"
SUPPORTED_JOINT_ATTENTION_DIM = 12288
# The hidden states of the original Qwen3-8B that FLUX.2 reads (HF convention: 0 = embeddings, h = output of layer h-1).
ORIGINAL_TAPS = (9, 18, 27)


class Flux2PrunedTextEncoder(nn.Module):
    REPO = PRUNED_ENCODER_REPO
    EXPECTED_CONFIG = {
        "model_type": "qwen3",
        "hidden_size": 4096,
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 12288,
        "vocab_size": 151936,
        "rms_norm_eps": 1e-6,
        "num_hidden_layers": 26,
        "attention_bias": False,
        "rope_parameters.rope_theta": 1000000.0,
    }
    EXPECTED_METADATA = {
        "text_encoder_out_layers": [9, 17, 25],
        "exported_num_hidden_layers": 26,
        "final_norm_identity": False,
    }

    def __init__(self, encoder: Qwen3TextEncoder, metadata: dict):
        super().__init__()
        self.encoder = encoder
        self.kept_original_layers = list(metadata["kept_original_layers"])

    def get_prompt_embeds(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        hidden_state_layers: tuple[int, ...] = ORIGINAL_TAPS,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        taps = Flux2PrunedTextEncoder.map_taps(self.kept_original_layers, hidden_state_layers)
        # Only the layers up to the last default tap are built, so a later tap has no hidden state to read.
        if max(taps) > len(self.encoder.layers):
            raise ValueError(
                f"Hidden states {hidden_state_layers} are not available from the pruned text encoder: "
                f"only the layers up to original hidden state {max(ORIGINAL_TAPS)} are built."
            )
        return self.encoder.get_prompt_embeds(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hidden_state_layers=taps,
            position_ids=position_ids,
        )

    @staticmethod
    def load(bits: int | None = None) -> "Flux2PrunedTextEncoder":
        revision = ModelRevisions.get(PRUNED_ENCODER_REPO)
        paths = {
            name: Flux2PrunedTextEncoder._friendly(
                lambda name=name: hf_hub_download(PRUNED_ENCODER_REPO, name, revision=revision)
            )
            for name in (CONFIG_FILE, METADATA_FILE, WEIGHTS_FILE)
        }
        with open(paths[CONFIG_FILE]) as f:
            Flux2PrunedTextEncoder.validate_config(json.load(f))
        with open(paths[METADATA_FILE]) as f:
            metadata = json.load(f)
        Flux2PrunedTextEncoder.validate_metadata(metadata)
        encoder = Flux2PrunedTextEncoder.build_encoder(metadata)
        Flux2PrunedTextEncoder.load_encoder_weights(encoder, mx.load(paths[WEIGHTS_FILE]))
        if bits is not None:
            nn.quantize(
                encoder,
                group_size=64,
                class_predicate=Qwen3SmallTextEncoderWeightDefinition.quantization_predicate,
                bits=bits,
            )
        return Flux2PrunedTextEncoder(encoder=encoder, metadata=metadata)

    @staticmethod
    def validate_config(config: dict) -> None:
        for field, expected in Flux2PrunedTextEncoder.EXPECTED_CONFIG.items():
            actual = config
            for part in field.split("."):
                actual = actual.get(part) if isinstance(actual, dict) else None
            if actual != expected:
                raise ValueError(
                    f"Unsupported pruned text encoder checkpoint: config field '{field}' is {actual!r}, "
                    f"expected {expected!r} ({PRUNED_ENCODER_REPO})."
                )

    @staticmethod
    def validate_metadata(metadata: dict) -> None:
        def reject(field: str, actual) -> None:
            raise ValueError(
                f"Unsupported pruned text encoder checkpoint: metadata field '{field}' is {actual!r} "
                f"({PRUNED_ENCODER_REPO})."
            )

        for field, expected in Flux2PrunedTextEncoder.EXPECTED_METADATA.items():
            if metadata.get(field) != expected:
                raise ValueError(
                    f"Unsupported pruned text encoder checkpoint: metadata field '{field}' is "
                    f"{metadata.get(field)!r}, expected {expected!r} ({PRUNED_ENCODER_REPO})."
                )
        num_layers = Flux2PrunedTextEncoder.EXPECTED_METADATA["exported_num_hidden_layers"]
        kept = metadata.get("kept_original_layers")
        if not isinstance(kept, list) or len(kept) != num_layers or kept != sorted(set(kept)) or min(kept) < 0:
            reject("kept_original_layers", kept)
        config = Flux2PrunedTextEncoder.EXPECTED_CONFIG
        limits = {
            "ffn_keep_by_exported_layer": (1, config["intermediate_size"]),
            "head_groups_by_exported_layer": (1, config["num_key_value_heads"]),
        }
        for field, (low, high) in limits.items():
            if not isinstance(metadata.get(field), dict):
                reject(field, metadata.get(field))
            for layer, value in metadata[field].items():
                if not (
                    layer.isdigit() and int(layer) < num_layers and isinstance(value, int) and low <= value <= high
                ):
                    reject(f"{field}[{layer!r}]", value)
        try:
            tapped = Flux2PrunedTextEncoder.map_taps(kept, ORIGINAL_TAPS)
        except ValueError:
            tapped = None
        if tapped is None or list(tapped) != metadata["text_encoder_out_layers"]:
            reject("kept_original_layers", kept)

    @staticmethod
    def map_taps(kept_original_layers: list[int], hidden_state_layers: tuple[int, ...]) -> tuple[int, ...]:
        # Original hidden state h is the output of original layer h-1, which sits at index i of the exported layers.
        mapped = []
        for h in hidden_state_layers:
            if h < 1 or (h - 1) not in kept_original_layers:
                raise ValueError(
                    f"Hidden state {h} is not available from the pruned text encoder: original layer {h - 1} "
                    f"was deleted ({PRUNED_ENCODER_REPO})."
                )
            mapped.append(kept_original_layers.index(h - 1) + 1)
        return tuple(mapped)

    @staticmethod
    def tap_map(metadata: dict) -> tuple[int, ...]:
        return Flux2PrunedTextEncoder.map_taps(metadata["kept_original_layers"], ORIGINAL_TAPS)

    @staticmethod
    def num_built_layers(metadata: dict) -> int:
        # Hidden state t is the output of exported layer t-1: later layers and the final norm feed no tap.
        return max(Flux2PrunedTextEncoder.tap_map(metadata))

    @staticmethod
    def layer_shapes(metadata: dict, num_layers: int) -> list[dict[str, int]]:
        config = Flux2PrunedTextEncoder.EXPECTED_CONFIG
        queries_per_kv = config["num_attention_heads"] // config["num_key_value_heads"]
        shapes = [{} for _ in range(num_layers)]
        for layer, keep in metadata["ffn_keep_by_exported_layer"].items():
            if int(layer) < num_layers:
                shapes[int(layer)]["intermediate_size"] = keep
        for layer, groups in metadata["head_groups_by_exported_layer"].items():
            if int(layer) < num_layers:
                shapes[int(layer)]["num_key_value_heads"] = groups
                shapes[int(layer)]["num_attention_heads"] = groups * queries_per_kv
        return shapes

    @staticmethod
    def build_encoder(metadata: dict) -> Qwen3TextEncoder:
        config = Flux2PrunedTextEncoder.EXPECTED_CONFIG
        num_layers = Flux2PrunedTextEncoder.num_built_layers(metadata)
        return Qwen3TextEncoder(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=num_layers,
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            intermediate_size=config["intermediate_size"],
            head_dim=config["head_dim"],
            rope_theta=config["rope_parameters.rope_theta"],
            rms_norm_eps=config["rms_norm_eps"],
            layer_shapes=Flux2PrunedTextEncoder.layer_shapes(metadata, num_layers),
        )

    @staticmethod
    def load_encoder_weights(encoder: Qwen3TextEncoder, raw: dict[str, mx.array]) -> None:
        # Layers past the last tap feed nothing, so they are not loaded.
        def beyond(key: str) -> bool:
            return key.startswith("layers.") and int(key.split(".")[1]) >= len(encoder.layers)

        mapped = {k: v for k, v in raw.items() if not beyond(k)}
        # rotary inv_freq is computed at construction, not stored in the checkpoint
        expected = {k: v for k, v in tree_flatten(encoder.parameters()) if not k.endswith("inv_freq")}
        missing = sorted(expected.keys() - mapped.keys())
        unexpected = sorted(mapped.keys() - expected.keys())
        if missing or unexpected:
            raise ValueError(
                f"Pruned text encoder weights do not fit: {len(missing)} missing {missing[:5]}, "
                f"{len(unexpected)} unexpected {unexpected[:5]}."
            )
        wrong_shape = sorted(k for k, v in mapped.items() if v.shape != expected[k].shape)
        if wrong_shape:
            raise ValueError(f"Pruned text encoder weights have the wrong shape: {wrong_shape[:5]}.")
        encoder.load_weights([(k, v.astype(ModelConfig.precision)) for k, v in mapped.items()], strict=False)

    @staticmethod
    def _friendly(fn):
        try:
            return fn()
        except (GatedRepoError, RepositoryNotFoundError) as e:
            raise RuntimeError(
                f"--small-text-encoder: Hugging Face repo '{PRUNED_ENCODER_REPO}' was not found or is not "
                "accessible (~10.2 GB needed)."
            ) from e
        except (HfHubHTTPError, OSError) as e:
            raise RuntimeError(
                f"--small-text-encoder: downloading '{PRUNED_ENCODER_REPO}' failed ({e}). "
                "It needs ~10.2 GB; check your network connection or HF_HUB_OFFLINE setting and retry."
            ) from e
