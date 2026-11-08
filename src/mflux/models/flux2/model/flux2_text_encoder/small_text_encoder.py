import mlx.core as mx
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.common.resolution.model_revisions import ModelRevisions
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_text_encoder.qwen3_small_text_adapter import Qwen3SmallTextAdapter
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.weights.flux2_weight_definition import Qwen3SmallTextEncoderWeightDefinition

SMALL_ENCODER_REPO = Qwen3SmallTextEncoderWeightDefinition.REPO
ADAPTER_REPO = "AiArtLab/qwen3-0.6b-4b-adapter"
ADAPTER_FILE = "adapter_v14_bal.safetensors"
SUPPORTED_JOINT_ATTENTION_DIM = 7680


class Flux2SmallTextEncoder(nn.Module):
    EXPECTED_METADATA = {
        "student_layers": "2,9,14,18,23,27",
        "teacher_layers": "9,18,27",
        "drop_first": "5",
        "len": "256",
        "norm": "ln-per-layer",
        "attention": "2",
        "attn_heads": "8",
        "attn_dim": "1024",
        "proj_layers": "3",
        "hidden": "8192",
    }
    ENCODER_HIDDEN = 1024
    OUT_DIM = SUPPORTED_JOINT_ATTENTION_DIM

    def __init__(self, encoder: Qwen3TextEncoder, adapter: Qwen3SmallTextAdapter, metadata: dict[str, str]):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.hidden_state_layers = tuple(int(x) for x in metadata["student_layers"].split(","))
        self.drop_first = int(metadata["drop_first"])
        self.max_sequence_length = int(metadata["len"])

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        return self.adapter(self.hidden_states(input_ids, attention_mask)).astype(mx.bfloat16)

    def hidden_states(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        return self.encoder.get_prompt_embeds(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hidden_state_layers=self.hidden_state_layers,
        )

    @staticmethod
    def load(bits: int | None = None) -> "Flux2SmallTextEncoder":
        metadata, adapter_weights = Flux2SmallTextEncoder._download_adapter()
        Flux2SmallTextEncoder.validate_metadata(metadata)
        adapter = Flux2SmallTextEncoder.build_adapter(metadata)
        Flux2SmallTextEncoder.load_adapter_weights(adapter, adapter_weights)
        encoder = Flux2SmallTextEncoder.build_encoder()
        Flux2SmallTextEncoder._load_encoder_weights(encoder)
        if bits is not None:
            nn.quantize(
                encoder,
                group_size=64,
                class_predicate=Qwen3SmallTextEncoderWeightDefinition.quantization_predicate,
                bits=bits,
            )
        return Flux2SmallTextEncoder(encoder=encoder, adapter=adapter, metadata=metadata)

    @staticmethod
    def validate_metadata(metadata: dict[str, str]) -> None:
        for field, expected in Flux2SmallTextEncoder.EXPECTED_METADATA.items():
            actual = metadata.get(field)
            if actual != expected:
                raise ValueError(
                    f"Unsupported small text encoder adapter: metadata field '{field}' is {actual!r}, "
                    f"expected {expected!r} ({ADAPTER_REPO}/{ADAPTER_FILE})."
                )

    @staticmethod
    def build_adapter(metadata: dict[str, str]) -> Qwen3SmallTextAdapter:
        n_slices = len(metadata["student_layers"].split(","))
        return Qwen3SmallTextAdapter(
            in_dim=Flux2SmallTextEncoder.ENCODER_HIDDEN * n_slices,
            out_dim=Flux2SmallTextEncoder.OUT_DIM,
            hidden=int(metadata["hidden"]),
            n_slices=n_slices,
            attn_dim=int(metadata["attn_dim"]),
            attn_heads=int(metadata["attn_heads"]),
            attn_blocks=int(metadata["attention"]),
            max_len=int(metadata["len"]),
        )

    @staticmethod
    def build_encoder() -> Qwen3TextEncoder:
        return Qwen3TextEncoder(
            vocab_size=151936,
            hidden_size=Flux2SmallTextEncoder.ENCODER_HIDDEN,
            num_hidden_layers=Qwen3SmallTextEncoderWeightDefinition.NUM_LAYERS,
            num_attention_heads=16,
            num_key_value_heads=8,
            intermediate_size=3072,
            head_dim=128,
            rope_theta=1000000.0,
            rms_norm_eps=1e-6,
        )

    @staticmethod
    def map_adapter_keys(raw: dict[str, mx.array]) -> dict[str, mx.array]:
        mapped = {}
        for key, value in raw.items():
            if key.endswith(".attn.in_proj_weight") or key.endswith(".attn.in_proj_bias"):
                prefix, name = key.rsplit(".attn.", 1)
                suffix = "weight" if name == "in_proj_weight" else "bias"
                for proj, chunk in zip(("query_proj", "key_proj", "value_proj"), mx.split(value, 3, axis=0)):
                    mapped[f"{prefix}.attn.{proj}.{suffix}"] = chunk
            else:
                mapped[key] = value
        return mapped

    @staticmethod
    def load_adapter_weights(adapter: Qwen3SmallTextAdapter, raw: dict[str, mx.array]) -> None:
        mapped = Flux2SmallTextEncoder.map_adapter_keys(raw)
        expected = dict(tree_flatten(adapter.parameters()))
        missing = sorted(expected.keys() - mapped.keys())
        unexpected = sorted(mapped.keys() - expected.keys())
        if missing or unexpected:
            raise ValueError(
                f"Small text encoder adapter weights do not fit: {len(missing)} missing {missing[:5]}, "
                f"{len(unexpected)} unexpected {unexpected[:5]}."
            )
        wrong_shape = sorted(k for k, v in mapped.items() if v.shape != expected[k].shape)
        if wrong_shape:
            raise ValueError(f"Small text encoder adapter weights have the wrong shape: {wrong_shape[:5]}.")
        adapter.load_weights([(k, v.astype(mx.float32)) for k, v in mapped.items()], strict=True)

    @staticmethod
    def _load_encoder_weights(encoder: Qwen3TextEncoder) -> None:
        weights = Flux2SmallTextEncoder._friendly(
            SMALL_ENCODER_REPO,
            lambda: WeightLoader.load(
                weight_definition=Qwen3SmallTextEncoderWeightDefinition,
                model_path=SMALL_ENCODER_REPO,
            ),
        )
        component = weights.components["text_encoder"]
        # rotary inv_freq is computed at construction, not stored in the checkpoint
        expected = {k for k, _ in tree_flatten(encoder.parameters()) if not k.endswith("inv_freq")}
        supplied = {k for k, _ in tree_flatten(component)}
        if expected != supplied:
            raise ValueError(
                f"{SMALL_ENCODER_REPO} weights do not fit the encoder: missing {sorted(expected - supplied)[:5]}, "
                f"unexpected {sorted(supplied - expected)[:5]}."
            )
        encoder.update(component, strict=False)

    @staticmethod
    def _download_adapter() -> tuple[dict[str, str], dict[str, mx.array]]:
        path = Flux2SmallTextEncoder._friendly(
            ADAPTER_REPO,
            lambda: hf_hub_download(ADAPTER_REPO, ADAPTER_FILE, revision=ModelRevisions.get(ADAPTER_REPO)),
        )
        weights, metadata = mx.load(path, return_metadata=True)
        return metadata, weights

    @staticmethod
    def _friendly(repo_id: str, fn):
        try:
            return fn()
        except (GatedRepoError, RepositoryNotFoundError) as e:
            raise RuntimeError(
                f"--small-text-encoder: Hugging Face repo '{repo_id}' was not found or is not accessible. "
                f"It needs '{SMALL_ENCODER_REPO}' and '{ADAPTER_REPO}' (~2.4 GB combined)."
            ) from e
        except (HfHubHTTPError, OSError) as e:
            raise RuntimeError(
                f"--small-text-encoder: downloading '{repo_id}' failed ({e}). "
                f"It needs '{SMALL_ENCODER_REPO}' and '{ADAPTER_REPO}' (~2.4 GB combined); "
                "check your network connection or HF_HUB_OFFLINE setting and retry."
            ) from e
