import json

import mlx.core as mx
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.common.config import ModelConfig
from mflux.models.common.resolution.model_revisions import ModelRevisions
from mflux.models.z_image.model.z_image_text_encoder.student_adapter import ZImageStudentAdapter
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder

SMALL_ENCODER_REPO = "SearchingMan/Z-Image-Turbo-student-adapter"
CONFIG_FILE = "text_encoder/config.json"
WEIGHTS_FILE = "text_encoder/model.safetensors"
STUDENT_PREFIX = "student.model."
ADAPTER_PREFIX = "adapter."


class ZImageSmallTextEncoder(nn.Module):
    EXPECTED_CONFIG = {
        "adapter_dim": 1024,
        "adapter_heads": 8,
        "adapter_blocks": 2,
        "adapter_ff_mult": 4,
        "hs_tap_index": -2,
        "teacher_hidden_size": 2560,
        "student_hidden_size": 2048,
    }
    EXPECTED_STUDENT_CONFIG = {
        "hidden_size": 2048,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "intermediate_size": 6144,
        "head_dim": 128,
    }

    def __init__(self, student: TextEncoder, adapter: ZImageStudentAdapter):
        super().__init__()
        self.student = student
        self.adapter = adapter

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        tap = self.student(input_ids, attention_mask).astype(mx.float32)
        return self.adapter(tap, attention_mask).astype(ModelConfig.precision)

    @staticmethod
    def load(bits: int | None = None) -> "ZImageSmallTextEncoder":
        revision = ModelRevisions.get(SMALL_ENCODER_REPO)
        config_path = ZImageSmallTextEncoder._friendly(
            lambda: hf_hub_download(SMALL_ENCODER_REPO, CONFIG_FILE, revision=revision)
        )
        weights_path = ZImageSmallTextEncoder._friendly(
            lambda: hf_hub_download(SMALL_ENCODER_REPO, WEIGHTS_FILE, revision=revision)
        )
        with open(config_path) as f:
            config = json.load(f)
        ZImageSmallTextEncoder.validate_config(config)
        raw = mx.load(weights_path)
        student = ZImageSmallTextEncoder.build_student()
        adapter = ZImageSmallTextEncoder.build_adapter()
        ZImageSmallTextEncoder.load_student_weights(student, raw)
        ZImageSmallTextEncoder.load_adapter_weights(adapter, raw)
        if bits is not None:
            nn.quantize(
                student,
                group_size=64,
                bits=bits,
                class_predicate=lambda path, module: hasattr(module, "to_quantized"),
            )
        return ZImageSmallTextEncoder(student=student, adapter=adapter)

    @staticmethod
    def validate_config(config: dict) -> None:
        student_config = config.get("student_config_dict") or {}
        checks = [(k, config.get(k), v) for k, v in ZImageSmallTextEncoder.EXPECTED_CONFIG.items()]
        checks += [
            (f"student_config_dict.{k}", student_config.get(k), v)
            for k, v in ZImageSmallTextEncoder.EXPECTED_STUDENT_CONFIG.items()
        ]
        for field, actual, expected in checks:
            if actual != expected:
                raise ValueError(
                    f"Unsupported small text encoder checkpoint: config field '{field}' is {actual!r}, "
                    f"expected {expected!r} ({SMALL_ENCODER_REPO})."
                )

    @staticmethod
    def build_student() -> TextEncoder:
        return TextEncoder(
            vocab_size=151936,
            hidden_size=2048,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            intermediate_size=6144,
            head_dim=128,
            # The config also lists rope_parameters.rope_theta 1e6, but transformers 4.57.6 (the version the
            # author trained and ran with) reads only rope_theta, so the student was trained at 10000.
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
        )

    @staticmethod
    def build_adapter() -> ZImageStudentAdapter:
        return ZImageStudentAdapter(student_dim=2048, teacher_dim=2560, dim=1024, heads=8, blocks=2, ff_mult=4)

    @staticmethod
    def map_student_keys(raw: dict[str, mx.array]) -> dict[str, mx.array]:
        return {k.removeprefix(STUDENT_PREFIX): v for k, v in raw.items() if k.startswith(STUDENT_PREFIX)}

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
                mapped[key.replace(".ff.0.", ".ff_in.").replace(".ff.3.", ".ff_out.")] = value
        return mapped

    @staticmethod
    def load_student_weights(student: TextEncoder, raw: dict[str, mx.array]) -> None:
        mapped = ZImageSmallTextEncoder.map_student_keys(raw)
        # rotary inv_freq is computed at construction, not stored in the checkpoint
        expected = {k: v for k, v in tree_flatten(student.parameters()) if not k.endswith("inv_freq")}
        ZImageSmallTextEncoder._check_keys("student", mapped, expected)
        student.load_weights([(k, v.astype(ModelConfig.precision)) for k, v in mapped.items()], strict=False)

    @staticmethod
    def load_adapter_weights(adapter: ZImageStudentAdapter, raw: dict[str, mx.array]) -> None:
        adapter_raw = {k.removeprefix(ADAPTER_PREFIX): v for k, v in raw.items() if k.startswith(ADAPTER_PREFIX)}
        mapped = ZImageSmallTextEncoder.map_adapter_keys(adapter_raw)
        expected = dict(tree_flatten(adapter.parameters()))
        ZImageSmallTextEncoder._check_keys("adapter", mapped, expected)
        adapter.load_weights([(k, v.astype(mx.float32)) for k, v in mapped.items()], strict=True)

    @staticmethod
    def _check_keys(name: str, mapped: dict[str, mx.array], expected: dict[str, mx.array]) -> None:
        missing = sorted(expected.keys() - mapped.keys())
        unexpected = sorted(mapped.keys() - expected.keys())
        if missing or unexpected:
            raise ValueError(
                f"Small text encoder {name} weights do not fit: {len(missing)} missing {missing[:5]}, "
                f"{len(unexpected)} unexpected {unexpected[:5]}."
            )
        wrong_shape = sorted(k for k, v in mapped.items() if v.shape != expected[k].shape)
        if wrong_shape:
            raise ValueError(f"Small text encoder {name} weights have the wrong shape: {wrong_shape[:5]}.")

    @staticmethod
    def _friendly(fn):
        try:
            return fn()
        except (GatedRepoError, RepositoryNotFoundError) as e:
            raise RuntimeError(
                f"--small-text-encoder: Hugging Face repo '{SMALL_ENCODER_REPO}' was not found or is not "
                "accessible (~3.6 GB needed)."
            ) from e
        except (HfHubHTTPError, OSError) as e:
            raise RuntimeError(
                f"--small-text-encoder: downloading '{SMALL_ENCODER_REPO}' failed ({e}). "
                "It needs ~3.6 GB; check your network connection or HF_HUB_OFFLINE setting and retry."
            ) from e
