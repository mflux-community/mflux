from functools import lru_cache
from typing import Literal

import mlx.core as mx

from mflux.models.common.resolution.config_resolution import ConfigResolution


class ModelConfig:
    precision: mx.Dtype = mx.bfloat16

    def __init__(
        self,
        priority: int,
        aliases: list[str],
        model_name: str,
        base_model: str | None,
        controlnet_model: str | None,
        custom_transformer_model: str | None,
        num_train_steps: int | None,
        max_sequence_length: int | None,
        supports_guidance: bool | None,
        requires_sigma_shift: bool | None,
        transformer_overrides: dict | None = None,
        text_encoder_overrides: dict | None = None,
        sigma_base_shift: float = 0.5,
        sigma_max_shift: float = 1.15,
        sigma_base_seq_len: int = 256,
        sigma_max_seq_len: int = 4096,
        sigma_shift_terminal: float | None = None,
        lora_training_steps: int | None = None,
        lora_training_guidance: float | None = None,
        supports_kv_cache: bool = False,
    ):
        self.aliases = aliases
        self.model_name = model_name
        self.base_model = base_model
        self.controlnet_model = controlnet_model
        self.custom_transformer_model = custom_transformer_model
        self.num_train_steps = num_train_steps
        self.max_sequence_length = max_sequence_length
        self.supports_guidance = supports_guidance
        self.requires_sigma_shift = requires_sigma_shift
        self.priority = priority
        self.transformer_overrides = transformer_overrides or {}
        self.text_encoder_overrides = text_encoder_overrides or {}
        self.sigma_base_shift = sigma_base_shift
        self.sigma_max_shift = sigma_max_shift
        self.sigma_base_seq_len = sigma_base_seq_len
        self.sigma_max_seq_len = sigma_max_seq_len
        self.sigma_shift_terminal = sigma_shift_terminal
        self.lora_training_steps = lora_training_steps
        self.lora_training_guidance = lora_training_guidance
        self.supports_kv_cache = supports_kv_cache

    @staticmethod
    @lru_cache
    def dev() -> "ModelConfig":
        return AVAILABLE_MODELS["dev"]

    @staticmethod
    @lru_cache
    def schnell() -> "ModelConfig":
        return AVAILABLE_MODELS["schnell"]

    @staticmethod
    @lru_cache
    def dev_kontext() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-kontext"]

    @staticmethod
    @lru_cache
    def dev_fill() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-fill"]

    @staticmethod
    @lru_cache
    def dev_redux() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-redux"]

    @staticmethod
    @lru_cache
    def dev_depth() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-depth"]

    @staticmethod
    @lru_cache
    def dev_controlnet_canny() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-controlnet-canny"]

    @staticmethod
    @lru_cache
    def schnell_controlnet_canny() -> "ModelConfig":
        return AVAILABLE_MODELS["schnell-controlnet-canny"]

    @staticmethod
    @lru_cache
    def dev_controlnet_upscaler() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-controlnet-upscaler"]

    @staticmethod
    @lru_cache
    def dev_fill_catvton() -> "ModelConfig":
        return AVAILABLE_MODELS["dev-fill-catvton"]

    @staticmethod
    @lru_cache
    def krea_dev() -> "ModelConfig":
        return AVAILABLE_MODELS["krea-dev"]

    @staticmethod
    @lru_cache
    def flux2_klein_4b() -> "ModelConfig":
        return AVAILABLE_MODELS["flux2-klein-4b"]

    @staticmethod
    @lru_cache
    def flux2_klein_9b() -> "ModelConfig":
        return AVAILABLE_MODELS["flux2-klein-9b"]

    @staticmethod
    @lru_cache
    def flux2_klein_9b_kv() -> "ModelConfig":
        return AVAILABLE_MODELS["flux2-klein-9b-kv"]

    @staticmethod
    @lru_cache
    def flux2_klein_base_4b() -> "ModelConfig":
        return AVAILABLE_MODELS["flux2-klein-base-4b"]

    @staticmethod
    @lru_cache
    def flux2_klein_base_9b() -> "ModelConfig":
        return AVAILABLE_MODELS["flux2-klein-base-9b"]

    @staticmethod
    @lru_cache
    def krea2() -> "ModelConfig":
        return AVAILABLE_MODELS["krea-2"]

    @staticmethod
    @lru_cache
    def qwen_image() -> "ModelConfig":
        return AVAILABLE_MODELS["qwen-image"]

    @staticmethod
    @lru_cache
    def qwen_image_edit() -> "ModelConfig":
        return AVAILABLE_MODELS["qwen-image-edit"]

    @staticmethod
    @lru_cache
    def fibo() -> "ModelConfig":
        return AVAILABLE_MODELS["fibo"]

    @staticmethod
    @lru_cache
    def fibo_lite() -> "ModelConfig":
        return AVAILABLE_MODELS["fibo-lite"]

    @staticmethod
    @lru_cache
    def fibo_edit() -> "ModelConfig":
        return AVAILABLE_MODELS["fibo-edit"]

    @staticmethod
    @lru_cache
    def fibo_edit_rmbg() -> "ModelConfig":
        return AVAILABLE_MODELS["fibo-edit-rmbg"]

    @staticmethod
    @lru_cache
    def ernie_image_turbo() -> "ModelConfig":
        return AVAILABLE_MODELS["ernie-image-turbo"]

    @staticmethod
    @lru_cache
    def ernie_image() -> "ModelConfig":
        return AVAILABLE_MODELS["ernie-image"]

    @staticmethod
    @lru_cache
    def z_image_turbo() -> "ModelConfig":
        return AVAILABLE_MODELS["z-image-turbo"]

    @staticmethod
    @lru_cache
    def z_image() -> "ModelConfig":
        return AVAILABLE_MODELS["z-image"]

    @staticmethod
    @lru_cache
    def seedvr2_3b() -> "ModelConfig":
        return AVAILABLE_MODELS["seedvr2-3b"]

    @staticmethod
    @lru_cache
    def seedvr2_7b() -> "ModelConfig":
        return AVAILABLE_MODELS["seedvr2-7b"]

    @staticmethod
    @lru_cache
    def ideogram4_fp8() -> "ModelConfig":
        return AVAILABLE_MODELS["ideogram-4-fp8"]

    def x_embedder_input_dim(self) -> int:
        if "Fill" in self.model_name:
            return 384
        if "Depth" in self.model_name:
            return 128
        else:
            return 64

    def is_canny(self) -> bool:
        return self.controlnet_model is not None and "Canny" in self.controlnet_model

    @staticmethod
    def from_name(
        model_name: str,
        base_model: Literal["dev", "schnell", "krea-dev"] | None = None,
    ) -> "ModelConfig":
        return ConfigResolution.resolve(model_name=model_name, base_model=base_model)


AVAILABLE_MODELS = {
    "krea-2": ModelConfig(
        priority=15,
        aliases=["krea-2", "krea2"],
        model_name="krea/Krea-2-Turbo",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=1024,
        supports_guidance=True,
        requires_sigma_shift=True,
        sigma_max_shift=1.15,
    ),
    "dev": ModelConfig(
        priority=0,
        aliases=["dev"],
        model_name="black-forest-labs/FLUX.1-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "schnell": ModelConfig(
        priority=1,
        aliases=["schnell"],
        model_name="black-forest-labs/FLUX.1-schnell",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=256,
        supports_guidance=False,
        requires_sigma_shift=False,
    ),
    "dev-kontext": ModelConfig(
        priority=2,
        aliases=["dev-kontext"],
        model_name="black-forest-labs/FLUX.1-Kontext-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "dev-fill": ModelConfig(
        priority=3,
        aliases=["dev-fill"],
        model_name="black-forest-labs/FLUX.1-Fill-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "dev-redux": ModelConfig(
        priority=4,
        aliases=["dev-redux"],
        model_name="black-forest-labs/FLUX.1-Redux-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "dev-depth": ModelConfig(
        priority=5,
        aliases=["dev-depth"],
        model_name="black-forest-labs/FLUX.1-Depth-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "dev-controlnet-canny": ModelConfig(
        priority=6,
        aliases=["dev-controlnet-canny"],
        model_name="black-forest-labs/FLUX.1-dev",
        base_model=None,
        controlnet_model="InstantX/FLUX.1-dev-Controlnet-Canny",
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "schnell-controlnet-canny": ModelConfig(
        priority=7,
        aliases=["schnell-controlnet-canny"],
        model_name="black-forest-labs/FLUX.1-schnell",
        base_model=None,
        controlnet_model="InstantX/FLUX.1-dev-Controlnet-Canny",
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=256,
        supports_guidance=False,
        requires_sigma_shift=False,
    ),
    "dev-controlnet-upscaler": ModelConfig(
        priority=8,
        aliases=["dev-controlnet-upscaler"],
        model_name="black-forest-labs/FLUX.1-dev",
        base_model=None,
        controlnet_model="jasperai/Flux.1-dev-Controlnet-Upscaler",
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=False,
        requires_sigma_shift=False,
    ),
    "dev-fill-catvton": ModelConfig(
        priority=9,
        aliases=["dev-fill-catvton"],
        model_name="black-forest-labs/FLUX.1-Fill-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model="xiaozaa/catvton-flux-beta",
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
    ),
    "krea-dev": ModelConfig(
        priority=10,
        aliases=["krea-dev", "dev-krea"],
        model_name="black-forest-labs/FLUX.1-Krea-dev",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "flux2-klein-4b": ModelConfig(
        priority=11,
        aliases=["flux2-klein-4b", "flux2-klein-4B", "flux2-klein", "klein-4b", "klein-4B"],
        model_name="black-forest-labs/FLUX.2-klein-4B",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        transformer_overrides={
            "num_layers": 5,
            "num_single_layers": 20,
            "num_attention_heads": 24,
            "joint_attention_dim": 7680,
        },
        text_encoder_overrides={
            "hidden_size": 2560,
            "intermediate_size": 9728,
        },
    ),
    "flux2-klein-9b": ModelConfig(
        priority=12,
        aliases=["flux2-klein-9b", "flux2-klein-9B", "klein-9b", "klein-9B"],
        model_name="black-forest-labs/FLUX.2-klein-9B",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        transformer_overrides={
            "num_layers": 8,
            "num_single_layers": 24,
            "num_attention_heads": 32,
            "joint_attention_dim": 12288,
        },
        text_encoder_overrides={
            "hidden_size": 4096,
            "intermediate_size": 12288,
        },
    ),
    "flux2-klein-9b-kv": ModelConfig(
        priority=12,
        aliases=[
            "flux2-klein-9b-kv",
            "flux2-klein-9B-kv",
            "flux2-klein-9b-KV",
            "klein-9b-kv",
            "klein-9B-kv",
        ],
        model_name="black-forest-labs/FLUX.2-klein-9b-kv",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        transformer_overrides={
            "num_layers": 8,
            "num_single_layers": 24,
            "num_attention_heads": 32,
            "joint_attention_dim": 12288,
        },
        text_encoder_overrides={
            "hidden_size": 4096,
            "intermediate_size": 12288,
        },
        supports_kv_cache=True,
    ),
    "flux2-klein-base-4b": ModelConfig(
        priority=13,
        aliases=[
            "flux2-klein-base-4b",
            "flux2-klein-base-4B",
            "flux2-base-4b",
            "flux2-base-4B",
            "klein-base-4b",
            "klein-base-4B",
        ],
        model_name="black-forest-labs/FLUX.2-klein-base-4B",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        transformer_overrides={
            "num_layers": 5,
            "num_single_layers": 20,
            "num_attention_heads": 24,
            "joint_attention_dim": 7680,
        },
        text_encoder_overrides={
            "hidden_size": 2560,
            "intermediate_size": 9728,
        },
    ),
    "flux2-klein-base-9b": ModelConfig(
        priority=14,
        aliases=[
            "flux2-klein-base-9b",
            "flux2-klein-base-9B",
            "flux2-base-9b",
            "flux2-base-9B",
            "klein-base-9b",
            "klein-base-9B",
        ],
        model_name="black-forest-labs/FLUX.2-klein-base-9B",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
        transformer_overrides={
            "num_layers": 8,
            "num_single_layers": 24,
            "num_attention_heads": 32,
            "joint_attention_dim": 12288,
        },
        text_encoder_overrides={
            "hidden_size": 4096,
            "intermediate_size": 12288,
        },
    ),
    "qwen-image": ModelConfig(
        priority=15,
        aliases=["qwen-image", "qwen"],
        model_name="Qwen/Qwen-Image",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=None,
        requires_sigma_shift=True,
        sigma_max_shift=0.9,
        sigma_max_seq_len=8192,
        sigma_shift_terminal=0.02,
    ),
    "qwen-image-edit": ModelConfig(
        priority=16,
        aliases=["qwen-image-edit", "qwen-edit", "qwen-edit-plus", "qwen-edit-2509"],
        model_name="Qwen/Qwen-Image-Edit-2509",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=None,
        requires_sigma_shift=True,
        sigma_max_shift=0.9,
        sigma_max_seq_len=8192,
        sigma_shift_terminal=0.02,
    ),
    "fibo": ModelConfig(
        priority=17,
        aliases=["fibo"],
        model_name="briaai/FIBO",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
    ),
    "fibo-lite": ModelConfig(
        priority=18,
        aliases=["fibo-lite", "fibo_lite"],
        model_name="briaai/Fibo-lite",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
    ),
    "fibo-edit": ModelConfig(
        priority=19,
        aliases=["fibo-edit", "fiboedit"],
        model_name="briaai/Fibo-Edit",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
    ),
    "fibo-edit-rmbg": ModelConfig(
        priority=24,
        aliases=["fibo-edit-rmbg", "fiboedit-rmbg"],
        model_name="briaai/Fibo-Edit-RMBG",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
    ),
    "z-image": ModelConfig(
        priority=20,
        aliases=["z-image", "zimage"],
        model_name="Tongyi-MAI/Z-Image",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=True,
    ),
    "z-image-turbo": ModelConfig(
        priority=21,
        aliases=["z-image-turbo", "zimage-turbo"],
        model_name="Tongyi-MAI/Z-Image-Turbo",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=False,  # Turbo model uses guidance_scale=0
        requires_sigma_shift=True,
    ),
    "seedvr2-3b": ModelConfig(
        priority=22,
        aliases=["seedvr2-3b", "seedvr2"],
        model_name="numz/SeedVR2_comfyUI",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=True,
        requires_sigma_shift=None,
    ),
    "ernie-image": ModelConfig(
        priority=27,
        aliases=["ernie-image"],
        model_name="baidu/ERNIE-Image",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=2048,
        supports_guidance=True,
        requires_sigma_shift=True,
        sigma_base_shift=1.3863,
        sigma_max_shift=1.3863,
        lora_training_steps=50,
        lora_training_guidance=4.0,
        transformer_overrides={"rope_axes_dim": [32, 48, 48]},
    ),
    "ernie-image-turbo": ModelConfig(
        priority=26,
        aliases=["ernie-image-turbo"],
        model_name="baidu/ERNIE-Image-Turbo",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=2048,
        supports_guidance=True,
        requires_sigma_shift=True,
        sigma_base_shift=1.3863,
        sigma_max_shift=1.3863,
        lora_training_steps=8,
        lora_training_guidance=1.0,
        transformer_overrides={"rope_axes_dim": [32, 48, 48]},
    ),
    "seedvr2-7b": ModelConfig(
        priority=23,
        aliases=["seedvr2-7b", "seedvr2-7B"],
        model_name="numz/SeedVR2_comfyUI",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=True,
        requires_sigma_shift=None,
        transformer_overrides={
            "vid_dim": 3072,
            "heads": 24,
            "num_layers": 36,
            "mm_layers": 36,
            "rope_dim": 64,
            "rope_on_text": False,
            "rope_freqs_for": "pixel",
            "mlp_type": "normal",
            "use_output_ada": False,
            "last_layer_vid_only": False,
        },
    ),
    "ideogram-4-fp8": ModelConfig(
        priority=25,
        aliases=[
            "ideogram-4-fp8",
            "ideogram4-fp8",
            "ideogram4",
            "ideogram-4",
            "ideogram",
        ],
        model_name="ideogram-ai/ideogram-4-fp8",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=2048,
        supports_guidance=True,
        requires_sigma_shift=False,
    ),
}
