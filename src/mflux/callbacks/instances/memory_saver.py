import gc

import mlx.core as mx
import PIL.Image
from tqdm import tqdm

from mflux.callbacks.callback import AfterLoopCallback, BeforeLoopCallback, InLoopCallback
from mflux.models.common.config.config import Config
from mflux.models.common.vae.tiling_config import TilingConfig


class MemorySaver(BeforeLoopCallback, InLoopCallback, AfterLoopCallback):
    def __init__(
        self,
        model,
        keep_transformer: bool = True,
        cache_limit_bytes: int | None = 1000**3,
        args=None,
        num_seeds: int = 1,
    ):
        self.model = model
        self.keep_transformer = keep_transformer
        self.peak_memory: int = 0
        self._num_seeds = num_seeds
        if cache_limit_bytes is not None:
            if model.tiling_config is None:
                self.model.tiling_config = TilingConfig()
            mx.set_cache_limit(cache_limit_bytes)
            mx.clear_cache()
            mx.reset_peak_memory()

    def call_before_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        canny_image: PIL.Image.Image | None = None,
        depth_image: PIL.Image.Image | None = None,
        control_images: list[PIL.Image.Image] | None = None,
    ) -> None:
        self.peak_memory = mx.get_peak_memory()
        # Only evict when safe: single-seed run, or embeddings are cached in prompt_cache
        # (Flux1 caches embeddings so encoder isn't needed for subsequent seeds;
        #  Flux2 re-encodes each call and has no prompt_cache — keep encoder for multi-seed)
        has_cached_embeds = hasattr(self.model, "prompt_cache") and prompt in (self.model.prompt_cache or {})
        if self._num_seeds <= 1 or has_cached_embeds:
            self._delete_text_encoders()

    def call_in_loop(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        time_steps: tqdm,
    ) -> None:
        self.peak_memory = mx.get_peak_memory()

    def call_after_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
    ) -> None:
        self.peak_memory = mx.get_peak_memory()
        if not self.keep_transformer:
            self._delete_transformer()
        else:
            # Clear Metal cache between seeds — without a cache limit, MLX accumulates
            # freed tensor buffers indefinitely across sequential denoising passes.
            gc.collect()
            mx.clear_cache()

    def _delete_text_encoders(self) -> None:
        # repeated image generation only works with the same prompt (cache)
        if hasattr(self.model, "clip_text_encoder"):
            self.model.clip_text_encoder = None
        if hasattr(self.model, "t5_text_encoder"):
            self.model.t5_text_encoder = None
        if hasattr(self.model, "text_encoder") and self.model.text_encoder is not None:
            self.model.text_encoder = None
        if hasattr(self.model, "image_encoder") and self.model.image_encoder is not None:
            self.model.image_encoder = None
        if hasattr(self.model, "image_embedder") and self.model.image_embedder is not None:
            self.model.image_embedder = None
        if hasattr(self.model, "depth_pro") and self.model.depth_pro is not None:
            self.model.depth_pro = None
        if hasattr(self.model, "qwen_vl_encoder") and self.model.qwen_vl_encoder is not None:
            self.model.qwen_vl_encoder = None
        # Clear VLM tokenizers from the tokenizers dict if present
        if hasattr(self.model, "tokenizers") and "qwen_vl" in self.model.tokenizers:
            self.model.tokenizers["qwen_vl"] = None
        gc.collect()
        mx.clear_cache()

    def _delete_transformer(self) -> None:
        self.model.transformer = None
        if hasattr(self.model, "transformer_controlnet"):
            self.model.transformer_controlnet = None
        if hasattr(self.model, "controlnet"):
            self.model.controlnet = None
        if hasattr(self.model, "conditional_transformer"):
            self.model.conditional_transformer = None
        if hasattr(self.model, "unconditional_transformer"):
            self.model.unconditional_transformer = None
        gc.collect()
        mx.clear_cache()

    def memory_stats(self) -> str:
        self.peak_memory = mx.get_peak_memory()
        return f"Peak MLX memory: {self.peak_memory / 10**9:.2f} GB"
