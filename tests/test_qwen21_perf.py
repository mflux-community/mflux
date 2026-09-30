from types import SimpleNamespace

import mlx.core as mx
import pytest
from tqdm import tqdm as _tqdm

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.qwen21.model.qwen21_transformer.qwen21_attention import Qwen21Attention
from mflux.models.qwen21.model.qwen21_transformer.qwen21_fused_kernels import (
    fused_qk_norm_rope,
    fused_qk_norm_rope_available,
)
from mflux.models.qwen21.model.qwen21_transformer.qwen21_time_text_embed import Qwen21TimeTextEmbed
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21


@pytest.mark.fast
class TestFusedQkNormRopeKernel:
    def test_fused_prologue_matches_composed_path(self):
        # head_dim must be a multiple of 64 for the fused kernel (full simdgroups)
        attention = Qwen21Attention(dim=128, num_heads=2, head_dim=64, eps=1e-6)
        assert attention.use_fused_prologue

        length = 17
        hidden_states = mx.random.normal((1, length, 128)).astype(mx.bfloat16)
        rope_cos = mx.random.normal((length, 32)).astype(mx.float32)
        rope_sin = mx.random.normal((length, 32)).astype(mx.float32)

        fused_query, fused_key, value = attention._project(hidden_states, rope_cos, rope_sin)
        attention.use_fused_prologue = False
        query, key, value_ref = attention._project(hidden_states, rope_cos, rope_sin)

        mx.eval(fused_query, fused_key, query, key, value, value_ref)
        # the fused kernel skips the bf16 round-trip between norm and rope, so
        # allow one bf16 rounding of disagreement against the composed path
        diff = mx.maximum(
            mx.abs(fused_query.astype(mx.float32) - query.astype(mx.float32)).max(),
            mx.abs(fused_key.astype(mx.float32) - key.astype(mx.float32)).max(),
        )
        assert diff.item() < 0.05
        assert mx.allclose(value, value_ref)

    def test_rope_width_mismatch_falls_back_to_composed_path(self):
        # an axes_dims_rope whose widths do not sum to head_dim must not reach the
        # kernel (it would silently read wrong angles); the composed path raises
        # the same loud broadcast error as stock instead
        attention = Qwen21Attention(dim=128, num_heads=2, head_dim=64, eps=1e-6)
        hidden_states = mx.random.normal((1, 17, 128)).astype(mx.bfloat16)
        mismatched_cos = mx.random.normal((17, 64)).astype(mx.float32)  # 64 != head_dim / 2
        mismatched_sin = mx.random.normal((17, 64)).astype(mx.float32)
        out = fused_qk_norm_rope(
            attention.to_q(hidden_states),
            attention.to_k(hidden_states),
            attention.norm_q.weight,
            attention.norm_k.weight,
            mismatched_cos,
            mismatched_sin,
            num_heads=2,
            head_dim=64,
        )
        assert out is None

    def test_non_default_eps_uses_composed_path(self):
        # the kernel hard-codes eps = 1e-6
        assert not Qwen21Attention(dim=128, num_heads=2, head_dim=64, eps=1e-5).use_fused_prologue

    def test_head_dim_above_reduction_buffer_is_unavailable(self):
        # the kernel reduces over at most 8 simdgroups (D/2 <= 256 threads)
        assert not fused_qk_norm_rope_available(1024)


@pytest.mark.fast
class TestTextPrefixCache:
    @staticmethod
    def _tiny_transformer() -> Qwen21Transformer:
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
    def _config() -> Config:
        return Config(
            width=64,
            height=64,
            guidance=1.0,
            scheduler="linear",
            model_config=ModelConfig.qwen_image_21(),
            num_inference_steps=40,
        )

    @staticmethod
    def _run(transformer: Qwen21Transformer, embedding: mx.array, latents: mx.array) -> mx.array:
        mask = mx.ones((1, embedding.shape[1]))
        out = transformer(
            t=5,
            config=TestTextPrefixCache._config(),
            hidden_states=latents,
            encoder_hidden_states=embedding,
            encoder_hidden_states_mask=mask,
        )
        mx.eval(out)
        return out

    @staticmethod
    def _latents() -> mx.array:
        return mx.random.normal((1, 16, 64)).astype(mx.bfloat16)

    def test_cache_is_bounded_to_two_most_recent_embeddings(self):
        transformer = self._tiny_transformer()
        latents = self._latents()
        embeddings = [mx.random.normal((1, 8, 64)).astype(mx.bfloat16) for _ in range(4)]
        for embedding in embeddings:
            self._run(transformer, embedding, latents)
        assert len(transformer._text_caches) == 2
        assert id(embeddings[-1]) in transformer._text_caches
        assert id(embeddings[-2]) in transformer._text_caches

    def test_clear_text_cache_releases_entries(self):
        transformer = self._tiny_transformer()
        self._run(transformer, mx.random.normal((1, 8, 64)).astype(mx.bfloat16), self._latents())
        assert transformer._text_caches
        transformer.clear_text_cache()
        assert not transformer._text_caches

    def test_cached_and_uncached_steps_agree_within_bf16_rounding(self):
        transformer = self._tiny_transformer()
        embedding = mx.random.normal((1, 8, 64)).astype(mx.bfloat16)
        latents = self._latents()
        cached = self._run(transformer, embedding, latents)
        transformer.use_text_cache = False
        uncached = self._run(transformer, embedding, latents)
        mx.eval(cached, uncached)
        # the two eager paths are bitwise identical; allow bf16 rounding for
        # compile-level differences in the two compiled graphs
        assert mx.abs(cached.astype(mx.float32) - uncached.astype(mx.float32)).max().item() < 0.05

    def test_repeat_embedding_reuses_cache_and_is_stable(self):
        transformer = self._tiny_transformer()
        embedding = mx.random.normal((1, 8, 64)).astype(mx.bfloat16)
        latents = self._latents()
        first = self._run(transformer, embedding, latents)
        second = self._run(transformer, embedding, latents)
        assert len(transformer._text_caches) == 1
        assert mx.array_equal(first, second)


@pytest.mark.fast
class TestTeacacheStepSelection:
    @staticmethod
    def _config(num_steps: int, first: int = 0) -> Config:
        config = Config(
            width=512,
            height=512,
            guidance=1.0,
            scheduler="linear",
            model_config=ModelConfig.qwen_image_21(),
            num_inference_steps=first + num_steps,
        )
        if first:
            # img2img-style run: advance the loop start like init_time_step does
            config._time_steps = _tqdm(range(first, first + num_steps))
        return config

    @staticmethod
    def _transformer() -> SimpleNamespace:
        return SimpleNamespace(time_text_embed=Qwen21TimeTextEmbed(embedding_dim=64))

    def test_skips_only_within_protected_window(self):
        skip = QwenImage21._teacache_skip_steps(self._transformer(), self._config(40), 0.25)
        assert len(skip) == round(0.25 * 40) == 10
        assert all(4 <= t < 36 for t in skip)  # first/last 10% never skipped

    def test_img2img_offset_window(self):
        skip = QwenImage21._teacache_skip_steps(self._transformer(), self._config(16, first=24), 0.25)
        assert skip
        assert all(25 <= t < 39 for t in skip)

    def test_selection_is_deterministic(self):
        transformer = self._transformer()  # one instance: signal weights fixed across calls
        first = QwenImage21._teacache_skip_steps(transformer, self._config(40), 0.3)
        second = QwenImage21._teacache_skip_steps(transformer, self._config(40), 0.3)
        assert first == second

    def test_too_few_steps_skips_nothing(self):
        assert not QwenImage21._teacache_skip_steps(self._transformer(), self._config(9), 0.3)

    def test_ratio_out_of_range_raises(self):
        with pytest.raises(ValueError):
            QwenImage21._teacache_skip_steps(self._transformer(), self._config(40), 1.0)
        with pytest.raises(ValueError):
            QwenImage21._teacache_skip_steps(self._transformer(), self._config(40), 0.0)
