import mlx.core as mx
import pytest

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.qwen21.model.qwen21_transformer import qwen21_fused_kernels
from mflux.models.qwen21.model.qwen21_transformer.qwen21_attention import Qwen21Attention
from mflux.models.qwen21.model.qwen21_transformer.qwen21_fused_kernels import (
    fused_qk_norm_rope,
    fused_qk_norm_rope_available,
)
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer import Qwen21Transformer
from mflux.models.qwen21.model.qwen21_transformer.qwen21_transformer_block import (
    Qwen21TransformerBlock,
    _gated_residual,
)


@pytest.mark.fast
class TestFusedQkNormRopeKernel:
    @pytest.mark.parametrize("head_dim", [64, 128, 192, 256, 320, 384, 448, 512])
    @pytest.mark.parametrize(
        ("dtype", "weight_dtype"),
        [
            (mx.bfloat16, mx.bfloat16),
            (mx.float16, mx.float16),
            (mx.float32, mx.float32),
            (mx.bfloat16, mx.float32),
            (mx.float16, mx.bfloat16),
            (mx.float32, mx.float16),
        ],
    )
    def test_native_rounding_and_promoted_dtype(self, head_dim, dtype, weight_dtype):
        self._require_kernel(head_dim)
        batch, length, heads = 2, 5, 3  # crosses batch/head and partial threadgroup boundaries
        shape = (batch, length, heads * head_dim)
        indices = mx.arange(batch * length * heads * head_dim).reshape(shape)
        # Dyadic inputs keep the sum of squares exact in FP32 across reduction orders.
        q = ((indices % 17 - 8).astype(mx.float32) / 8).astype(dtype)
        k = ((indices % 13 - 6).astype(mx.float32) / 4).astype(dtype)
        wq = (0.75 + (mx.arange(head_dim) % 9).astype(mx.float32) / 16).astype(weight_dtype)
        wk = (1.25 - (mx.arange(head_dim) % 7).astype(mx.float32) / 16).astype(dtype)
        angles = mx.arange(length * head_dim // 2).reshape(length, head_dim // 2)
        cos = (angles % 7 - 3).astype(mx.float32) / 4
        sin = (angles % 5 - 2).astype(mx.float32) / 4
        self._assert_native_parity(q, k, wq, wk, cos, sin, heads, head_dim)

    @pytest.mark.parametrize("seed", [0, 42])
    @pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
    @pytest.mark.parametrize(
        ("dtype", "max_abs_error"), [(mx.bfloat16, 1 / 64), (mx.float16, 0.002), (mx.float32, 1e-5)]
    )
    @pytest.mark.parametrize("view", ["contiguous", "packed", "transposed"])
    def test_seeded_random_native_parity(self, seed, head_dim, dtype, max_abs_error, view):
        self._require_kernel(head_dim)
        mx.random.seed(seed)
        batch, length, heads = 2, 257, 3
        width = heads * head_dim
        if view == "packed":
            packed = mx.random.normal((batch, 2 * length, 6 * width)).astype(dtype)
            q = packed[:, 1::2, 1 : 2 * width : 2]
            k = packed[:, 1::2, 2 * width + 1 : 4 * width : 2]
        else:
            q = mx.random.normal((batch, length, width)).astype(dtype)
            k = mx.random.normal((batch, length, width)).astype(dtype)
            if view == "transposed":
                q = mx.contiguous(q.transpose(0, 2, 1)).transpose(0, 2, 1)
                k = mx.contiguous(k.transpose(2, 1, 0)).transpose(2, 1, 0)
        wq = mx.random.uniform(0.5, 1.5, (2 * head_dim,)).astype(dtype)[1::2]
        wk = mx.random.uniform(0.5, 1.5, (2 * head_dim,)).astype(dtype)[::2]
        angles = mx.random.uniform(-mx.pi, mx.pi, (head_dim, 2 * length))
        cos = mx.cos(angles).T[1::2, 1::2]
        sin = mx.sin(angles).T[1::2, 1::2]
        actual = fused_qk_norm_rope(q, k, wq, wk, cos, sin, heads, head_dim)
        assert actual is not None
        expected = [
            Qwen21Attention._apply_rope(
                mx.fast.rms_norm(x.reshape(batch, length, heads, head_dim), w, 1e-6), cos, sin
            ).transpose(0, 2, 1, 3)
            for x, w in ((q, wq), (k, wk))
        ]
        mx.eval(actual, expected)
        for result, reference in zip(actual, expected, strict=True):
            assert result.shape == reference.shape
            assert result.dtype == reference.dtype
            error = mx.max(mx.abs(result.astype(mx.float32) - reference.astype(mx.float32))).item()
            differing_fraction = mx.mean((result != reference).astype(mx.float32)).item()
            assert error <= max_abs_error, (error, differing_fraction)
            assert differing_fraction < 1e-3, (error, differing_fraction)

    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("view", ["packed", "transposed", "broadcast"])
    def test_strided_projections_weights_and_rope(self, head_dim, view):
        self._require_kernel(head_dim)
        batch, length, heads = 2, 7, 3
        width = heads * head_dim
        packed = mx.arange(batch * 2 * length * 6 * width).reshape(batch, 2 * length, 6 * width)
        packed = ((packed % 17 - 8).astype(mx.float32) / 8).astype(mx.bfloat16)
        q = packed[:, 1::2, 1 : 2 * width : 2]
        k = packed[:, 1::2, 2 * width + 1 : 4 * width : 2]
        wq = (0.75 + mx.arange(2 * head_dim).astype(mx.float32) % 9 / 16).astype(mx.bfloat16)[1::2]
        wk = (1.25 - mx.arange(2 * head_dim).astype(mx.float32) % 7 / 16)[::2]
        angles = mx.arange(2 * length * head_dim).reshape(head_dim, 2 * length)
        cos = ((angles % 7 - 3).astype(mx.float32) / 4).T[1::2, 1::2]
        sin = ((angles % 5 - 2).astype(mx.float32) / 4).T[::2, ::2]
        if view == "transposed":
            q = mx.contiguous(q.transpose(0, 2, 1)).transpose(0, 2, 1)
            k = mx.contiguous(k.transpose(2, 1, 0)).transpose(2, 1, 0)
        elif view == "broadcast":
            q = mx.broadcast_to(q[:1, :1], q.shape)
            k = mx.broadcast_to(k[:1, :1], k.shape)
            wq = mx.broadcast_to(wq[:1], wq.shape)
            wk = mx.broadcast_to(wk[:1], wk.shape)
            cos = mx.broadcast_to(cos[:1], cos.shape)
            sin = mx.broadcast_to(sin[:1], sin.shape)
        self._assert_native_parity(q, k, wq, wk, cos, sin, heads, head_dim)

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_attention_consumes_head_major_outputs(self, head_dim):
        self._require_kernel(head_dim)
        attention = Qwen21Attention(dim=2 * head_dim, num_heads=2, head_dim=head_dim)
        identity = mx.eye(2 * head_dim, dtype=mx.bfloat16)
        attention.to_q.weight = identity
        attention.to_k.weight = -identity
        attention.to_v.weight = identity * 0.5
        attention.norm_q.weight = mx.full((head_dim,), 0.875, dtype=mx.bfloat16)
        attention.norm_k.weight = mx.full((head_dim,), 1.125, dtype=mx.bfloat16)
        indices = mx.arange(2 * 17 * 2 * head_dim).reshape(2, 17, 2 * head_dim)
        hidden_states = ((indices % 17 - 8).astype(mx.float32) / 8).astype(mx.bfloat16)
        angles = mx.arange(17 * head_dim // 2).reshape(17, head_dim // 2).astype(mx.float32) / 32
        cos, sin = mx.cos(angles), mx.sin(angles)
        actual = attention._project(hidden_states, cos, sin)
        attention.use_fused_prologue = False
        expected = attention._project(hidden_states, cos, sin)
        mx.eval(actual, expected)
        for result, reference in zip(actual, expected, strict=True):
            assert result.shape == (2, 2, 17, head_dim)
            assert result.dtype == reference.dtype
            assert mx.array_equal(result, reference)

    @pytest.mark.parametrize("head_dim", [0, 32, 96, 576, 1024])
    def test_unsupported_head_dim_is_unavailable(self, head_dim):
        assert not fused_qk_norm_rope_available(head_dim)

    @pytest.mark.parametrize("mismatch", ["rope_width", "rope_length", "weight", "projection", "dtype"])
    def test_unsupported_inputs_request_composed_fallback(self, mismatch):
        self._require_kernel(64)
        q, k = mx.ones((1, 3, 128)), mx.ones((1, 3, 128))
        wq, wk = mx.ones((64,)), mx.ones((64,))
        cos, sin = mx.ones((3, 32)), mx.zeros((3, 32))
        if mismatch == "rope_width":
            cos = mx.ones((3, 64))
        elif mismatch == "rope_length":
            sin = mx.zeros((2, 32))
        elif mismatch == "weight":
            wk = mx.ones((32,))
        elif mismatch == "projection":
            k = mx.ones((1, 3, 64))
        else:
            q = q.astype(mx.int32)
        assert fused_qk_norm_rope(q, k, wq, wk, cos, sin, 2, 64) is None

    def test_disable_switch_applies_to_direct_calls(self, monkeypatch):
        monkeypatch.setenv("MFLUX_QWEN21_DISABLE_FUSED_PROLOGUE", "1")
        q = mx.ones((1, 3, 128))
        w = mx.ones((64,))
        cos, sin = mx.ones((3, 32)), mx.zeros((3, 32))
        assert not fused_qk_norm_rope_available(64)
        assert fused_qk_norm_rope(q, q, w, w, cos, sin, 2, 64) is None

    @pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
    def test_kernel_creation_failure_preserves_composed_attention(self, monkeypatch, error_type):
        if not mx.metal.is_available() or mx.default_device() != mx.gpu:
            pytest.skip("Kernel construction requires the Metal device")
        monkeypatch.delenv("MFLUX_QWEN21_DISABLE_FUSED_PROLOGUE", raising=False)
        monkeypatch.setattr(qwen21_fused_kernels, "_fused_qk_norm_rope_kernel", None)
        monkeypatch.setattr(qwen21_fused_kernels, "_fused_qk_norm_rope_unavailable", False)
        attempts = 0

        def unavailable_kernel(**kwargs):
            nonlocal attempts
            attempts += 1
            raise error_type("Metal kernel construction is unsupported")

        monkeypatch.setattr(mx.fast, "metal_kernel", unavailable_kernel)
        mx.random.seed(42)
        hidden_states = mx.random.normal((2, 7, 128))
        angles = mx.random.normal((7, 32))
        cos, sin = mx.cos(angles), mx.sin(angles)
        for _ in range(2):
            attention = Qwen21Attention(dim=128, num_heads=2, head_dim=64)
            actual = attention(hidden_states, cos, sin, attn_mask=None)
            precision = attention.compute_precision
            hidden = precision.to_compute(hidden_states)
            q = attention.to_q(hidden).reshape(2, 7, 2, 64)
            k = attention.to_k(hidden).reshape(2, 7, 2, 64)
            v = attention.to_v(hidden).reshape(2, 7, 2, 64).transpose(0, 2, 1, 3)
            q = Qwen21Attention._apply_rope(mx.fast.rms_norm(q, attention.norm_q.weight, 1e-6), cos, sin)
            k = Qwen21Attention._apply_rope(mx.fast.rms_norm(k, attention.norm_k.weight, 1e-6), cos, sin)
            attended = mx.fast.scaled_dot_product_attention(
                q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v, scale=64**-0.5
            )
            expected = attention.to_out[0](attended.transpose(0, 2, 1, 3).reshape(2, 7, 128))
            expected = precision.to_stream(expected, hidden_states.dtype)
            mx.eval(actual, expected)
            assert mx.array_equal(actual, expected)
            assert not attention.use_fused_prologue
        assert fused_qk_norm_rope(hidden_states, hidden_states, mx.ones((64,)), mx.ones((64,)), cos, sin, 2, 64) is None
        assert attempts == 1

    @pytest.mark.parametrize("eps", [1e-5, 1e-3])
    @pytest.mark.parametrize("mutate_norm", [False, True])
    def test_non_default_eps_uses_native_norm(self, eps, mutate_norm):
        attention = Qwen21Attention(dim=128, num_heads=2, head_dim=64, eps=1e-6 if mutate_norm else eps)
        if mutate_norm:
            attention.norm_q.eps = eps
            attention.norm_k.eps = eps
        hidden_states = mx.full((1, 3, 128), 0.001)
        cos, sin = mx.ones((3, 32)), mx.zeros((3, 32))
        actual_q, actual_k, _ = attention._project(hidden_states, cos, sin)
        q = attention.to_q(hidden_states).reshape(1, 3, 2, 64)
        k = attention.to_k(hidden_states).reshape(1, 3, 2, 64)
        expected_q = mx.fast.rms_norm(q, attention.norm_q.weight, eps).transpose(0, 2, 1, 3)
        expected_k = mx.fast.rms_norm(k, attention.norm_k.weight, eps).transpose(0, 2, 1, 3)
        mx.eval(actual_q, actual_k, expected_q, expected_k)
        assert mx.array_equal(actual_q, expected_q)
        assert mx.array_equal(actual_k, expected_k)

    def test_cpu_device_requests_composed_fallback(self):
        with mx.stream(mx.cpu):
            assert not fused_qk_norm_rope_available(64)
            q, w = mx.ones((1, 3, 128)), mx.ones((64,))
            cos, sin = mx.ones((3, 32)), mx.zeros((3, 32))
            assert fused_qk_norm_rope(q, q, w, w, cos, sin, 2, 64) is None

    @staticmethod
    def _require_kernel(head_dim):
        if not fused_qk_norm_rope_available(head_dim):
            pytest.skip("Qwen21 fused prologue is disabled or Metal is unavailable")

    @staticmethod
    def _assert_native_parity(q, k, wq, wk, cos, sin, heads, head_dim):
        actual = fused_qk_norm_rope(q, k, wq, wk, cos, sin, heads, head_dim)
        assert actual is not None
        batch, length, _ = q.shape
        expected = [
            Qwen21Attention._apply_rope(
                mx.fast.rms_norm(x.reshape(batch, length, heads, head_dim), w, 1e-6), cos, sin
            ).transpose(0, 2, 1, 3)
            for x, w in ((q, wq), (k, wk))
        ]
        mx.eval(actual, expected)
        for result, reference in zip(actual, expected, strict=True):
            assert result.shape == (batch, heads, length, head_dim)
            assert result.dtype == reference.dtype
            if result.dtype == mx.float32:
                assert mx.allclose(result, reference, atol=1e-6, rtol=1e-6)
            else:
                assert mx.array_equal(result, reference)


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
class TestReferenceBlockFusion:
    # The cached denoise pass of forward_reference: fused Q/K prologue, broadcast modulation rows and the
    # compiled gated residual give the same values as the composed ops.
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_cached_attention_fused_prologue_is_bit_identical(self, head_dim):
        if not fused_qk_norm_rope_available(head_dim):
            pytest.skip("fused Q/K norm + RoPE kernel unavailable")
        mx.random.seed(0)
        attention = Qwen21Attention(dim=2 * head_dim, num_heads=2, head_dim=head_dim)
        attention.update(
            {
                "to_q": {"weight": mx.random.normal((2 * head_dim, 2 * head_dim)).astype(mx.bfloat16) / 16},
                "to_k": {"weight": mx.random.normal((2 * head_dim, 2 * head_dim)).astype(mx.bfloat16) / 16},
                "to_v": {"weight": mx.random.normal((2 * head_dim, 2 * head_dim)).astype(mx.bfloat16) / 16},
            }
        )
        # the checkpoints store the Q/K norm weights in the same dtype as the projections
        attention.norm_q.weight = attention.norm_q.weight.astype(mx.bfloat16)
        attention.norm_k.weight = attention.norm_k.weight.astype(mx.bfloat16)
        x = mx.random.normal((1, 24, 2 * head_dim)).astype(mx.bfloat16)
        angles = mx.arange(24 * head_dim // 2).reshape(24, head_dim // 2).astype(mx.float32) / 64
        rope = (mx.cos(angles), mx.sin(angles))
        prefix = tuple(mx.random.normal((1, 2, 8, head_dim)).astype(mx.bfloat16) for _ in range(2))
        fused, _ = attention.forward_reference(x, None, rope, cached=prefix)
        attention.use_fused_prologue = False
        composed, _ = attention.forward_reference(x, None, rope, cached=prefix)
        mx.eval(fused, composed)
        assert fused.dtype == composed.dtype
        assert mx.array_equal(fused, composed)

    def test_select_rows_without_mask_broadcasts_the_target_row(self):
        value = mx.random.normal((5, 32)).astype(mx.bfloat16)
        mask = mx.ones((7,), dtype=mx.bool_)
        broadcast = Qwen21TransformerBlock.select_rows(value, None)
        masked = Qwen21TransformerBlock.select_rows(value, mask)
        assert mx.array_equal(mx.broadcast_to(broadcast, masked.shape), masked)

    def test_gated_residual_matches_composed_ops(self):
        x, gate, y = (mx.random.normal((1, 9, 32)).astype(mx.bfloat16) for _ in range(3))
        assert mx.array_equal(_gated_residual(x, gate, y), x + mx.tanh(gate) * y)
