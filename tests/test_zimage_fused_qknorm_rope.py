import mlx.core as mx
import pytest

from mflux.models.qwen21.model.qwen21_transformer import qwen21_fused_kernels
from mflux.models.qwen21.model.qwen21_transformer.qwen21_fused_kernels import (
    fused_qk_norm_rope,
    fused_qk_norm_rope_available,
)
from mflux.models.z_image.model.z_image_transformer.attention import ZImageAttention

pytestmark = pytest.mark.fast

HEADS, HEAD_DIM = 30, 128


class TestZImageFusedQkNormRope:
    @staticmethod
    def _freqs(length: int) -> mx.array:
        angles = mx.random.uniform(-mx.pi, mx.pi, (length, HEAD_DIM // 2))
        return mx.stack([mx.cos(angles), mx.sin(angles)], axis=-1)

    @staticmethod
    def _run(attn, x, freqs, monkeypatch, fused: bool):
        if fused:
            monkeypatch.delenv(ZImageAttention.DISABLE_FUSED_ENV, raising=False)
        else:
            monkeypatch.setenv(ZImageAttention.DISABLE_FUSED_ENV, "1")
        out = attn(x, attention_mask=None, freqs_cis=freqs)
        mx.eval(out)
        return out.astype(mx.float32)

    @staticmethod
    def _require_kernel(eps: float = 1e-5):
        if not fused_qk_norm_rope_available(HEAD_DIM, eps, ZImageAttention.DISABLE_FUSED_ENV):
            pytest.skip("Metal kernel unavailable")

    @staticmethod
    def _compare_op(length: int, eps) -> None:
        mx.random.seed(3)
        q = mx.random.normal((1, length, HEADS * HEAD_DIM)).astype(mx.bfloat16)
        k = mx.random.normal((1, length, HEADS * HEAD_DIM)).astype(mx.bfloat16)
        wq = mx.random.uniform(0.5, 1.5, (HEAD_DIM,)).astype(mx.bfloat16)
        wk = mx.random.uniform(0.5, 1.5, (HEAD_DIM,)).astype(mx.bfloat16)
        f = TestZImageFusedQkNormRope._freqs(length)
        got = fused_qk_norm_rope(
            q, k, wq, wk, f[..., 0], f[..., 1], HEADS, HEAD_DIM, eps=eps, disable_env=ZImageAttention.DISABLE_FUSED_ENV
        )
        assert got is not None
        for x, w, g in ((q, wq, got[0]), (k, wk, got[1])):
            n = mx.fast.rms_norm(x.reshape(1, length, HEADS, HEAD_DIM), w, float(eps))
            ref = ZImageAttention.apply_rotary_emb(n, f).transpose(0, 2, 1, 3)
            mx.eval(g, ref)
            a, b = g.astype(mx.float32), ref.astype(mx.float32)
            err = mx.max(mx.abs(a - b)).item()
            cos = (mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b))).item()
            print(f"op L={length} eps={eps} max_abs={err:.5f} cosine={cos:.8f} dtype={g.dtype}")
            assert g.dtype == ref.dtype
            assert err <= 1 / 64
            assert cos > 0.99999

    @pytest.mark.parametrize("length", [4224, 2304 + 32])  # 1024x1024 and 768x768 (+32 text) tokens
    @pytest.mark.parametrize("seed", [0, 1])
    def test_fused_matches_composed(self, length, seed, monkeypatch):
        self._require_kernel()
        mx.random.seed(seed)
        attn = ZImageAttention(dim=HEADS * HEAD_DIM, n_heads=HEADS)
        attn.norm_q.weight = mx.random.uniform(0.5, 1.5, (HEAD_DIM,)).astype(mx.bfloat16)
        attn.norm_k.weight = mx.random.uniform(0.5, 1.5, (HEAD_DIM,)).astype(mx.bfloat16)
        attn.set_dtype(mx.bfloat16)
        x = mx.random.normal((1, length, HEADS * HEAD_DIM)).astype(mx.bfloat16)
        freqs = self._freqs(length)
        ref = self._run(attn, x, freqs, monkeypatch, fused=False)
        got = self._run(attn, x, freqs, monkeypatch, fused=True)
        err = mx.max(mx.abs(got - ref)).item()
        cos = (mx.sum(got * ref) / (mx.linalg.norm(got) * mx.linalg.norm(ref))).item()
        print(f"L={length} seed={seed} max_abs={err:.5f} cosine={cos:.8f}")
        assert err <= 1 / 64
        assert cos > 0.9999

    @pytest.mark.parametrize("length", [4224, 2336])
    def test_op_level_q_k_match(self, length):
        self._require_kernel()
        self._compare_op(length, 1e-5)

    @pytest.mark.parametrize("eps", [0, 1, 1e-6])
    def test_integer_and_default_eps_build_a_valid_kernel(self, eps):
        # an int eps used to render as the invalid Metal literal "0f"
        self._require_kernel(eps)
        self._compare_op(256, eps)

    def test_disable_env_falls_back(self, monkeypatch):
        monkeypatch.setenv(ZImageAttention.DISABLE_FUSED_ENV, "1")
        assert not fused_qk_norm_rope_available(HEAD_DIM, 1e-5, ZImageAttention.DISABLE_FUSED_ENV)

    def test_failed_eps_kernel_does_not_disable_the_default_kernel(self, monkeypatch):
        self._require_kernel(1e-6)
        monkeypatch.delenv(ZImageAttention.DISABLE_FUSED_ENV, raising=False)
        monkeypatch.setattr(qwen21_fused_kernels, "_fused_qk_norm_rope_unavailable", False)
        monkeypatch.setattr(qwen21_fused_kernels, "_fused_qk_norm_rope_unavailable_eps", set())
        monkeypatch.setattr(qwen21_fused_kernels, "_fused_qk_norm_rope_kernels", {})
        real = mx.fast.metal_kernel

        def failing_for_eps(**kwargs):
            if kwargs["name"] != "fused_qk_norm_rope":
                raise RuntimeError("unsupported")
            return real(**kwargs)

        monkeypatch.setattr(mx.fast, "metal_kernel", failing_for_eps)
        assert not fused_qk_norm_rope_available(HEAD_DIM, 1e-5, ZImageAttention.DISABLE_FUSED_ENV)
        assert fused_qk_norm_rope_available(HEAD_DIM, 1e-6, ZImageAttention.DISABLE_FUSED_ENV)
