import mlx.core as mx
from mlx import nn


class RoPEModule(nn.Module):
    def __init__(self, dim: int = 64, freqs_for: str = "lang", theta: float = 10000.0, max_freq: float = 256.0):
        super().__init__()
        self.dim = dim
        self.freqs_for = freqs_for
        self.rope_dim = 3
        self.freq_dim = dim // self.rope_dim
        if freqs_for == "pixel":
            self.freqs = mx.linspace(1.0, max_freq / 2.0, self.freq_dim // 2, dtype=mx.float32) * mx.pi
        else:
            self.freqs = 1.0 / (
                theta ** (mx.arange(0, self.freq_dim, 2, dtype=mx.float32)[: (self.freq_dim // 2)] / self.freq_dim)
            )

    def __call__(
        self,
        vid_q: mx.array,
        vid_k: mx.array,
        vid_shape: mx.array,
        txt_q: mx.array | None = None,
        txt_k: mx.array | None = None,
        txt_shape: mx.array | None = None,
    ) -> tuple[mx.array, mx.array] | tuple[mx.array, mx.array, mx.array, mx.array]:
        if txt_q is None or txt_k is None or txt_shape is None:
            return RoPEModule._apply_rope_3d(
                vid_q,
                vid_k,
                vid_shape,
                self.freqs,
                self.rope_dim,
                self.freqs_for,
            )
        return RoPEModule._apply_mm_rope_3d(
            vid_q,
            vid_k,
            vid_shape,
            txt_q,
            txt_k,
            txt_shape,
            self.freqs,
            self.rope_dim,
            self.freqs_for,
        )

    @classmethod
    def _apply_rope_3d(
        cls,
        vid_q: mx.array,
        vid_k: mx.array,
        vid_shape: mx.array,
        freqs: mx.array,
        rope_dim: int = 3,
        freqs_for: str = "lang",
    ) -> tuple[mx.array, mx.array]:
        vid_freq_list = []
        if freqs_for == "pixel":
            # For pixel-space RoPE, the position grid is normalized with linspace(-1, 1).
            # We must use exact dimensions per sample/window; slicing a larger precomputed grid
            # changes spacing and introduces spatial bias.
            for b in range(vid_shape.shape[0]):
                f, h, w = int(vid_shape[b, 0]), int(vid_shape[b, 1]), int(vid_shape[b, 2])
                vid_freq = RoPEModule._get_axial_freqs(freqs, f, h, w, freqs_for=freqs_for).reshape(
                    -1, freqs.size * 2 * rope_dim
                )
                vid_freq_list.append(vid_freq)
        else:
            max_temporal = int(mx.max(vid_shape[:, 0]))
            max_height = int(mx.max(vid_shape[:, 1]))
            max_width = int(mx.max(vid_shape[:, 2]))

            clamp_temporal = min(max_temporal + 16, 1024)
            clamp_height = min(max_height + 4, 128)
            clamp_width = min(max_width + 4, 128)

            vid_freqs_full = RoPEModule._get_axial_freqs(
                freqs, clamp_temporal, clamp_height, clamp_width, freqs_for=freqs_for
            )

            for b in range(vid_shape.shape[0]):
                f, h, w = int(vid_shape[b, 0]), int(vid_shape[b, 1]), int(vid_shape[b, 2])
                vid_freq = vid_freqs_full[:f, :h, :w].reshape(-1, vid_freqs_full.shape[-1])
                vid_freq_list.append(vid_freq)

        vid_freqs = mx.concatenate(vid_freq_list, axis=0)

        vid_q = RoPEModule._apply_rotary_emb(vid_freqs[:, None, :], vid_q)
        vid_k = RoPEModule._apply_rotary_emb(vid_freqs[:, None, :], vid_k)

        return vid_q, vid_k

    @staticmethod
    def _rotate_half(x: mx.array) -> mx.array:
        x = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x[..., 0], x[..., 1]
        x = mx.stack([-x2, x1], axis=-1)
        return x.reshape(*x.shape[:-2], -1)

    @classmethod
    def _get_axial_freqs(cls, freqs: mx.array, *dims: int, freqs_for: str = "lang") -> mx.array:
        freq_dim_per_axis = len(freqs) * 2
        target_shape = list(dims) + [freq_dim_per_axis]
        all_freqs = []

        for ind, dim_size in enumerate(dims):
            if freqs_for == "pixel":
                pos = mx.linspace(-1.0, 1.0, dim_size, dtype=mx.float32)
            else:
                pos = mx.arange(dim_size, dtype=mx.float32)
            axis_freqs = mx.outer(pos, freqs.astype(mx.float32))
            axis_freqs = mx.repeat(axis_freqs, 2, axis=-1)

            shape = [1] * len(dims) + [freq_dim_per_axis]
            shape[ind] = dim_size
            axis_freqs = axis_freqs.reshape(*shape)
            all_freqs.append(axis_freqs)

        broadcasted = [mx.broadcast_to(f, target_shape) for f in all_freqs]
        output = mx.concatenate(broadcasted, axis=-1)
        return output

    @staticmethod
    def _apply_rotary_emb(freqs: mx.array, t: mx.array) -> mx.array:
        rot_dim = freqs.shape[-1]
        t_middle = t[..., :rot_dim]
        t_right = t[..., rot_dim:]

        t_dtype = t_middle.dtype
        freqs_f = freqs.astype(mx.float32)
        t_middle_f = t_middle.astype(mx.float32)
        cos_freqs = mx.cos(freqs_f)
        sin_freqs = mx.sin(freqs_f)
        t_transformed = (t_middle_f * cos_freqs) + (RoPEModule._rotate_half(t_middle_f) * sin_freqs)
        t_transformed = t_transformed.astype(t_dtype)

        if t_right.shape[-1] > 0:
            return mx.concatenate([t_transformed, t_right], axis=-1)
        return t_transformed

    @classmethod
    def _apply_mm_rope_3d(
        cls,
        vid_q: mx.array,
        vid_k: mx.array,
        vid_shape: mx.array,
        txt_q: mx.array,
        txt_k: mx.array,
        txt_shape: mx.array,
        freqs: mx.array,
        rope_dim: int = 3,
        freqs_for: str = "lang",
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        vid_freq_list = []
        txt_freq_list = []
        if freqs_for == "pixel":
            # Same rationale as _apply_rope_3d: compute exact grids in pixel mode.
            for b in range(vid_shape.shape[0]):
                f, h, w = int(vid_shape[b, 0]), int(vid_shape[b, 1]), int(vid_shape[b, 2])
                txt_len = int(txt_shape[b, 0])

                full = RoPEModule._get_axial_freqs(freqs, txt_len + f, h, w, freqs_for=freqs_for)
                vid_freq = full[txt_len : txt_len + f, :h, :w].reshape(-1, full.shape[-1])
                txt_freq = RoPEModule._get_axial_freqs(freqs, txt_len, freqs_for=freqs_for)
                txt_freq = mx.tile(txt_freq, (1, rope_dim))

                vid_freq_list.append(vid_freq)
                txt_freq_list.append(txt_freq)
        else:
            max_temporal = int(mx.max(vid_shape[:, 0] + txt_shape[:, 0]))
            max_height = int(mx.max(vid_shape[:, 1]))
            max_width = int(mx.max(vid_shape[:, 2]))
            max_txt_len = int(mx.max(txt_shape[:, 0]))

            clamp_temporal = min(max_temporal + 16, 1024)
            clamp_height = min(max_height + 4, 128)
            clamp_width = min(max_width + 4, 128)

            vid_freqs_full = RoPEModule._get_axial_freqs(
                freqs, clamp_temporal, clamp_height, clamp_width, freqs_for=freqs_for
            )
            txt_freqs_1d = RoPEModule._get_axial_freqs(freqs, min(max_txt_len + 16, 1024), freqs_for=freqs_for)

            for b in range(vid_shape.shape[0]):
                f, h, w = int(vid_shape[b, 0]), int(vid_shape[b, 1]), int(vid_shape[b, 2])
                txt_len = int(txt_shape[b, 0])

                vid_freq = vid_freqs_full[txt_len : txt_len + f, :h, :w].reshape(-1, vid_freqs_full.shape[-1])
                txt_freq = mx.tile(txt_freqs_1d[:txt_len], (1, rope_dim))

                vid_freq_list.append(vid_freq)
                txt_freq_list.append(txt_freq)

        vid_freqs = mx.concatenate(vid_freq_list, axis=0)
        txt_freqs = mx.concatenate(txt_freq_list, axis=0)

        vid_q = RoPEModule._apply_rotary_emb(vid_freqs[:, None, :], vid_q)
        vid_k = RoPEModule._apply_rotary_emb(vid_freqs[:, None, :], vid_k)

        txt_q = RoPEModule._apply_rotary_emb(txt_freqs[:, None, :], txt_q)
        txt_k = RoPEModule._apply_rotary_emb(txt_freqs[:, None, :], txt_k)

        return vid_q, vid_k, txt_q, txt_k
