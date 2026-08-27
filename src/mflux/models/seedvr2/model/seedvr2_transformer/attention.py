import mlx.core as mx
from mlx import nn

from mflux.models.seedvr2.model.seedvr2_transformer.rms_norm import RMSNorm
from mflux.models.seedvr2.model.seedvr2_transformer.rope import RoPEModule
from mflux.models.seedvr2.model.seedvr2_transformer.window import WindowPartitioner


class MMAttention(nn.Module):
    def __init__(
        self,
        vid_dim: int,
        txt_dim: int,
        heads: int = 20,
        head_dim: int = 128,
        qk_bias: bool = False,
        qk_norm_eps: float = 1e-5,
        rope_dim: int = 128,
        rope_freqs_for: str = "lang",
        rope_on_text: bool = True,
        shared_weights: bool = False,
        window: tuple[int, int, int] = (4, 3, 3),
        shift: bool = False,
    ):
        super().__init__()
        self.shared_weights = shared_weights
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.window = window
        self.shift = shift
        self.rope_on_text = rope_on_text

        inner_dim = heads * head_dim

        self.proj_qkv_vid = nn.Linear(vid_dim, 3 * inner_dim, bias=qk_bias)
        self.proj_out_vid = nn.Linear(inner_dim, vid_dim, bias=True)
        self.norm_q_vid = RMSNorm(head_dim, eps=qk_norm_eps)
        self.norm_k_vid = RMSNorm(head_dim, eps=qk_norm_eps)

        if shared_weights:
            self.proj_qkv_txt = self.proj_qkv_vid
            self.proj_out_txt = self.proj_out_vid
            self.norm_q_txt = self.norm_q_vid
            self.norm_k_txt = self.norm_k_vid
        else:
            self.proj_qkv_txt = nn.Linear(txt_dim, 3 * inner_dim, bias=qk_bias)
            self.proj_out_txt = nn.Linear(inner_dim, txt_dim, bias=True)
            self.norm_q_txt = RMSNorm(head_dim, eps=qk_norm_eps)
            self.norm_k_txt = RMSNorm(head_dim, eps=qk_norm_eps)

        self.rope = RoPEModule(dim=rope_dim, freqs_for=rope_freqs_for)

    def __call__(self, vid, txt, vid_shape, txt_shape):
        B, L, Bt, Lt = vid.shape[0], vid.shape[1], txt.shape[0], txt.shape[1]

        # 1. Project to QKV and Partition
        qkv_vid = self.proj_qkv_vid(vid.reshape(-1, vid.shape[-1])).reshape(-1, 3, self.heads, self.head_dim)
        qkv_txt = self.proj_qkv_txt(txt.reshape(-1, txt.shape[-1])).reshape(-1, 3, self.heads, self.head_dim)

        partitioner = WindowPartitioner(vid_shape, self.window, self.shift)
        qkv_vid = partitioner.partition(qkv_vid)

        # 2. Normalize and repeat text
        q_vid, k_vid, v_vid = self.norm_q_vid(qkv_vid[:, 0]), self.norm_k_vid(qkv_vid[:, 1]), qkv_vid[:, 2]
        q_txt, k_txt, v_txt = self.norm_q_txt(qkv_txt[:, 0]), self.norm_k_txt(qkv_txt[:, 1]), qkv_txt[:, 2]

        counts, txt_len = partitioner.window_counts, txt_shape[:, 0]
        win_to_batch = MMAttention._window_to_batch(counts)
        qkv_t_rep = self._repeat_text_for_windows(mx.stack([q_txt, k_txt, v_txt], axis=1), txt_len, win_to_batch)
        q_txt_rep, k_txt_rep, v_txt_rep = qkv_t_rep[:, 0], qkv_t_rep[:, 1], qkv_t_rep[:, 2]

        # 3. Apply RoPE
        if self.rope_on_text:
            q_vid, k_vid, q_txt_rep, k_txt_rep = self.rope(
                vid_q=q_vid,
                vid_k=k_vid,
                vid_shape=partitioner.window_shapes,
                txt_q=q_txt_rep,
                txt_k=k_txt_rep,
                txt_shape=txt_shape[win_to_batch],
            )
        else:
            q_vid, k_vid = self.rope(
                vid_q=q_vid,
                vid_k=k_vid,
                vid_shape=partitioner.window_shapes,
            )

        # 4. Attention
        vid_lens = mx.prod(partitioner.window_shapes, axis=1)
        qkv = self._concat_with_text(
            mx.stack([q_vid, k_vid, v_vid], axis=1),
            mx.stack([q_txt_rep, k_txt_rep, v_txt_rep], axis=1),
            vid_lens,
            txt_len,
            counts,
        )

        win_lens = vid_lens + txt_len[win_to_batch]
        windows = mx.split(qkv, mx.cumsum(win_lens[:-1]).tolist())

        out = []
        for w in windows:
            q, k, v = [x[None].transpose(0, 2, 1, 3) for x in [w[:, 0], w[:, 1], w[:, 2]]]
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
            out.append(o.transpose(0, 2, 1, 3).squeeze(0))

        # 5. Coalesce and Project Out
        out = mx.concatenate(out, axis=0).reshape(-1, self.heads * self.head_dim)
        vid_out, txt_out = self._unconcat_and_coalesce(out, vid_lens, txt_len, counts, win_to_batch)

        return (
            self.proj_out_vid(partitioner.reverse(vid_out)).reshape(B, L, -1),
            self.proj_out_txt(txt_out).reshape(Bt, Lt, -1),
        )

    @staticmethod
    def _window_to_batch(counts: list[int]) -> mx.array:
        # Each batch element is partitioned into its own number of windows, so expanding a
        # per-batch value to per-window is a gather, not a repeat: mx.repeat only takes a
        # scalar count. Passing mx.array(counts) happened to work for a single batch element
        # on mlx <= 0.32.1, which coerced the one-element array to an int, and raised a
        # TypeError for anything longer; mlx 0.32.2 dropped that coercion entirely.
        return mx.concatenate([mx.full((count,), index, dtype=mx.int32) for index, count in enumerate(counts)])

    @staticmethod
    def _repeat_text_for_windows(txt, txt_len, win_to_batch):
        B, L = txt_len.shape[0], int(txt_len[0])
        txt = txt.reshape(B, L, *txt.shape[1:])
        return txt[win_to_batch].reshape(-1, *txt.shape[2:])

    @staticmethod
    def _concat_with_text(vid, txt, vid_lens, txt_len, counts):
        v_parts = mx.split(vid, mx.cumsum(vid_lens[:-1]).tolist())
        t_parts = mx.split(txt, mx.arange(int(txt_len[0]), txt.shape[0], int(txt_len[0])).tolist())
        parts = [p for pair in zip(v_parts, t_parts) for p in pair]
        return mx.concatenate(parts, axis=0)

    @staticmethod
    def _unconcat_and_coalesce(combined, vid_lens, txt_len, counts, win_to_batch):
        lens = mx.stack([vid_lens, txt_len[win_to_batch]], axis=1).reshape(-1)
        parts = mx.split(combined, mx.cumsum(lens[:-1]).tolist())

        vid_out = mx.concatenate(parts[0::2], axis=0)
        t_parts = parts[1::2]

        final_txt, offset = [], 0
        for count in counts:
            final_txt.append(mx.stack(t_parts[offset : offset + count]).mean(axis=0))
            offset += count
        return vid_out, mx.concatenate(final_txt, axis=0)
