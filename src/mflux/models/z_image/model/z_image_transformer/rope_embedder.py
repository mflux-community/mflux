import mlx.core as mx


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: list[int] | None = None,
        axes_lens: list[int] | None = None,
    ):
        if axes_dims is None:
            axes_dims = [16, 56, 56]
        if axes_lens is None:
            axes_lens = [64, 128, 128]

        self.axes_dims = axes_dims
        self.freqs_cis = RopeEmbedder._precompute_freqs_cis(axes_dims, axes_lens, theta)
        # Materialise the tables now: they are plain arrays outside the module's parameter tree,
        # so nothing else evaluates them. Left lazy, they are captured as a pending graph by the
        # first mx.compile of predict and diverge from a later eager touch, changing the output
        # across generate_image calls in the same process (issue #714). Three small float32 arrays.
        mx.eval(*self.freqs_cis)

    def __call__(self, ids: mx.array) -> mx.array:
        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i].astype(mx.int32)
            result.append(self.freqs_cis[i][index])
        return mx.concatenate(result, axis=1)

    @staticmethod
    def _precompute_freqs_cis(axes_dims, axes_lens, theta) -> list[mx.array]:
        freqs_cis = []
        for d, e in zip(axes_dims, axes_lens):
            freqs = 1.0 / (theta ** (mx.arange(0, d, 2, dtype=mx.float32) / d))
            timestep = mx.arange(e, dtype=mx.float32)
            freqs = mx.outer(timestep, freqs)
            cos_freqs = mx.cos(freqs)
            sin_freqs = mx.sin(freqs)
            freqs_cis_i = mx.stack([cos_freqs, sin_freqs], axis=-1)
            freqs_cis.append(freqs_cis_i)
        return freqs_cis
