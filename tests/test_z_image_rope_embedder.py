import mlx.core as mx

from mflux.models.z_image.model.z_image_transformer.rope_embedder import RopeEmbedder


class TestRopeEmbedderEagerEval:
    @staticmethod
    def _rope() -> RopeEmbedder:
        # The geometry z_image_transformer builds; constructed directly, no weights, no model load.
        return RopeEmbedder(theta=256.0, axes_dims=[32, 48, 48], axes_lens=[1024, 512, 512])

    @staticmethod
    def _pos_ids() -> mx.array:
        n = mx.arange(8)
        return mx.stack([n, n % 512, n % 512], axis=1).astype(mx.int32)

    def test_rope_output_survives_a_recompile_after_the_tables_are_touched(self):
        # Regression for issue #714: a fresh mx.compile after the RoPE tables are eagerly evaluated
        # must give the same output as the first compile. Left lazy, the tables are captured as a
        # pending graph and the next compile picks up different values, so a seed's image depended
        # on process history. Reproduces on the pinned mlx 0.32 (mlx 0.31 did not drift); delete the
        # mx.eval in RopeEmbedder.__init__ to see it go red.
        ids = self._pos_ids()
        rope = self._rope()

        first_fn = mx.compile(lambda pos: rope(pos))
        first = first_fn(ids)
        mx.eval(first)

        mx.eval(*rope.freqs_cis)  # eager touch between the two compiled functions

        second_fn = mx.compile(lambda pos: rope(pos))
        second = second_fn(ids)
        mx.eval(second)

        assert mx.array_equal(first, second).item()
