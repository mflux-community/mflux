import mlx.core as mx

from mflux.models.z_image.model.z_image_transformer.rope_embedder import RopeEmbedder


def _rope() -> RopeEmbedder:
    # The geometry the Z-Image transformer builds (axes_dims / axes_lens from
    # z_image_transformer.transformer). Constructed directly: no weights, no model load.
    return RopeEmbedder(theta=256.0, axes_dims=[32, 48, 48], axes_lens=[1024, 512, 512])


def _pos_ids() -> mx.array:
    # (N, 3): one token index and two spatial indices, each below its axis length.
    n = mx.arange(8)
    return mx.stack([n, n % 512, n % 512], axis=1).astype(mx.int32)


def test_rope_output_survives_a_recompile_after_the_tables_are_touched():
    # Z-Image wraps `predict` in mx.compile and rebuilds it every generate_image call. When the
    # RoPE cos/sin tables are left unevaluated, the first compiled call inlines their pending
    # graph; a later eager touch of the same tables materialises them; and the next fresh compile
    # then captures those materialised values, so one seed produces a different image depending on
    # what ran earlier in the process (issue #714). Evaluating the tables in __init__ makes them
    # constants before any compile sees them. Reproduces on mlx 0.32.x, which mflux 0.19 pins;
    # mlx 0.31 traced the tables identically. Delete the mx.eval in RopeEmbedder.__init__ and this
    # goes red on 0.32.
    ids = _pos_ids()
    rope = _rope()

    first_fn = mx.compile(lambda pos: rope(pos))
    first = first_fn(ids)
    mx.eval(first)

    mx.eval(*rope.freqs_cis)  # an eager touch between the two compiled functions

    second_fn = mx.compile(lambda pos: rope(pos))
    second = second_fn(ids)
    mx.eval(second)

    assert mx.array_equal(first, second).item()
