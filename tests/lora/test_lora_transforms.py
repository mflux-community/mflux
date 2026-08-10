import mlx.core as mx
import pytest

from mflux.models.common.lora.mapping.lora_transforms import LoraTransforms

# Reference dimensions for a FLUX transformer (hidden size 3072).
HIDDEN = 3072
MLP = 4 * HIDDEN  # 12288


@pytest.mark.fast
class TestLoraTransforms:
    """Regression tests for fused-qkv LoRA tensor splitting.

    A fused qkv (or qkv+mlp) LoRA stores a single shared ``down`` projection of
    shape ``(rank, in_features)`` and a single ``up`` projection of shape
    ``(fused_out, rank)``. When the fused layer is split into its individual
    q/k/v(/mlp) components, only the ``up`` projection is sliced along its output
    dimension; the ``down`` projection is shared and must be returned at full
    rank. Splitting the rank dimension (which happened whenever ``rank`` was
    divisible by the number of splits, e.g. rank 64) produced a rank-mismatched
    adapter and a matmul shape error at the first denoise step.
    """

    @pytest.mark.parametrize("rank", [16, 32, 64])
    def test_double_block_down_is_shared_full_rank(self, rank: int):
        # Fused qkv down: (rank, hidden), shared across q/k/v.
        down = mx.random.normal((rank, HIDDEN))
        for split in (
            LoraTransforms.split_q_down,
            LoraTransforms.split_k_down,
            LoraTransforms.split_v_down,
        ):
            out = split(down)
            assert out.shape == (rank, HIDDEN)
            assert mx.array_equal(out, down)

    @pytest.mark.parametrize("rank", [16, 32, 64])
    def test_double_block_up_is_output_split(self, rank: int):
        # Fused qkv up: (3 * hidden, rank), split along the output dimension.
        up = mx.random.normal((3 * HIDDEN, rank))
        q = LoraTransforms.split_q_up(up)
        k = LoraTransforms.split_k_up(up)
        v = LoraTransforms.split_v_up(up)
        for part in (q, k, v):
            assert part.shape == (HIDDEN, rank)
        assert mx.array_equal(q, up[0:HIDDEN, :])
        assert mx.array_equal(k, up[HIDDEN : 2 * HIDDEN, :])
        assert mx.array_equal(v, up[2 * HIDDEN : 3 * HIDDEN, :])

    @pytest.mark.parametrize("rank", [16, 32, 64])
    def test_single_block_down_is_shared_full_rank(self, rank: int):
        # Fused qkv+mlp down (single-block linear1): (rank, hidden), shared.
        down = mx.random.normal((rank, HIDDEN))
        for split in (
            LoraTransforms.split_single_q_down,
            LoraTransforms.split_single_k_down,
            LoraTransforms.split_single_v_down,
            LoraTransforms.split_single_mlp_down,
        ):
            out = split(down)
            assert out.shape == (rank, HIDDEN)
            assert mx.array_equal(out, down)

    @pytest.mark.parametrize("rank", [16, 32, 64])
    def test_single_block_up_is_output_split(self, rank: int):
        # Fused qkv+mlp up (single-block linear1): (3*hidden + mlp, rank).
        fused_out = 3 * HIDDEN + MLP  # 21504
        up = mx.random.normal((fused_out, rank))
        q = LoraTransforms.split_single_q_up(up)
        k = LoraTransforms.split_single_k_up(up)
        v = LoraTransforms.split_single_v_up(up)
        mlp = LoraTransforms.split_single_mlp_up(up)
        assert q.shape == (HIDDEN, rank)
        assert k.shape == (HIDDEN, rank)
        assert v.shape == (HIDDEN, rank)
        assert mlp.shape == (MLP, rank)
        # Concatenating the parts reconstructs the fused up projection.
        assert mx.array_equal(mx.concatenate([q, k, v, mlp], axis=0), up)

    def test_fused_down_and_up_ranks_match_after_split(self):
        # End-to-end shape invariant: for a rank divisible by the split count
        # (the case that previously broke), the per-component down and up share
        # the same rank, so up @ down is well-formed.
        rank = 64
        down = mx.random.normal((rank, HIDDEN))
        up = mx.random.normal((3 * HIDDEN + MLP, rank))
        q_down = LoraTransforms.split_single_q_down(down)
        q_up = LoraTransforms.split_single_q_up(up)
        # up: (out, rank), down: (rank, in) -> product is (out, in).
        assert q_up.shape[1] == q_down.shape[0] == rank
        product = q_up @ q_down
        assert product.shape == (HIDDEN, HIDDEN)
