import mlx.core as mx
import pytest

from mflux.models.flux2.model.flux2_transformer.flux2_kv_cache import Flux2KVCache


def _kv(tokens: int, offset: float) -> tuple[mx.array, mx.array]:
    key = mx.arange(tokens, dtype=mx.float32).reshape(1, 1, tokens, 1) + offset
    return key, key + 100


@pytest.mark.fast
def test_store_reference_keeps_the_trailing_reference_tokens():
    cache = Flux2KVCache(num_double_layers=1, num_single_layers=1)
    cache.configure(mode="extract", num_ref_tokens=2)
    key, value = _kv(5, 0)
    cache.store_reference("double", 0, key, value)
    cache.configure(mode="cached", num_ref_tokens=2)
    stored_key, stored_value = cache.append_reference("double", 0, key[:, :, :0], value[:, :, :0])
    assert stored_key.reshape(-1).tolist() == [3, 4]
    assert stored_value.reshape(-1).tolist() == [103, 104]


@pytest.mark.fast
def test_arrays_lists_every_stored_reference_for_evaluation():
    # #1003: the edit pipeline evaluates these with the latents at the end of the extract step.
    cache = Flux2KVCache(num_double_layers=2, num_single_layers=1)
    assert cache.arrays() == []
    cache.configure(mode="extract", num_ref_tokens=2)
    cache.store_reference("double", 0, *_kv(4, 0))
    cache.store_reference("single", 0, *_kv(4, 10))
    arrays = cache.arrays()
    assert len(arrays) == 4  # key and value of the two filled layers; the empty double layer adds nothing
    mx.eval(*arrays)
    assert [a.reshape(-1).tolist() for a in arrays] == [[2, 3], [102, 103], [12, 13], [112, 113]]


@pytest.mark.fast
def test_store_reference_does_nothing_outside_the_extract_step():
    cache = Flux2KVCache(num_double_layers=1, num_single_layers=1)
    cache.configure(mode="cached", num_ref_tokens=2)
    cache.store_reference("double", 0, *_kv(4, 0))
    assert cache.arrays() == []
