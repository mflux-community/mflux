import mlx.core as mx
import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux.variants.controlnet.transformer_controlnet import TransformerControlnet


@pytest.mark.fast
def test_the_controlnet_single_block_heads_start_at_zero():
    # Like controlnet_blocks (and diffusers' zero_module), so a head the checkpoint lacks adds nothing
    # instead of a random residual.
    controlnet = TransformerControlnet(ModelConfig.dev(), num_transformer_blocks=0, num_single_transformer_blocks=1)

    head = controlnet.controlnet_single_blocks[0]

    assert not mx.any(head.weight).item()
    assert not mx.any(head.bias).item()
