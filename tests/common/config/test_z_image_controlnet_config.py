import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.z_image.variants.controlnet.transformer_controlnet import ZImageControlNetConfig
from mflux.models.z_image.weights.z_image_controlnet_weight_definition import ZImageControlnetWeightDefinition

# These are the regressions that shipped silently in the original port and produced a plausible but
# uncontrolled image (no crash), so they need explicit guards rather than a golden-image run.


@pytest.mark.fast
def test_control_residuals_are_strided_over_the_base_blocks():
    # The 15 control blocks feed the 30-block base transformer every OTHER block. A contiguous
    # [0..14] mapping lands residuals on the wrong blocks: control is ignored at low strength and
    # diverges to noise at high strength. The published checkpoint ships no config.json, so the
    # placement lives in the hardcoded defaults and must stay strided.
    cfg = ZImageControlNetConfig.defaults_union_2_1()
    assert cfg.control_layers_places == list(range(0, 30, 2))
    assert len(cfg.control_layers_places) == 15
    assert 0 in cfg.control_layers_places
    assert cfg.control_in_dim == 33  # control_all_x_embedder is [dim, 132]; 132 = f_patch * patch**2 * 33


@pytest.mark.fast
def test_controlnet_alias_resolves():
    config = ModelConfig.z_image_turbo_controlnet_union_2_1()
    assert "z-image-controlnet" in config.aliases
    assert config.controlnet_model == "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1"


@pytest.mark.fast
def test_quantization_skips_layers_not_divisible_by_group_size():
    # MLX quantizes in groups of 64 along the last weight axis. The control patch-embed
    # (in_features 132) is not a multiple of 64 and must be kept full precision instead of raising.
    class FakeWeight:
        def __init__(self, last_dim):
            self.shape = (3840, last_dim)

    class FakeLinear:
        def __init__(self, last_dim):
            self.weight = FakeWeight(last_dim)

        def to_quantized(self):  # presence is what the base predicate checks
            return self

    predicate = ZImageControlnetWeightDefinition.quantization_predicate
    assert predicate("control_all_x_embedder.2-1", FakeLinear(132)) is False
    assert predicate("control_layers.0.attention.to_q", FakeLinear(3840)) is True
