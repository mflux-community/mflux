import mlx.nn as nn
import pytest

from mflux.models.flux.weights.flux_weight_definition import FluxReduxWeightDefinition
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip


class TinySiglipStandIn(nn.Module):
    # Mirrors SiglipVisionTransformer's attribute roots (embeddings/encoder/post_layernorm/
    # head) at tiny dimensions; fixed-size architectures get stand-ins per
    # TinyCheckpointRoundtrip's TinyVAEStandIn precedent. The Conv2d exercises the
    # definition's quantization_predicate exclusion, and the saved tree having no
    # "vision_model" root exercises WeightApplier's weight_subkey fallback on reload.
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Conv2d(3, 64, kernel_size=3)
        self.encoder = nn.Linear(1152, 1152)
        self.post_layernorm = nn.LayerNorm(1152)
        self.head = nn.Linear(1152, 128)


class TinyReduxEncoderStandIn(nn.Module):
    # Same attribute names as the real ReduxEncoder (redux_up/redux_down), small dims.
    def __init__(self) -> None:
        super().__init__()
        self.redux_up = nn.Linear(64, 128)
        self.redux_down = nn.Linear(128, 64)


class TestTinyFluxReduxModelSaving:
    @pytest.mark.fast
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path):
        # The redux components are the save side of issue #667's dev-redux gap: saved
        # under image_encoder/ and image_embedder/ (model_attr differs from the
        # component names), reloaded through WeightLoader's mflux-format detection.
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=FluxReduxWeightDefinition,
            make_components=TestTinyFluxReduxModelSaving._tiny_components,
            base_path=tmp_path / "flux_redux_tiny_q8",
            bits=8,
            # Force shard boundaries so index.json/weight_map multi-shard paths are
            # exercised — the size-based split never shards test-sized tensors.
            tensors_per_shard=2,
        )

    @staticmethod
    def _tiny_components():
        return {
            "siglip": TinySiglipStandIn(),
            "redux_encoder": TinyReduxEncoderStandIn(),
        }
