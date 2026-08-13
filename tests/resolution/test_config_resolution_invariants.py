import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.resolution.config_resolution import ConfigResolution

# Fields that describe how a model runs rather than where its weights live.
# Resolution may rewrite identity (model_name/base_model); it must never drop these.
CARRIED_FIELDS = (
    "num_train_steps",
    "max_sequence_length",
    "supports_guidance",
    "requires_sigma_shift",
    "sigma_base_shift",
    "sigma_max_shift",
    "sigma_base_seq_len",
    "sigma_max_seq_len",
    "sigma_shift_terminal",
    "transformer_overrides",
    "text_encoder_overrides",
)

ROOTS = sorted(
    {m.model_name: m for m in AVAILABLE_MODELS.values() if m.base_model is None}.values(),
    key=lambda m: m.model_name or "",
)

# Some aliases are claimed by more than one root (e.g. "dev"). Those resolve by
# priority and are a separate concern; this file is about configuration surviving
# resolution, so it only probes names that belong to exactly one root.
_alias_owners: dict[str, int] = {}
for _root in ROOTS:
    for _alias in _root.aliases:
        _alias_owners[_alias] = _alias_owners.get(_alias, 0) + 1


def _unambiguous_aliases(root):
    return [a for a in root.aliases if _alias_owners[a] == 1]


@pytest.mark.fast
@pytest.mark.parametrize("root", ROOTS, ids=lambda r: r.model_name)
def test_inference_preserves_run_configuration(root):
    # A local path, a quantized folder, or a newer repo revision resolves by substring
    # inference rather than exact match. Inference rewrites identity; everything that
    # describes how the model runs has to survive it.
    #
    # Regression guard for #574: moving a registered model_name silently moves every
    # name that used to match it exactly onto this path, and a lossy _create_config
    # then hands back generic defaults for the scheduler.
    for alias in _unambiguous_aliases(root):
        for probe in (f"/models/{alias}-q4", f"user/{alias}-bf16"):
            resolved = ConfigResolution.resolve(model_name=probe)
            for field in CARRIED_FIELDS:
                assert getattr(resolved, field) == getattr(root, field), (
                    f"inference on {probe!r} dropped {field}: {getattr(resolved, field)!r} != {getattr(root, field)!r}"
                )
