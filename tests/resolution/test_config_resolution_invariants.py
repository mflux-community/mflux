import pytest

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.resolution.config_resolution import ConfigResolution

# Resolution is allowed to rewrite identity and nothing else. Deriving the field
# list from the config itself rather than naming fields keeps this honest as
# ModelConfig grows -- a new field is covered the day it is added.
IDENTITY_FIELDS = frozenset({"model_name", "base_model"})


def _carried_fields(config) -> list[str]:
    return sorted(f for f in vars(config) if not f.startswith("_") and f not in IDENTITY_FIELDS)


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
            for field in _carried_fields(root):
                assert getattr(resolved, field) == getattr(root, field), (
                    f"inference on {probe!r} dropped {field}: {getattr(resolved, field)!r} != {getattr(root, field)!r}"
                )
