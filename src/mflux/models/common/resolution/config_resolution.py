import logging
from typing import TYPE_CHECKING

from mflux.models.common.resolution.actions import ConfigAction, Rule

if TYPE_CHECKING:
    from mflux.models.common.config.model_config import ModelConfig

logger = logging.getLogger(__name__)


class ConfigResolution:
    RULES = frozenset(
        {
            Rule(priority=0, name="exact_match", check="is_exact_match", action=ConfigAction.EXACT_MATCH),
            Rule(priority=1, name="explicit_base", check="has_explicit_base", action=ConfigAction.EXPLICIT_BASE),
            Rule(priority=2, name="infer_substring", check="can_infer_substring", action=ConfigAction.INFER_SUBSTRING),
            Rule(priority=3, name="error", check="always", action=ConfigAction.ERROR),
        }
    )

    @staticmethod
    def resolve(model_name: str | None, base_model: str | None = None) -> "ModelConfig":
        from mflux.models.common.config.model_config import AVAILABLE_MODELS, ModelConfig
        from mflux.utils.exceptions import InvalidBaseModel, ModelConfigError

        # `--base-model schnell` with no custom checkpoint asks for the base model itself.
        # Without this, the explicit-base rule builds a config whose model_name is None,
        # and every initializer that reads model_config.model_name fails to resolve a
        # weights path, surfacing as a misleading per-component error far from the cause.
        # Promoting the alias makes from_name yield the same config shape whichever
        # keyword named the model.
        if model_name is None and base_model is not None:
            model_name = base_model

        # Naming neither leaves nothing to resolve. Without this the substring rule
        # lowercases None and the caller gets an AttributeError from two frames deep.
        if model_name is None:
            raise ModelConfigError("No model requested: pass a model_name, a base_model, or both.")

        base_models = sorted(
            [m for m in AVAILABLE_MODELS.values() if m.base_model is None],
            key=lambda x: x.priority,
        )

        ctx = {
            "model_name": model_name,
            "base_model": base_model,
            "base_models": base_models,
            "ModelConfig": ModelConfig,
            "InvalidBaseModel": InvalidBaseModel,
            "ModelConfigError": ModelConfigError,
        }

        for rule in sorted(ConfigResolution.RULES, key=lambda r: r.priority):
            if ConfigResolution._check(rule.check, ctx):
                logger.debug(f"Config resolution: '{model_name}' → rule '{rule.name}' ({rule.action.value})")
                return ConfigResolution._execute(rule.action, ctx)

        raise ValueError(f"No rule matched for model_name: {model_name}")

    @staticmethod
    def _check(check: str, ctx: dict) -> bool:
        if check == "is_exact_match":
            model_name = ctx["model_name"]
            for base in ctx["base_models"]:
                if model_name == base.model_name or model_name in base.aliases:
                    return True
            return False
        if check == "has_explicit_base":
            return ctx["base_model"] is not None
        if check == "can_infer_substring":
            model_name_lower = ctx["model_name"].lower()
            for base in ctx["base_models"]:
                for alias in base.aliases:
                    if alias and alias.lower() in model_name_lower:
                        return True
            return False
        if check == "always":
            return True
        return False

    @staticmethod
    def _execute(action: ConfigAction, ctx: dict) -> "ModelConfig":
        model_name = ctx["model_name"]
        base_model = ctx["base_model"]
        base_models = ctx["base_models"]
        InvalidBaseModel = ctx["InvalidBaseModel"]
        ModelConfigError = ctx["ModelConfigError"]

        if action == ConfigAction.EXACT_MATCH:
            for base in base_models:
                if model_name == base.model_name or model_name in base.aliases:
                    return base
            raise ValueError("Exact match check passed but no match found")

        if action == ConfigAction.EXPLICIT_BASE:
            allowed_names = []
            for base in base_models:
                allowed_names.extend(base.aliases + [base.model_name])
            if base_model not in allowed_names:
                raise InvalidBaseModel(f"Invalid base_model. Choose one of {allowed_names}")

            default_base = next(
                (b for b in base_models if base_model == b.model_name or base_model in b.aliases),
                None,
            )
            return ConfigResolution._create_config(model_name, default_base)

        if action == ConfigAction.INFER_SUBSTRING:
            model_name_lower = model_name.lower()
            matching_bases = [
                (b, alias) for b in base_models for alias in b.aliases if alias and alias.lower() in model_name_lower
            ]
            if not matching_bases:
                raise ModelConfigError(f"Cannot infer base_model from {model_name}")

            default_base = sorted(matching_bases, key=lambda x: (-len(x[1]), x[0].priority))[0][0]
            return ConfigResolution._create_config(model_name, default_base)

        if action == ConfigAction.ERROR:
            raise ModelConfigError(f"Cannot infer base_model from {model_name}")

        raise ValueError(f"Unknown action: {action}")

    @staticmethod
    def _create_config(model_name: str, base: "ModelConfig") -> "ModelConfig":
        from mflux.models.common.config.model_config import ModelConfig

        # Carry every field the base declares and rewrite only identity. Enumerating
        # fields here is what silently dropped the sigma schedule (and, for ERNIE and
        # klein-9b-kv, the LoRA guidance and KV-cache flag): each field added to
        # ModelConfig had to be remembered here too, and eventually one wasn't.
        # Mutable containers are copied so an inferred config cannot mutate its base.
        carried = {
            key: dict(value) if isinstance(value, dict) else list(value) if isinstance(value, list) else value
            for key, value in vars(base).items()
            if not key.startswith("_")
        }
        carried["model_name"] = model_name
        carried["base_model"] = base.model_name
        return ModelConfig(**carried)
