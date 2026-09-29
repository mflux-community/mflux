import copy
import logging
from functools import cache
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
        return ConfigResolution._resolve(model_name=model_name, base_model=base_model)[0]

    @staticmethod
    def resolve_key(model_name: str | None, base_model: str | None = None) -> str | None:
        # The canonical AVAILABLE_MODELS key of the entry resolve() built its answer from:
        # for a built-in name the config itself, for a custom checkpoint the base it was
        # derived from. Callers that need to know *which model* a name refers to (which
        # class owns the weights, which defaults apply) ask here instead of pattern-matching
        # the name themselves, which is how `mflux-save` came to file lens, klein and
        # seedvr2 checkpoints under Flux1.
        #
        # Matched by identity, not by name: several entries share a repo id (the FLUX.1-dev
        # ControlNets; z-image-turbo and its ControlNet), so a reverse lookup on model_name
        # picks whichever one sorts first rather than the one that was resolved. None only
        # if a caller has put a config outside the registry into the resolution path.
        from mflux.models.common.config.model_config import AVAILABLE_MODELS

        root = ConfigResolution._resolve(model_name=model_name, base_model=base_model)[1]
        return next((key for key, config in AVAILABLE_MODELS.items() if config is root), None)

    @staticmethod
    def resolve_restricted(
        model_name: str | None,
        registry_key: str,
        model_path: str | None = None,
        extra_keys: tuple[str, ...] = (),
        base_model: str | None = None,
    ) -> "ModelConfig":
        # Resolve --model for a CLI hard-wired to one registry model or a closed family of them: `registry_key` is the
        # entry an omitted --model runs; `extra_keys` are siblings this CLI can equally serve. A builtin name must alias
        # one of these — anything else errors instead of silently loading a foreign config. Custom checkpoints (model_path
        # set) load their weights from the path and take their geometry from the family: an explicit base_model selects
        # the entry (naming anything outside the family is an error); without it the checkpoint's name is searched for a
        # family alias, the way from_name inferred it before #650, so mlx-community/flux2-klein-9b-8bit runs as klein-9b
        # rather than dying inside attention on the default entry's shapes, and a turbo-named z-image checkpoint runs as
        # z-image-turbo. Only the family is searched, so an unrelated name (~/models/my-finetune) keeps the default entry
        # instead of being rejected. Compared by identity: entries can share a repo id.
        from mflux.models.common.config.model_config import AVAILABLE_MODELS
        from mflux.utils.exceptions import ModelConfigError

        allowed = [AVAILABLE_MODELS[registry_key]] + [AVAILABLE_MODELS[key] for key in extra_keys]
        expected = allowed[0]
        if model_name is None or model_path is not None:
            if base_model is not None:
                root = next(
                    (config for config in allowed if base_model == config.model_name or base_model in config.aliases),
                    None,
                )
                if root is None:
                    aliases = [alias for config in allowed for alias in config.aliases]
                    raise ModelConfigError(
                        f"'{base_model}' is not {expected.model_name}; this CLI only accepts the aliases {aliases}."
                    )
                return root
            if model_path is not None:
                inferred = ConfigResolution.infer_family_member(model_path, registry_key, extra_keys)
                if inferred is not None:
                    return inferred
            return expected
        resolved = ConfigResolution.resolve(model_name=model_name)
        if all(resolved is not config for config in allowed):
            aliases = [alias for config in allowed for alias in config.aliases]
            raise ModelConfigError(
                f"'{model_name}' is not {expected.model_name}; this CLI only accepts the aliases {aliases}."
            )
        return resolved

    @staticmethod
    def infer_family_member(
        model_path: str, registry_key: str, extra_keys: tuple[str, ...] = ()
    ) -> "ModelConfig | None":
        # The entry a custom checkpoint's name selects within a restricted CLI's family, or
        # None when the name carries no family alias. resolve_restricted uses it for the
        # geometry; the flux2 commands ask it again to learn whether the name, rather than
        # the fallback, chose the config (a name that spells the default entry must be
        # judged like the default entry, not like an unknown checkpoint).
        #
        # An exact repo id wins outright; otherwise only the checkpoint's BASENAME is
        # searched (#701): the basename is the only path segment that names the checkpoint,
        # and a parent directory that happens to name a sibling
        # (/Volumes/flux2-klein-9b-experiments/my-4b-finetune) must not reshape it.
        from mflux.models.common.config.model_config import AVAILABLE_MODELS

        allowed = [AVAILABLE_MODELS[registry_key]] + [AVAILABLE_MODELS[key] for key in extra_keys]
        exact = next((config for config in allowed if model_path == config.model_name), None)
        if exact is not None:
            return exact
        basename = model_path.rstrip("/").rsplit("/", 1)[-1]
        return ConfigResolution._infer_from_name(basename, allowed)

    @staticmethod
    def _infer_from_name(name: str, candidates: list["ModelConfig"]) -> "ModelConfig | None":
        # Shared by the INFER_SUBSTRING rule and resolve_restricted so the two cannot drift: the
        # longest matching alias wins (a "flux2-klein-base-9b" name is not claimed by "flux2-klein"),
        # then registry priority. None when no alias appears in the name.
        lowered = name.lower()
        matches = [
            (config, alias) for config in candidates for alias in config.aliases if alias and alias.lower() in lowered
        ]
        if not matches:
            return None
        return sorted(matches, key=lambda match: (-len(match[1]), match[0].priority))[0][0]

    @staticmethod
    def _resolve(model_name: str | None, base_model: str | None = None) -> tuple["ModelConfig", "ModelConfig"]:
        # Returns the resolved config and the registry entry it came from. The root is kept
        # separate because _create_config rewrites identity onto a copy, which leaves the
        # result indistinguishable from every other config sharing that base.
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

        # Several roots share a model_name with a ControlNet derivative (z-image-turbo and
        # its Union ControlNet; the FLUX.1-dev/schnell ControlNets), and the match rules
        # return the first hit. A derived variant is always addressed by its own key or
        # alias, so a bare repo id must land on the base entry — sorted ahead of the
        # ControlNets here, where priority alone put the Z-Image ControlNet (15) before
        # plain turbo (21) and made `Tongyi-MAI/Z-Image-Turbo` resolve to the ControlNet.
        base_models = sorted(
            [m for m in AVAILABLE_MODELS.values() if m.base_model is None],
            key=lambda x: (x.controlnet_model is not None, x.priority),
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
    @cache
    def base_model_names() -> tuple[str, ...]:
        # Every value the explicit-base rule accepts: each root config's aliases and its
        # repo id, in priority order. The CLI validates --base-model against this instead
        # of a hand-maintained argparse choices= list, which had drifted far enough to
        # reject names the resolver itself accepts (e.g. `--base-model qwen-image`).
        # Cached like the defaults.py helpers (AVAILABLE_MODELS is fixed after import),
        # and a tuple so nobody can mutate the cached value through a reference (#595).
        from mflux.models.common.config.model_config import AVAILABLE_MODELS

        names = []
        for config in sorted(AVAILABLE_MODELS.values(), key=lambda m: m.priority):
            if config.base_model is None:
                names.extend(config.aliases + [config.model_name])
        return tuple(names)

    @staticmethod
    @cache
    def base_model_keys() -> tuple[str, ...]:
        # The canonical key of each root config: the short, printable form of
        # base_model_names(), which is too long to put in an error message once every
        # alias and repo id is spelled out.
        from mflux.models.common.config.model_config import AVAILABLE_MODELS

        return tuple(key for key, config in AVAILABLE_MODELS.items() if config.base_model is None)

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
    def _execute(action: ConfigAction, ctx: dict) -> tuple["ModelConfig", "ModelConfig"]:
        model_name = ctx["model_name"]
        base_model = ctx["base_model"]
        base_models = ctx["base_models"]
        InvalidBaseModel = ctx["InvalidBaseModel"]
        ModelConfigError = ctx["ModelConfigError"]

        if action == ConfigAction.EXACT_MATCH:
            for base in base_models:
                if model_name == base.model_name or model_name in base.aliases:
                    return base, base
            raise ValueError("Exact match check passed but no match found")

        if action == ConfigAction.EXPLICIT_BASE:
            allowed_names = ConfigResolution.base_model_names()
            if base_model not in allowed_names:
                raise InvalidBaseModel(f"Invalid base_model. Choose one of {list(allowed_names)}")

            default_base = next(
                (b for b in base_models if base_model == b.model_name or base_model in b.aliases),
                None,
            )
            return ConfigResolution._create_config(model_name, default_base), default_base

        if action == ConfigAction.INFER_SUBSTRING:
            default_base = ConfigResolution._infer_from_name(model_name, base_models)
            if default_base is None:
                raise ModelConfigError(f"Cannot infer base_model from {model_name}")
            return ConfigResolution._create_config(model_name, default_base), default_base

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
        # Deep copy rather than a shallow one: AVAILABLE_MODELS is a process-wide
        # singleton, and overrides nest (ERNIE's transformer_overrides holds a
        # rope_axes_dim list), so a shallow copy leaves an inferred config able to
        # mutate the registry for every later resolution.
        carried = {key: copy.deepcopy(value) for key, value in vars(base).items() if not key.startswith("_")}
        carried["model_name"] = model_name
        carried["base_model"] = base.model_name
        return ModelConfig(**carried)
