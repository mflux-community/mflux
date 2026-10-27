import json
import math
import re


class PeftLoRAMetadata:
    # diffusers and PEFT save a LoRA's config, lora_alpha included, as JSON in the safetensors metadata instead of as
    # per-layer .alpha tensors. A pipeline save prefixes each config key with its component ("transformer.lora_alpha");
    # a model save does not. Without it a layer's update would be applied at alpha = rank.
    METADATA_KEY = "lora_adapter_metadata"

    @staticmethod
    def configs(metadata: dict[str, str] | None) -> dict[str, dict]:
        # The LoRA configs in the file, by component prefix ("" for a model save); empty when there are none
        raw = (metadata or {}).get(PeftLoRAMetadata.METADATA_KEY)
        if not raw:
            return {}
        try:
            flat = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(flat, dict):
            return {}
        configs: dict[str, dict] = {}
        for key, value in flat.items():
            component, _, name = key.rpartition(".")
            configs.setdefault(component, {})[name] = value
        return {component: config for component, config in configs.items() if "lora_alpha" in config}

    @staticmethod
    def alpha(weight_key: str, configs: dict[str, dict], rank: int) -> float | None:
        # The .alpha value that gives this layer the scale PEFT gives it: lora_alpha / rank, or lora_alpha / sqrt(rank)
        # with rsLoRA, where alpha_pattern can set lora_alpha per module. None when no config covers the key
        found = PeftLoRAMetadata._config_for(weight_key, configs)
        if found is None:
            return None
        config, module = found
        alpha = float(config["lora_alpha"])
        patterns = config.get("alpha_pattern") or {}
        for pattern, value in patterns.items():
            if re.match(rf"(.*\.)?({pattern})$", module):
                alpha = float(value)
                break
        if config.get("use_rslora"):
            # The loader divides the .alpha value by the rank
            return alpha * rank / math.sqrt(rank)
        return alpha

    @staticmethod
    def _config_for(weight_key: str, configs: dict[str, dict]) -> tuple[dict, str] | None:
        # The config of the key's component and the key's module name inside it, as PEFT names it
        module = re.sub(r"\.lora_[AB](\.[^.]+)?\.weight$", "", weight_key)
        if module == weight_key:
            return None
        for component, config in configs.items():
            if component and module.startswith(f"{component}."):
                return config, module.removeprefix(f"{component}.")
        if "" in configs:
            return configs[""], module
        if len(configs) == 1:
            # Keys saved without their component prefix, read with the only config there is
            return next(iter(configs.values())), module
        return None
