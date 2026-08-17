"""mflux-capabilities: a machine-readable dump of what each CLI actually honours.

Self-healing by construction, with no hand-maintained registry to rot:

- Commands are discovered from the installed console scripts (importlib.metadata), so a
  new CLI appears in the dump the moment it is wired into pyproject.
- Accepted options, defaults, types and help come from each CLI's live ``build_parser()``,
  so the dump can never list an option the parser no longer takes. The reported default is
  the PARSER default: several CLIs derive the effective value after parsing (sometimes per
  model), and the dump does not claim what it cannot introspect; declared runtime defaults
  are schema-v2 material.
- The honoured/ignored classification is read from the same module-level
  ``IGNORED_OPTIONS`` / ``CONDITIONAL_OPTIONS`` / ``REJECTED_OPTIONS`` constants the runtime
  warnings and errors use, so the
  dump and the warnings cannot disagree.
- A CLI that has not adopted the ``build_parser()`` convention is still listed, marked
  ``"coverage": "runtime-only"`` instead of being silently omitted.

Everything an option record claims is either introspected live or declared in the one
place the runtime also reads. Nothing here is transcribed by hand.
"""

import argparse
import importlib.metadata
import json
import sys
from typing import Any

SCHEMA_VERSION = 1

# Console-script prefixes that produce images and belong in the dump. Save/train/depth
# tooling can join in a later schema revision. The upscale commands are here because they
# produce images like any other: leaving them out is what let mflux-upscale-seedvr2 drop
# --metadata (#577) without the dump ever showing it.
COMMAND_PREFIXES = ("mflux-generate", "mflux-concept", "mflux-upscale")


def discover_commands() -> list[tuple[str, str]]:
    """(command, module) pairs from the installed entry points, sorted by command."""
    entry_points = importlib.metadata.entry_points()
    scripts = entry_points.select(group="console_scripts")
    found = [
        (entry_point.name, entry_point.value.split(":")[0])
        for entry_point in scripts
        if entry_point.name.startswith(COMMAND_PREFIXES)
    ]
    return sorted(set(found))


def _option_type_name(action: argparse.Action) -> str:
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
        return "bool"
    if isinstance(action, argparse.BooleanOptionalAction):
        return "bool"
    if action.type is None:
        return "str"
    name = getattr(action.type, "__name__", str(action.type))
    if name == "<lambda>":
        # An anonymous converter has no publishable name; the default's type is the
        # closest truthful description of what the option yields (e.g. int for the
        # clamped battery-percentage converter).
        return type(action.default).__name__ if action.default is not None else "str"
    return name


def _jsonable(value: Any) -> Any:
    """Parser defaults are arbitrary Python (Path, enums); the dump is a wire format.
    Normalize at record-build time so every output format serializes the same value."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        normalized = {str(k): _jsonable(v) for k, v in value.items()}
        if len(normalized) != len(value):
            # {1: "a", "1": "b"} would silently publish an incomplete record.
            raise ValueError(f"parser default mapping loses keys when stringified: {sorted(map(str, value))}")
        return normalized
    return str(value)


def _describe_option(action: argparse.Action, ignored: dict, conditional: dict, rejected: dict) -> dict[str, Any]:
    if isinstance(action, argparse.BooleanOptionalAction):
        # argparse registers the positive form first; publishing the longest string would make
        # the negated --no-* flag canonical while parser_default stays True.
        flag = action.option_strings[0]
    else:
        flag = max(action.option_strings, key=len)
    record: dict[str, Any] = {
        "flag": flag,
        "type": _option_type_name(action),
        "parser_default": _jsonable(action.default) if action.default is not argparse.SUPPRESS else None,
        "status": "honored",
    }
    if action.option_strings and len(action.option_strings) > 1:
        record["aliases"] = [o for o in action.option_strings if o != flag]
    if action.choices is not None:
        record["choices"] = sorted(str(c) for c in action.choices)
    if action.required:
        record["required"] = True
    if action.nargs not in (None, 0):
        record["nargs"] = str(action.nargs)
    if action.help:
        record["help"] = action.help
    if flag in rejected:
        record["status"] = "rejected"
        record["reason"] = rejected[flag]
    elif flag in ignored:
        record["status"] = "ignored"
        record["reason"] = ignored[flag]
    elif flag in conditional:
        record["status"] = "conditional"
        record["condition"] = conditional[flag]["condition"]
        record["reason"] = conditional[flag]["reason"]
    return record


def _traits(parser, module) -> dict[str, Any]:
    traits: dict[str, Any] = {
        "lora": bool(getattr(parser, "supports_lora", False)),
        "img2img": bool(getattr(parser, "supports_image_to_image", False)),
        "controlnet": bool(getattr(parser, "supports_controlnet", False)),
        "outpaint": bool(getattr(parser, "supports_image_outpaint", False)),
        "metadata_config": bool(getattr(parser, "supports_metadata_config", False)),
    }
    default_model = parser.get_default("model")
    if default_model:
        traits["default_model"] = default_model
        try:
            from mflux.models.common.config.model_config import ModelConfig

            traits["supports_guidance"] = bool(ModelConfig.from_name(default_model).supports_guidance)
        except Exception:  # noqa: BLE001 -- a saved-path default model has no builtin config
            pass
    return traits


def describe_command(command: str, module_path: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"command": command, "module": module_path}
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001 -- a broken import must not hide the rest of the dump
        entry["coverage"] = "import-error"
        entry["error"] = str(exc)
        return entry

    build_parser = getattr(module, "build_parser", None)
    if build_parser is None:
        # Listed rather than dropped: coverage gaps must be visible in the dump itself.
        entry["coverage"] = "runtime-only"
        return entry

    parser = build_parser()
    ignored = getattr(module, "IGNORED_OPTIONS", {})
    conditional = getattr(module, "CONDITIONAL_OPTIONS", {})
    rejected = getattr(module, "REJECTED_OPTIONS", {})
    entry["coverage"] = "full"
    entry["description"] = parser.description
    records = {
        id(action): _describe_option(action, ignored, conditional, rejected)
        for action in parser._actions
        if action.option_strings and "-h" not in action.option_strings
    }
    # A required mutually-exclusive group is invisible on the per-option records: each member
    # looks optional, so a contract-built invocation can omit all of them and exit 2. Publish
    # the group so generators know exactly one member is required.
    for group_idx, group in enumerate(parser._mutually_exclusive_groups):
        for action in group._group_actions:
            record = records.get(id(action))
            if record is None:
                continue
            record["choice_group"] = group_idx
            if group.required:
                record["choice_required"] = True
    entry["options"] = list(records.values())
    entry["traits"] = _traits(parser, module)
    return entry


def build_capabilities() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mflux_version": importlib.metadata.version("mflux"),
        "commands": [describe_command(command, module) for command, module in discover_commands()],
    }


def _to_markdown(capabilities: dict[str, Any]) -> str:
    lines = [
        f"# mflux capabilities (schema {capabilities['schema_version']}, mflux {capabilities['mflux_version']})",
        "",
    ]
    for cmd in capabilities["commands"]:
        lines.append(f"## {cmd['command']}")
        if cmd.get("coverage") != "full":
            lines.append(f"_coverage: {cmd.get('coverage')}_")
            lines.append("")
            continue
        lines.append("")
        lines.append("| option | type | default | status | notes |")
        lines.append("|---|---|---|---|---|")
        for opt in cmd["options"]:
            notes = opt.get("reason", "")
            if opt["status"] == "conditional":
                notes = f"honoured when {opt['condition']}. {opt.get('reason', '')}".strip()
            default = "" if opt["parser_default"] is None else repr(opt["parser_default"])
            lines.append(f"| `{opt['flag']}` | {opt['type']} | {default} | {opt['status']} | {notes} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump which options each mflux CLI actually honours.")
    parser.add_argument(
        "--command", type=str, default=None, help="Only dump this command (e.g. mflux-generate-z-image)."
    )
    parser.add_argument("--format", type=str, default="json", choices=["json", "yaml", "markdown"])
    args = parser.parse_args()

    capabilities = build_capabilities()
    if args.command:
        capabilities["commands"] = [c for c in capabilities["commands"] if c["command"] == args.command]
        if not capabilities["commands"]:
            parser.error(f"unknown command {args.command!r}; run without --command to list all")

    if args.format == "json":
        # Serialize to a string BEFORE writing: json.dump streams, so a late
        # serialization error would leave a truncated document on stdout that a
        # redirecting consumer then parses as garbage.
        sys.stdout.write(json.dumps(capabilities, indent=2) + "\n")
    elif args.format == "yaml":
        import yaml

        yaml.safe_dump(capabilities, sys.stdout, sort_keys=False)
    else:
        sys.stdout.write(_to_markdown(capabilities))


if __name__ == "__main__":
    main()
