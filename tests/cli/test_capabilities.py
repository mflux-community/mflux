import argparse
import importlib
import json
import sys

import pytest

from mflux.cli import capabilities


@pytest.fixture(scope="module")
def caps():
    return capabilities.build_capabilities()


@pytest.mark.fast
def test_every_generate_entrypoint_has_full_coverage(caps):
    # The dump discovers commands from the installed entry points, so this fails the
    # moment someone adds a generate CLI without the build_parser() convention: the
    # command shows up on its own (self-healing coverage) and this test names it.
    assert caps["commands"], "entry-point discovery found nothing"
    not_full = [(c["command"], c.get("coverage")) for c in caps["commands"] if c.get("coverage") != "full"]
    assert not_full == [], f"commands without full capability coverage: {not_full}"


@pytest.mark.fast
def test_declared_flags_exist_in_their_parsers(caps):
    # A declaration for a flag the parser no longer takes is a lie waiting to be
    # printed; catch it at the source.
    for command in caps["commands"]:
        module = importlib.import_module(command["module"])
        parser = module.build_parser()
        known_flags = {flag for action in parser._actions for flag in action.option_strings}
        declared = set(getattr(module, "IGNORED_OPTIONS", {})) | set(getattr(module, "CONDITIONAL_OPTIONS", {}))
        missing = declared - known_flags
        assert not missing, f"{command['command']} declares options its parser does not take: {sorted(missing)}"


@pytest.mark.fast
def test_option_records_are_well_formed(caps):
    for command in caps["commands"]:
        for option in command["options"]:
            assert option["flag"].startswith("--")
            assert option["status"] in ("honored", "ignored", "conditional", "rejected")
            if option["status"] == "ignored":
                assert option["reason"]
            if option["status"] == "conditional":
                assert option["condition"]
                assert option["reason"]


@pytest.mark.fast
def test_known_statuses_survive(caps):
    def status_of(command_name: str, flag: str) -> dict:
        command = next(c for c in caps["commands"] if c["command"] == command_name)
        return next(o for o in command["options"] if o["flag"] == flag)

    assert status_of("mflux-generate", "--negative-prompt")["status"] == "ignored"
    assert status_of("mflux-generate-ideogram4", "--steps")["status"] == "ignored"
    assert status_of("mflux-generate-z-image-turbo", "--guidance")["status"] == "ignored"
    assert status_of("mflux-generate-z-image", "--negative-prompt")["status"] == "conditional"
    assert status_of("mflux-generate-z-image", "--seed")["status"] == "honored"


@pytest.mark.fast
def test_json_output_is_valid(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mflux-capabilities", "--command", "mflux-generate-z-image-turbo"])
    capabilities.main()
    dumped = json.loads(capsys.readouterr().out)
    assert dumped["schema_version"] == capabilities.SCHEMA_VERSION
    assert len(dumped["commands"]) == 1
    assert dumped["commands"][0]["command"] == "mflux-generate-z-image-turbo"
    assert dumped["commands"][0]["coverage"] == "full"


@pytest.mark.fast
def test_full_dump_serializes_to_every_format(caps):
    # A parser default that is not JSON-native (a Path parser default)
    # truncated the streamed JSON mid-value; defaults are normalized at record-build
    # time so every wire format serializes the whole document.
    text = json.dumps(caps)
    assert text.endswith("}")
    for command in caps["commands"]:
        for option in command["options"]:
            default = option["parser_default"]
            assert default is None or isinstance(default, (str, int, float, bool, list, dict)), (
                f"{command['command']} {option['flag']} default is {type(default).__name__}"
            )


@pytest.mark.fast
def test_flux_family_negative_prompt_gaps_are_declared(caps):
    # The controlnet and depth variants never read negative_prompt (zero mentions in
    # their variant trees), same as the base FLUX.1 CLI that already declared it.
    for command_name in (
        "mflux-generate-controlnet",
        "mflux-generate-depth",
        "mflux-generate-fill",
        "mflux-generate-redux",
        "mflux-generate-kontext",
        "mflux-generate-in-context",
        "mflux-generate-in-context-catvton",
        "mflux-generate-in-context-edit",
    ):
        command = next(c for c in caps["commands"] if c["command"] == command_name)
        option = next(o for o in command["options"] if o["flag"] == "--negative-prompt")
        assert option["status"] == "ignored", command_name


@pytest.mark.fast
def test_flux_guidance_is_conditional_on_the_resolved_model(caps):
    # dev honours --guidance; schnell has supports_guidance=False and builds no
    # guidance embedder, so the same CLI gives two answers keyed on --base-model.
    command = next(c for c in caps["commands"] if c["command"] == "mflux-generate")
    option = next(o for o in command["options"] if o["flag"] == "--guidance")
    assert option["status"] == "conditional"
    assert "schnell" in option["condition"]


@pytest.mark.fast
def test_controlnet_guidance_is_conditional_on_the_resolved_model(caps):
    # --model schnell resolves to schnell_controlnet_canny (supports_guidance=False),
    # same shape as the base CLI: two answers keyed on the model flag.
    command = next(c for c in caps["commands"] if c["command"] == "mflux-generate-controlnet")
    option = next(o for o in command["options"] if o["flag"] == "--guidance")
    assert option["status"] == "conditional"
    assert "schnell" in option["condition"]
    assert option["reason"]


@pytest.mark.fast
def test_the_upscale_commands_are_in_the_dump(caps):
    # They produce images like any other command, but COMMAND_PREFIXES left both out —
    # which is how mflux-upscale-seedvr2 dropped --metadata (#577) with the dump silent.
    commands = {c["command"]: c for c in caps["commands"]}
    for command_name in ("mflux-upscale-controlnet", "mflux-upscale-seedvr2"):
        assert command_name in commands, sorted(commands)
        assert commands[command_name]["coverage"] == "full"
    seedvr2 = commands["mflux-upscale-seedvr2"]
    assert next(o for o in seedvr2["options"] if o["flag"] == "--metadata")["status"] == "honored"


@pytest.mark.fast
def test_the_upscale_commands_declare_what_they_cannot_honour(caps):
    # Undeclared reads as honored, so publishing these two required saying which flags
    # their hardcoded configs ignore — otherwise widening the prefixes just prints more
    # of the false contract the dump exists to catch.
    def status_of(command_name: str, flag: str) -> dict:
        command = next(c for c in caps["commands"] if c["command"] == command_name)
        return next(o for o in command["options"] if o["flag"] == flag)

    assert status_of("mflux-upscale-controlnet", "--negative-prompt")["status"] == "ignored"
    assert status_of("mflux-upscale-controlnet", "--base-model")["status"] == "ignored"
    assert status_of("mflux-upscale-seedvr2", "--base-model")["status"] == "ignored"
    # --model is the half-honored one: a path loads weights, a built-in name does nothing.
    model = status_of("mflux-upscale-controlnet", "--model")
    assert model["status"] == "conditional"
    assert model["condition"] and model["reason"]
    # SeedVR2 resolves --model itself, so it stays honored there.
    assert status_of("mflux-upscale-seedvr2", "--model")["status"] == "honored"


@pytest.mark.fast
def test_capabilities_lambda_converter_publishes_default_type():
    caps = capabilities.build_capabilities()
    for command in caps["commands"]:
        for option in command["options"]:
            assert option["type"] != "<lambda>", f"{command['command']} {option['flag']} leaks <lambda>"


@pytest.mark.fast
def test_jsonable_preserves_mappings():
    # A dict default must stay a JSON object, not become a Python repr string.
    from pathlib import Path

    assert capabilities._jsonable({"width": 1024, "p": Path("x")}) == {"width": 1024, "p": "x"}
    assert capabilities._jsonable({1: {"p": Path("x")}}) == {"1": {"p": "x"}}
    assert capabilities._jsonable([Path("a"), 2]) == ["a", 2]
    with pytest.raises(ValueError, match="loses keys"):
        capabilities._jsonable({1: "a", "1": "b"})


@pytest.mark.fast
def test_boolean_optional_flags_publish_the_positive_form():
    caps = capabilities.build_capabilities()
    for command in caps["commands"]:
        for option in command["options"]:
            spellings = [option["flag"], *(option.get("aliases") or [])]
            negatives = [flag for flag in spellings if flag.startswith("--no-")]
            paired = [flag for flag in negatives if "--" + flag[len("--no-") :] in spellings]
            if not paired:
                # A lone --no-* flag is its own option rather than the negated half of a
                # pair, and renaming it is not this contract's business.
                continue
            # Both spellings drive one option, so the canonical one has to be the positive
            # form whatever the default is: a caller builds invocations from these strings.
            assert not option["flag"].startswith("--no-"), (
                f"{command['command']} publishes {option['flag']} as canonical over "
                f"{'--' + option['flag'][len('--no-') :]}"
            )


@pytest.mark.fast
def test_a_boolean_optional_action_publishes_its_positive_spelling():
    # The command sweep above only sees flags some CLI happens to declare. This pins the
    # rule at the source, since argparse stores the negated spelling as a second option
    # string and picking the longest one would make --no-bake-lora canonical.
    parser = argparse.ArgumentParser()
    action = parser.add_argument("--bake-lora", action=argparse.BooleanOptionalAction, default=True)

    record = capabilities._describe_option(action, {}, {}, {})

    assert record["flag"] == "--bake-lora"
    assert record["aliases"] == ["--no-bake-lora"]
    assert record["parser_default"] is True


@pytest.mark.fast
def test_required_prompt_group_is_published_as_a_required_choice():
    caps = capabilities.build_capabilities()
    # mflux-generate-controlnet requires exactly one of --prompt/--prompt-file; a contract
    # that publishes both as plain optional flags builds invocations that exit 2.
    command = next(c for c in caps["commands"] if c["command"] == "mflux-generate-controlnet")
    by_flag = {o["flag"]: o for o in command["options"]}
    prompt, prompt_file = by_flag["--prompt"], by_flag["--prompt-file"]
    assert prompt.get("choice_required") is True
    assert prompt_file.get("choice_required") is True
    assert prompt["choice_group"] == prompt_file["choice_group"]


@pytest.mark.fast
def test_flux2_negative_prompt_is_rejected_not_honored():
    caps = capabilities.build_capabilities()
    for name in ("mflux-generate-flux2", "mflux-generate-flux2-edit"):
        command = next(c for c in caps["commands"] if c["command"] == name)
        option = next(o for o in command["options"] if o["flag"] == "--negative-prompt")
        assert option["status"] == "rejected", f"{name} publishes --negative-prompt as {option['status']}"


@pytest.mark.fast
def test_dropped_negative_prompts_are_not_honored():
    caps = capabilities.build_capabilities()
    for name, expected in (("mflux-generate-krea2", "conditional"), ("mflux-generate-ernie-image-turbo", "ignored")):
        command = next(c for c in caps["commands"] if c["command"] == name)
        option = next(o for o in command["options"] if o["flag"] == "--negative-prompt")
        assert option["status"] == expected, f"{name} publishes --negative-prompt as {option['status']}"
