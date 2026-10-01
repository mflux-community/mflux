import pytest

from mflux.cli.completions.generator import CompletionGenerator


@pytest.mark.fast
def test_completion_generator_includes_fibo_edit_command():
    generator = CompletionGenerator()

    assert "mflux-generate-fibo-edit" in generator.commands

    parser = generator.create_parser_for_command("mflux-generate-fibo-edit")
    script = generator.generate_command_function("mflux-generate-fibo-edit", parser)

    assert "_mflux_generate_fibo_edit()" in script
    assert "--image-path" in script
    assert "--mask-path" in script
    assert "--prompt" in script


@pytest.mark.fast
def test_completion_generator_includes_krea2_command():
    generator = CompletionGenerator()

    assert "mflux-generate-krea2" in generator.commands

    parser = generator.create_parser_for_command("mflux-generate-krea2")
    script = generator.generate_command_function("mflux-generate-krea2", parser)

    assert "_mflux_generate_krea2()" in script
    assert "--prompt" in script
    assert "--scheduler" in script
    assert "--image-path" in script
    assert "--image-strength" in script


@pytest.mark.fast
def test_completion_generator_builds_a_parser_for_every_command():
    # Regression: mflux-upscale-controlnet added its lora arguments twice, so any
    # invocation of mflux-completions died with `argparse.ArgumentError: conflicting
    # option string: --lora-style` before writing a single completion. Building each
    # command's parser is exactly what every mflux-completions code path does first,
    # so a duplicate-argument conflict on any command fails here.
    generator = CompletionGenerator()

    for command in generator.commands:
        parser = generator.create_parser_for_command(command)
        assert parser is not None, command


@pytest.mark.fast
def test_completion_generator_includes_atomic_lora_and_image_flags():
    generator = CompletionGenerator()
    parser = generator.create_parser_for_command("mflux-generate")
    script = generator.generate_command_function("mflux-generate", parser)

    # The atomic --lora / --image flags must be discoverable via shell completion
    # alongside the retained legacy flags. Match the exact option-spec token
    # ("'--lora''[") so this can't be satisfied by --lora-paths / --image-path etc.
    assert "'--lora''[" in script
    assert "'--image''[" in script
    assert "'--lora-paths''[" in script
    assert "'--image-path''[" in script


@pytest.mark.fast
def test_completion_generator_tracks_installed_console_scripts():
    # The command list is read off the installed entry points, so it can no longer
    # drift behind pyproject the way the hand list did (it was six commands behind:
    # boogu, lens, both ernie-image, ideogram4 and capabilities).
    import importlib.metadata

    generator = CompletionGenerator()

    installed = {
        entry_point.name
        for entry_point in importlib.metadata.distribution("mflux").entry_points
        if entry_point.group == "console_scripts"
    }
    assert set(generator.commands) == installed
    for command in (
        "mflux-generate-boogu",
        "mflux-generate-lens",
        "mflux-generate-ernie-image",
        "mflux-generate-ernie-image-turbo",
        "mflux-generate-ideogram4",
        "mflux-capabilities",
    ):
        assert command in generator.commands


@pytest.mark.fast
def test_completion_generator_prefers_the_cli_own_parser():
    # A build_parser command completes what its CLI actually parses; drift between the
    # completion recipe and the real parser is impossible for these. Lens never had a
    # recipe in the old hand-maintained chain, so its flags only complete via this path.
    generator = CompletionGenerator()

    script = generator.generate_command_function(
        "mflux-generate-lens", generator.create_parser_for_command("mflux-generate-lens")
    )
    assert "_mflux_generate_lens()" in script
    assert "--prompt" in script


@pytest.mark.fast
def test_completion_generator_completes_base_model():
    # --base-model lost its completions when #592 dropped the (stale) argparse choices=
    # list; the flag must route to a helper fed from the resolver instead (#595).
    generator = CompletionGenerator()
    parser = generator.create_parser_for_command("mflux-generate")
    script = generator.generate_command_function("mflux-generate", parser)

    assert "_mflux_base_models" in script

    helpers = generator.generate_helper_functions()
    assert "_mflux_base_models()" in helpers
    # The canonical keys, and only them: every alias plus repo id would bury the
    # useful suggestions (base_model_names() is ~84 entries).
    helper_body = helpers.split("_mflux_base_models() {", 1)[1].split("}", 1)[0]
    from mflux.models.common.resolution.config_resolution import ConfigResolution

    for key in ConfigResolution.base_model_keys():
        assert f"'{key}[" in helper_body, key
    assert "black-forest-labs" not in helper_body


@pytest.mark.fast
def test_completion_generator_completes_every_spelling_of_a_renamed_flag():
    generator = CompletionGenerator()
    parser = generator.create_parser_for_command("mflux-generate")
    script = generator.generate_command_function("mflux-generate", parser)

    # The renamed flags (#703) keep their old names as aliases. Completion must offer all
    # spellings, which includes the three spellings of the sidecar reader.
    assert "{--config-from-conf,--config-from-metadata,-C}" in script
    assert "{--make-conf,--metadata}" in script
    assert "{--no-exif,--no-metadata}" in script
