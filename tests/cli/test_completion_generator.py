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
