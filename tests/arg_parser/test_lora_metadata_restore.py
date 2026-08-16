# A metadata sidecar records the LoRA paths and scales of the run that produced it, and
# --config-from-metadata restores them. That restore used to concatenate instead of deferring
# to the command line: the sidecar's adapters always applied, so no spelling could replace
# them, and when only one of the two lists came from the CLI the counts disagreed and
# LoraResolution.resolve_scales padded or truncated the difference away.

import json
import sys

import pytest

from mflux.models.flux.cli import flux_generate

ELSEWHERE = "/Users/someone-else/loras/anime.safetensors"


@pytest.fixture
def sidecar(tmp_path):
    def write(lora_paths: list[str], lora_scales: list[float]):
        path = tmp_path / "prior.metadata.json"
        path.write_text(
            json.dumps(
                {
                    "model": "schnell",
                    "prompt": "a red cube",
                    "seed": 7,
                    "steps": 2,
                    "lora_paths": lora_paths,
                    "lora_scales": lora_scales,
                }
            )
        )
        return path

    return write


@pytest.fixture
def adapter(tmp_path):
    def make(name: str) -> str:
        path = tmp_path / name
        path.touch()
        return str(path)

    return make


@pytest.fixture
def parse(monkeypatch):
    def run(*argv: str):
        monkeypatch.setattr(sys, "argv", ["prog", *argv])
        return flux_generate.build_parser().parse_args()

    return run


@pytest.mark.fast
def test_the_sidecar_supplies_the_loras_when_the_cli_names_none(parse, sidecar, adapter):
    meta = adapter("meta.safetensors")
    args = parse("--config-from-metadata", str(sidecar([meta], [0.3])))
    assert args.lora_paths == [meta]
    assert args.lora_scales == [pytest.approx(0.3)]


@pytest.mark.fast
@pytest.mark.parametrize(
    "spelling",
    [["--lora", "CLI", "0.9"], ["--lora-paths", "CLI", "--lora-scales", "0.9"]],
    ids=["atomic", "legacy"],
)
def test_a_lora_on_the_command_line_replaces_the_sidecars(parse, sidecar, adapter, spelling):
    cli, meta = adapter("cli.safetensors"), adapter("meta.safetensors")
    argv = [cli if token == "CLI" else token for token in spelling]
    args = parse(*argv, "--config-from-metadata", str(sidecar([meta], [0.3])))
    assert args.lora_paths == [cli]
    assert args.lora_scales == [pytest.approx(0.9)]


@pytest.mark.fast
def test_cli_paths_take_the_sidecars_scales_with_them(parse, sidecar, adapter):
    # Keeping the sidecar's scales here is what left two paths facing one scale, which
    # resolve_scales pads: the CLI's adapter ran at 1.0 behind a warning.
    cli, meta = adapter("cli.safetensors"), adapter("meta.safetensors")
    args = parse("--lora-paths", cli, "--config-from-metadata", str(sidecar([meta], [0.3])))
    assert args.lora_paths == [cli]
    assert args.lora_scales is None


@pytest.mark.fast
def test_scales_alone_rerun_the_sidecars_loras_at_the_new_strength(parse, sidecar, adapter):
    # The other half of the same misalignment: one path met two scales, and resolve_scales
    # truncated, so the requested strength was dropped for the sidecar's.
    meta = adapter("meta.safetensors")
    args = parse("--lora-scales", "0.8", "--config-from-metadata", str(sidecar([meta], [0.3])))
    assert args.lora_paths == [meta]
    assert args.lora_scales == [pytest.approx(0.8)]


@pytest.mark.fast
def test_lora_paths_with_no_values_clears_the_sidecars_loras(parse, sidecar, adapter):
    meta = adapter("meta.safetensors")
    args = parse("--config-from-metadata", str(sidecar([meta], [0.3])), "--lora-paths")
    assert args.lora_paths == []


@pytest.mark.fast
def test_a_sidecar_lora_that_is_missing_here_names_the_sidecar_and_the_way_out(parse, sidecar, capsys):
    # Sidecars record the absolute path the generating machine resolved, so one carried
    # between machines fails on an argument that is nowhere on the command line.
    path = sidecar([ELSEWHERE], [0.3])
    with pytest.raises(SystemExit) as exit_info:
        parse("--config-from-metadata", str(path))
    assert exit_info.value.code == 2
    stderr = capsys.readouterr().err
    assert ELSEWHERE in stderr
    assert str(path) in stderr
    assert "--lora-paths with no values" in stderr


@pytest.mark.fast
def test_your_own_copy_stands_in_for_a_sidecar_lora_that_is_missing_here(parse, sidecar, adapter):
    # Merging meant the sidecar's dead path was resolved even when the user had named a
    # replacement, so substituting your own copy of the adapter was not possible at all.
    cli = adapter("mine.safetensors")
    args = parse("--lora", cli, "0.9", "--config-from-metadata", str(sidecar([ELSEWHERE], [0.3])))
    assert args.lora_paths == [cli]


@pytest.mark.fast
def test_a_bad_path_typed_on_the_command_line_is_not_blamed_on_the_sidecar(parse, sidecar, adapter, capsys):
    with pytest.raises(SystemExit):
        parse("--lora", ELSEWHERE, "--config-from-metadata", str(sidecar([adapter("meta.safetensors")], [0.3])))
    assert "came from" not in capsys.readouterr().err
