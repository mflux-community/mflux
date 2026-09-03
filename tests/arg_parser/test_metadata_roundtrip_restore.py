# A metadata sidecar records the size a run generated at, the negative prompt it used and
# the init images an edit CLI was pointed at, but --config-from-metadata never read any of
# them back: a sidecar-only rerun regenerated at the wrong size with no negative prompt,
# and on the edit CLIs argparse rejected it outright, before the restore block ever ran.

import json
import sys

import pytest

from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.flux.cli import flux_generate_redux
from mflux.models.flux2.cli import flux2_edit_generate, flux2_generate
from mflux.models.krea2.cli import krea2_generate
from mflux.models.qwen.cli import qwen_image_edit_generate
from mflux.models.z_image.cli import z_image_generate
from mflux.utils.exceptions import ModelConfigError

ELSEWHERE = "/Users/someone-else/photos/cat.png"

SCALE_FACTOR_CLIS = [krea2_generate, qwen_image_edit_generate, flux2_edit_generate]
EDIT_CLIS = [qwen_image_edit_generate, flux2_edit_generate]


@pytest.fixture
def source_image(tmp_path):
    def make(name: str) -> str:
        path = tmp_path / name
        path.touch()
        return str(path)

    return make


@pytest.fixture
def sidecar(tmp_path, source_image):
    def write(**overrides):
        metadata = {
            "model": "krea-2",
            "prompt": "a red cube",
            "seed": 7,
            "steps": 8,
            "height": 1536,
            "width": 768,
            "negative_prompt": "blurry, watermark",
            "image_paths": [source_image("first.png"), source_image("second.png")],
        }
        metadata.update(overrides)
        path = tmp_path / f"prior-{len(list(tmp_path.glob('prior-*.json')))}.metadata.json"
        path.write_text(json.dumps(metadata))
        return path

    return write


@pytest.fixture
def parse(monkeypatch):
    def run(cli, *argv: str):
        monkeypatch.setattr(sys, "argv", ["prog", *argv])
        return cli.build_parser().parse_args()

    return run


@pytest.mark.fast
@pytest.mark.parametrize("cli", SCALE_FACTOR_CLIS, ids=lambda cli: cli.__name__.rsplit(".", 1)[-1])
def test_the_sidecar_supplies_the_size_it_generated_at(parse, sidecar, cli):
    # These CLIs default to "auto", which without an init image falls through to 1024x1024.
    args = parse(cli, "--config-from-metadata", str(sidecar()))
    assert (args.width, args.height) == (768, 1536)


@pytest.mark.fast
def test_the_sidecar_supplies_the_size_on_a_fixed_dimension_cli(parse, sidecar):
    # z-image has no scale factors: its default is the literal 1024 that reruns came out at.
    args = parse(z_image_generate, "--config-from-metadata", str(sidecar()))
    assert (args.width, args.height) == (768, 1536)


@pytest.mark.fast
@pytest.mark.parametrize("cli", [krea2_generate, z_image_generate], ids=["krea2", "z-image"])
def test_the_sidecar_supplies_the_negative_prompt(parse, sidecar, cli):
    args = parse(cli, "--config-from-metadata", str(sidecar()))
    assert args.negative_prompt == "blurry, watermark"


@pytest.mark.fast
def test_the_command_line_overrides_the_sidecar_one_dimension_at_a_time(parse, sidecar):
    args = parse(krea2_generate, "--config-from-metadata", str(sidecar()), "--height", "512")
    assert (args.width, args.height) == (768, 512)


@pytest.mark.fast
def test_a_negative_prompt_on_the_command_line_replaces_the_sidecars(parse, sidecar):
    args = parse(krea2_generate, "--config-from-metadata", str(sidecar()), "--negative-prompt", "mine")
    assert args.negative_prompt == "mine"


@pytest.mark.fast
def test_a_sidecar_without_the_keys_leaves_the_defaults_alone(parse, sidecar):
    path = sidecar(height=None, width=None, negative_prompt=None)
    args = parse(z_image_generate, "--config-from-metadata", str(path))
    assert (args.width, args.height) == (1024, 1024)
    assert args.negative_prompt == ""


@pytest.mark.fast
@pytest.mark.parametrize("cli", EDIT_CLIS, ids=["qwen-edit", "flux2-edit"])
def test_the_sidecar_supplies_the_init_images(parse, sidecar, cli):
    # argparse checks required= before parse_args reaches the metadata block, so this used
    # to exit 2 on images the sidecar was carrying all along.
    path = sidecar()
    expected = json.loads(path.read_text())["image_paths"]
    args = parse(cli, "--config-from-metadata", str(path))
    assert [str(image) for image in args.image_paths] == expected


@pytest.mark.fast
@pytest.mark.parametrize("cli", EDIT_CLIS, ids=["qwen-edit", "flux2-edit"])
def test_init_images_on_the_command_line_replace_the_sidecars(parse, sidecar, source_image, cli):
    mine = source_image("mine.png")
    args = parse(cli, "--config-from-metadata", str(sidecar()), "--image-paths", mine)
    assert [str(image) for image in args.image_paths] == [mine]


@pytest.mark.fast
@pytest.mark.parametrize("cli", EDIT_CLIS, ids=["qwen-edit", "flux2-edit"])
def test_an_edit_cli_still_requires_init_images_from_somewhere(parse, capsys, cli):
    with pytest.raises(SystemExit) as exit_info:
        parse(cli, "--prompt", "a red cube")
    assert exit_info.value.code == 2
    assert "--image-paths" in capsys.readouterr().err


@pytest.mark.fast
def test_a_sidecar_init_image_that_is_missing_here_names_the_sidecar(parse, sidecar, capsys):
    # Sidecars record the absolute path the generating machine resolved. Without this the
    # rerun loaded the whole model first and then died on a bare FileNotFoundError, since
    # nothing opens the file until dimension resolution.
    path = sidecar(image_paths=[ELSEWHERE])
    with pytest.raises(SystemExit) as exit_info:
        parse(qwen_image_edit_generate, "--config-from-metadata", str(path))
    assert exit_info.value.code == 2
    stderr = capsys.readouterr().err
    assert ELSEWHERE in stderr
    assert str(path) in stderr


@pytest.mark.fast
def test_flux2_rejects_a_negative_prompt_that_was_actually_typed(monkeypatch, capsys, source_image):
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "a red cube", "--image-paths", source_image("a.png"), "--negative-prompt", "blurry"])  # fmt: off
    with pytest.raises(SystemExit) as exit_info:
        flux2_edit_generate.main()
    assert exit_info.value.code == 2
    assert "not supported for FLUX.2" in capsys.readouterr().err


@pytest.mark.fast
def test_flux2_does_not_blame_the_user_for_a_sidecars_negative_prompt(monkeypatch, sidecar):
    # FLUX.2 has no negative branch, but the sidecar of a CFG model carries one and the
    # rerun must not be rejected for an argument that is nowhere on the command line.
    reached_model_load = RuntimeError("reached model load")

    def explode(*args, **kwargs):
        raise reached_model_load

    monkeypatch.setattr(flux2_edit_generate, "Flux2KleinEdit", explode)
    monkeypatch.setattr(sys, "argv", ["prog", "--config-from-metadata", str(sidecar(model="flux2-klein-4b"))])
    with pytest.raises(RuntimeError) as exit_info:
        flux2_edit_generate.main()
    assert exit_info.value is reached_model_load


@pytest.mark.fast
def test_the_sidecar_supplies_redux_images(parse, sidecar, source_image):
    # argparse required= on --redux-image-paths used to reject a -C-only restore
    # before the metadata block ran, the same failure the edit CLIs had.
    refs = [source_image("style-a.png"), source_image("style-b.png")]
    path = sidecar(redux_image_paths=refs, redux_image_strengths=[0.8, 0.5])
    args = parse(flux_generate_redux, "--config-from-metadata", str(path))
    assert [str(image) for image in args.redux_image_paths] == refs
    assert args.redux_image_strengths == [0.8, 0.5]


@pytest.mark.fast
def test_redux_images_on_the_command_line_replace_the_sidecars(parse, sidecar, source_image):
    mine = source_image("mine.png")
    path = sidecar(redux_image_paths=[source_image("old.png")], redux_image_strengths=[0.2])
    args = parse(
        flux_generate_redux,
        "--config-from-metadata",
        str(path),
        "--redux-image-paths",
        mine,
        "--redux-image-strengths",
        "0.9",
    )
    assert [str(image) for image in args.redux_image_paths] == [mine]
    assert args.redux_image_strengths == [0.9]


@pytest.mark.fast
def test_redux_still_requires_images_from_somewhere(parse, capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse(flux_generate_redux, "--prompt", "a red cube")
    assert exit_info.value.code == 2
    assert "--redux-image-paths" in capsys.readouterr().err


@pytest.mark.fast
def test_a_sidecar_redux_image_that_is_missing_here_names_the_sidecar(parse, sidecar, capsys):
    path = sidecar(redux_image_paths=[ELSEWHERE])
    with pytest.raises(SystemExit) as exit_info:
        parse(flux_generate_redux, "--config-from-metadata", str(path))
    assert exit_info.value.code == 2
    stderr = capsys.readouterr().err
    assert ELSEWHERE in stderr
    assert str(path) in stderr


@pytest.mark.fast
def test_an_old_repr_redux_sidecar_is_not_treated_as_paths(parse, sidecar, capsys):
    # Pre-#649 sidecars wrote str(list). Iterating that string would invent one
    # "path" per character; reject it and ask for --redux-image-paths instead.
    path = sidecar(redux_image_paths="[PosixPath('/tmp/a.png')]")
    with pytest.raises(SystemExit) as exit_info:
        parse(flux_generate_redux, "--config-from-metadata", str(path))
    assert exit_info.value.code == 2
    assert "--redux-image-paths" in capsys.readouterr().err


# A local-checkpoint run stores the registry entry it resolved to in "model" and the
# actual weights source in "model_path". Restoring the entry alone replays against the
# registry weights with no error, which is #705: the source is what the command line
# actually said, so it wins. Pre-#705 sidecars carry no model_path and restore as before.
@pytest.mark.fast
def test_the_sidecar_replays_the_weights_source_it_ran_from(parse, sidecar):
    prior = sidecar(
        model="black-forest-labs/FLUX.2-klein-4B",
        model_path="/models/klein-4b-q4",
        image_paths=None,
    )
    args = parse(flux2_generate, "--config-from-metadata", str(prior))
    assert args.model == "/models/klein-4b-q4"
    assert args.model_path == "/models/klein-4b-q4"


@pytest.mark.fast
def test_an_explicit_model_beats_the_sidecar_weights_source(parse, sidecar):
    prior = sidecar(
        model="black-forest-labs/FLUX.2-klein-4B",
        model_path="/models/klein-4b-q4",
        image_paths=None,
    )
    args = parse(flux2_generate, "--model", "flux2-klein-4b", "--config-from-metadata", str(prior))
    assert args.model == "flux2-klein-4b"
    assert args.model_path is None


@pytest.mark.fast
def test_a_sidecar_without_a_weights_source_restores_the_entry(parse, sidecar):
    prior = sidecar()  # the fixture's krea-2 sidecar, as written before #705
    args = parse(krea2_generate, "--config-from-metadata", str(prior))
    assert args.model == "krea-2"
    assert args.model_path is None


@pytest.mark.fast
def test_the_recorded_entry_rides_along_as_the_base(parse, sidecar):
    # The weights source alone erases the run's family. The entry it resolved to comes
    # back as the explicit base, so the right CLI pins the exact entry without
    # re-inferring it from the basename (#708 review).
    prior = sidecar(model="black-forest-labs/FLUX.2-klein-4B", model_path="/models/my-finetune", image_paths=None)
    args = parse(flux2_generate, "--config-from-metadata", str(prior))
    assert args.base_model == "black-forest-labs/FLUX.2-klein-4B"
    resolved = ConfigResolution.resolve_restricted(
        args.model,
        "flux2-klein-4b",
        model_path=args.model_path,
        base_model=args.base_model,
        extra_keys=("flux2-klein-9b", "flux2-klein-base-4b", "flux2-klein-base-9b", "flux2-klein-9b-kv"),
    )
    assert resolved.model_name == "black-forest-labs/FLUX.2-klein-4B"


@pytest.mark.fast
def test_a_cross_family_replay_is_still_rejected(parse, sidecar):
    # Replaying a flux2 checkpoint sidecar through the z-image command used to die inside
    # weight loading on the default entry's geometry once the source was preferred; the
    # ride-along base restores the clean rejection the entry produced by name.
    prior = sidecar(model="black-forest-labs/FLUX.2-klein-4B", model_path="/models/klein-4b-q4", image_paths=None)
    args = parse(z_image_generate, "--config-from-metadata", str(prior))
    with pytest.raises(ModelConfigError):
        ConfigResolution.resolve_restricted(
            args.model,
            "z-image",
            model_path=args.model_path,
            base_model=args.base_model,
            extra_keys=("z-image-turbo",),
        )


@pytest.mark.fast
def test_an_explicit_base_model_beats_the_ride_along(parse, sidecar):
    prior = sidecar(model="black-forest-labs/FLUX.2-klein-4B", model_path="/models/my-finetune", image_paths=None)
    args = parse(flux2_generate, "--base-model", "flux2-klein-base-4b", "--config-from-metadata", str(prior))
    assert args.base_model == "flux2-klein-base-4b"
