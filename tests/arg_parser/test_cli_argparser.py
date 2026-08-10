import json
import random
from pathlib import Path
from unittest.mock import patch

import pytest

from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.fibo.cli.fibo_edit import _resolve_fibo_edit_model_config
from mflux.utils.box_values import BoxValues
from mflux.utils.scale_factor import ScaleFactor


def _create_mflux_generate_parser(with_controlnet=False, require_model_arg=False) -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image based on a prompt.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=require_model_arg)
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_lora_arguments()
    parser.add_image_to_image_arguments(required=False)
    parser.add_image_outpaint_arguments()
    if with_controlnet:
        parser.add_controlnet_arguments(mode="canny")
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_generate_parser() -> CommandLineParser:
    return _create_mflux_generate_parser(with_controlnet=False, require_model_arg=False)


@pytest.fixture
def mflux_generate_controlnet_parser() -> CommandLineParser:
    return _create_mflux_generate_parser(with_controlnet=True, require_model_arg=False)


@pytest.fixture
def mflux_save_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Save a quantized version of Flux.1 to disk.")  # fmt: off
    parser.add_general_arguments()
    parser.add_model_arguments(path_type="save", require_model_arg=True)
    parser.add_lora_arguments()
    return parser


@pytest.fixture
def mflux_generate_minimal_argv() -> list[str]:
    return ["mflux-generate", "--prompt", "meaning of life"]


@pytest.fixture
def mflux_generate_minimal_model_argv() -> list[str]:
    return ["mflux-generate", "--prompt", "meaning of life", "--model", "dev"]


@pytest.fixture
def mflux_generate_controlnet_minimal_argv() -> list[str]:
    return ["mflux-generate-controlnet", "--prompt", "meaning of life, imitated"]


@pytest.fixture
def temp_dir(tmp_path_factory) -> Path:
    # Create a temporary directory for the module
    temp_dir = tmp_path_factory.mktemp("mflux_cli_argparser_tests")
    return Path(temp_dir)


@pytest.fixture
def base_metadata_dict() -> dict:
    return {
        "mflux_version": "0.4.0",
        "model": "dev",
        "seed": 42042,
        "steps": 14,
        "guidance": None,
        "precision": "mlx.core.bfloat16",
        "quantize": None,
        "generation_time_seconds": 42.0,
        "lora_paths": None,
        "lora_scales": None,
        "image": None,
        "image_strength": None,
        "controlnet_image": None,
        "controlnet_strength": None,
        "controlnet_save_canny": False,
    }


@pytest.fixture
def mflux_fill_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using the fill tool to complete masked areas.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_fill_arguments()
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_fill_minimal_argv() -> list[str]:
    return [
        "mflux-generate-fill",
        "--prompt",
        "meaning of life",
        "--image-path",
        "image.png",
        "--masked-image-path",
        "mask.png",
    ]


@pytest.fixture
def mflux_save_depth_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Save depth map from an image.")
    parser.add_general_arguments()
    parser.add_save_depth_arguments()
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_save_depth_minimal_argv() -> list[str]:
    return ["mflux-save-depth", "--image-path", "image.png"]


@pytest.fixture
def mflux_redux_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate redux images.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_redux_arguments()
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_redux_minimal_argv() -> list[str]:
    return ["mflux-generate-redux", "--redux-image-paths", "image1.png", "image2.png"]


@pytest.fixture
def mflux_concept_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image with concept attention based on a prompt and concept.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    parser.add_concept_attention_arguments()
    return parser


@pytest.fixture
def mflux_concept_minimal_argv() -> list[str]:
    return [
        "mflux-concept",
        "--prompt",
        "a beautiful landscape with a car",
        "--concept",
        "car",
    ]


@pytest.fixture
def mflux_catvton_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate virtual try-on images using in-context learning.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False, require_prompt=False)
    parser.add_catvton_arguments()
    parser.add_in_context_arguments()
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_catvton_minimal_argv() -> list[str]:
    return [
        "mflux-generate-in-context-catvton",
        "--person-image",
        "person.png",
        "--person-mask",
        "person_mask.png",
        "--garment-image",
        "garment.png",
    ]


@pytest.fixture
def mflux_in_context_edit_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate images using in-context editing.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False, require_prompt=False)
    parser.add_in_context_edit_arguments()
    parser.add_in_context_arguments()
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_in_context_edit_minimal_argv() -> list[str]:
    return [
        "mflux-generate-in-context-edit",
        "--reference-image",
        "reference.png",
        "--instruction",
        "make the hair black",
    ]


@pytest.mark.fast
def test_model_path_requires_model_arg(mflux_generate_parser):
    # when loading a model via --path, the model name still need to be specified
    with patch("sys.argv", "mflux-generate", "--path", "/some/saved/model"):
        assert pytest.raises(SystemExit, mflux_generate_parser.parse_args)


@pytest.mark.fast
def test_model_arg_not_in_file(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):
    metadata_file = temp_dir / "model.json"
    with metadata_file.open("wt") as m:
        del base_metadata_dict["model"]
        json.dump(base_metadata_dict, m, indent=4)

    # Create a parser that requires the model argument
    parser_with_required_model = _create_mflux_generate_parser(require_model_arg=True)

    # test model arg not provided in either flag or file should raise SystemExit with required parser
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        pytest.raises(SystemExit, parser_with_required_model.parse_args)

    # test value read from flag
    with patch('sys.argv', mflux_generate_minimal_argv + ['--model', 'dev', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.model == "dev"
        assert args.base_model is None
    # test value read from flag
    with patch('sys.argv', mflux_generate_minimal_argv + ['--model', 'schnell', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.model == "schnell"
        assert args.base_model is None


@pytest.mark.fast
def test_model_arg_in_file(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):
    metadata_file = temp_dir / "model.json"
    with metadata_file.open("wt") as m:
        base_metadata_dict["model"] = "dev"
        json.dump(base_metadata_dict, m, indent=4)
    # test value read from file
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.model == "dev"
    # test value read from flag, overrides value from file
    with patch('sys.argv', mflux_generate_minimal_argv + ['--model', 'schnell', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.model == "schnell"


@pytest.mark.fast
def test_base_model_arg_in_file(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):
    metadata_file = temp_dir / "model.json"
    with metadata_file.open("wt") as m:
        base_metadata_dict["model"] = "some-lab/some-model"
        base_metadata_dict["base_model"] = "dev"
        json.dump(base_metadata_dict, m, indent=4)
    # test value read from file
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.model == "some-lab/some-model"
        assert args.base_model == "dev"
    # test value read from flag, overrides value from file
    with patch('sys.argv', mflux_generate_minimal_argv + ['--base-model', 'schnell', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.model == "some-lab/some-model"
        # override metadata base model with CLI --base-model
        assert args.base_model == "schnell"


@pytest.mark.fast
def test_prompt_arg(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):
    metadata_file = temp_dir / "prompt.json"
    file_prompt = "origin of the universe"
    with metadata_file.open("wt") as m:
        base_metadata_dict["prompt"] = file_prompt
        json.dump(base_metadata_dict, m, indent=4)
    # test metadata config accepted, use mflux_generate_minimal_argv without fixture --prompt
    with patch('sys.argv', mflux_generate_minimal_argv[:-2] + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.prompt == file_prompt
    # test CLI override, use mflux_generate_minimal_argv without fixture --prompt
    cli_prompt = "place where monsters come from"
    with patch('sys.argv', mflux_generate_minimal_argv[:-2] + ['--prompt', cli_prompt, '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.prompt == cli_prompt


@pytest.mark.fast
def test_prompt_file_arg(mflux_generate_parser, mflux_generate_minimal_argv, temp_dir):
    # Create a prompt file
    prompt_content = "prompt from a file being re-read for each generation"
    prompt_file = temp_dir / "prompt.txt"
    with prompt_file.open("wt") as pf:
        pf.write(prompt_content)

    # Test that prompt-file is correctly read
    with patch('sys.argv', ["mflux-generate", "--prompt-file", prompt_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.prompt_file == prompt_file
        assert args.prompt is None  # prompt should be None since we're using prompt-file


@pytest.mark.fast
def test_prompt_and_prompt_file_mutually_exclusive(mflux_generate_parser, temp_dir):
    # Create a prompt file
    prompt_file = temp_dir / "prompt.txt"
    with prompt_file.open("wt") as pf:
        pf.write("some prompt content")

    # Test that using both --prompt and --prompt-file raises an error
    with pytest.raises(SystemExit):
        with patch('sys.argv', ["mflux-generate", "--prompt", "direct prompt", "--prompt-file", prompt_file.as_posix()]):  # fmt: off
            mflux_generate_parser.parse_args()


@pytest.mark.fast
def test_guidance_arg(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):  # fmt: off
    metadata_file = temp_dir / "guidance.json"
    with metadata_file.open("wt") as m:
        base_metadata_dict["guidance"] = 4.2
        json.dump(base_metadata_dict, m, indent=4)
    # test metadata config accepted
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.guidance == pytest.approx(4.2)
    # test CLI override
    with patch('sys.argv', mflux_generate_minimal_argv + ['--guidance', '5.0', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.guidance == pytest.approx(5.0)


@pytest.mark.fast
def test_quantize_arg(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):  # fmt: off
    metadata_file = temp_dir / "quantize.json"
    with metadata_file.open("wt") as m:
        base_metadata_dict["quantize"] = 4
        json.dump(base_metadata_dict, m, indent=4)
    # test metadata config accepted
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.quantize == 4
    # test CLI override
    with patch('sys.argv', mflux_generate_minimal_argv + ['--quantize', '8', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.quantize == 8


@pytest.mark.fast
def test_seed_arg(mflux_generate_parser, mflux_generate_minimal_model_argv, base_metadata_dict, temp_dir):  # fmt: off
    metadata_file = temp_dir / "seed.json"
    with metadata_file.open("wt") as m:
        base_metadata_dict["seed"] = 24
        json.dump(base_metadata_dict, m, indent=4)
    # test metadata config accepted
    with patch('sys.argv', mflux_generate_minimal_model_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.seed == [24]
        assert "_seed_{seed}" not in args.output

    # test CLI override
    with patch('sys.argv', mflux_generate_minimal_model_argv + ['--seed', '2424', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        # --seed arg overrides metadata
        assert args.seed == [2424]
        assert "_seed_{seed}" not in args.output

    with patch('sys.argv', mflux_generate_minimal_model_argv + ['--seed', '2424', '4848', '9696']):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.seed == [2424, 4848, 9696]
        assert "_seed_{seed}" in args.output

    with patch('sys.argv', mflux_generate_minimal_model_argv + ['--auto-seeds', '5', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        # auto-seeds defers to value from metadata, and is ignored
        assert len(args.seed) == 1
        assert args.seed == [24]
        assert "_seed_{seed}" not in args.output


@pytest.mark.fast
def test_auto_seeds_arg(mflux_generate_parser, mflux_generate_minimal_model_argv):
    with patch("sys.argv", mflux_generate_minimal_model_argv + ["--seed", "24", "48", "--auto-seeds", "5"]):
        args = mflux_generate_parser.parse_args()
        # auto-seeds defers to explicit values of --seed
        assert len(args.seed) == 2
        assert args.seed == [24, 48]
        assert "_seed_{seed}" in args.output

    for _ in range(0, 10):
        random_auto_seed_count = random.randint(2, 100)
        with patch("sys.argv", mflux_generate_minimal_model_argv + ["--auto-seeds", str(random_auto_seed_count)]):
            args = mflux_generate_parser.parse_args()
            assert len(set(args.seed)) == random_auto_seed_count
            assert "_seed_{seed}" in args.output
            for _ in args.seed:
                assert isinstance(_, int)


@pytest.mark.fast
def test_steps_arg(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):  # fmt: off
    metadata_file = temp_dir / "steps.json"
    with metadata_file.open("wt") as m:
        base_metadata_dict["steps"] = 8
        json.dump(base_metadata_dict, m, indent=4)

    # test user default value for dev
    with patch("sys.argv", mflux_generate_minimal_argv + ["--model", "dev"]):
        args = mflux_generate_parser.parse_args()
        assert args.steps == 25

    # test user default value for schnell
    with patch("sys.argv", mflux_generate_minimal_argv + ["--model", "schnell"]):
        args = mflux_generate_parser.parse_args()
        assert args.steps == 4

    # test metadata config accepted
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.steps == 8

    # test CLI override
    with patch('sys.argv', mflux_generate_minimal_argv + ['--steps', '12', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.steps == 12


@pytest.mark.fast
def test_lora_args(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):  # fmt: off
    test_paths = ["/some/lora/1.safetensors", "/some/lora/2.safetensors"]
    metadata_file = temp_dir / "lora_args.json"
    with metadata_file.open("wt") as m:
        base_metadata_dict["lora_paths"] = test_paths
        base_metadata_dict["lora_scales"] = [0.3, 0.7]
        json.dump(base_metadata_dict, m, indent=4)

    # test user default value
    with patch("sys.argv", mflux_generate_minimal_argv + ["-m", "schnell"]):
        args = mflux_generate_parser.parse_args()
        assert args.lora_paths is None
        assert args.lora_scales is None

    # Mock LoraResolution.resolve to bypass file validation for test purposes
    with patch("mflux.cli.parser.parsers.LoraResolution.resolve", side_effect=lambda x: x):
        # test metadata config accepted
        with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
            args = mflux_generate_parser.parse_args()
            assert args.lora_paths == test_paths
            assert args.lora_scales == [pytest.approx(0.3), pytest.approx(0.7)]

        # test CLI override that merges CLI loras and config file loras
        new_loras = [
            "--lora-paths",
            "/some/lora/3.safetensors",
            "/some/lora/4.safetensors",
            "--lora-scales",
            "0.1",
            "0.9",
        ]
        with patch('sys.argv', mflux_generate_minimal_argv + new_loras + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
            args = mflux_generate_parser.parse_args()
            assert len(args.lora_paths) == 4
            assert args.lora_paths == test_paths + new_loras[1:3]
            assert len(args.lora_scales) == 4
            assert args.lora_scales == [pytest.approx(v) for v in [0.3, 0.7, 0.1, 0.9]]


@pytest.mark.fast
def test_image_to_image_args(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):  # fmt: off
    metadata_file = temp_dir / "image_to_image.json"
    test_path = "/some/awesome/image.png"
    with metadata_file.open("wt") as m:
        base_metadata_dict["image_path"] = test_path
        json.dump(base_metadata_dict, m, indent=4)

    # test user default value
    with patch("sys.argv", mflux_generate_minimal_argv + ["-m", "dev"]):
        args = mflux_generate_parser.parse_args()
        assert args.image_path is None
        assert args.image_strength == 0.4  # default

    # test metadata config accepted
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.image_path == test_path
        assert args.image_strength == 0.4  # default

    # test strength override
    with patch('sys.argv', mflux_generate_minimal_argv + ['--image-strength', '0.7', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.image_path == test_path
        assert args.image_strength == 0.7

    # test image path override
    with patch('sys.argv', mflux_generate_minimal_argv + ['--image-path', '/some/better/image.png', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.image_path == Path("/some/better/image.png")
        assert args.image_strength == 0.4  # default


@pytest.mark.fast
def test_atomic_lora_arg(mflux_generate_parser, mflux_generate_minimal_model_argv):
    # Each --lora pairs a path with an optional scale; omitted scale defaults to 1.0.
    argv = mflux_generate_minimal_model_argv + [
        "--lora",
        "/some/lora/A.safetensors",
        "0.7",
        "--lora",
        "/some/lora/B.safetensors",
    ]
    with patch("mflux.cli.parser.parsers.LoraResolution.resolve", side_effect=lambda x: x):
        with patch("sys.argv", argv):
            args = mflux_generate_parser.parse_args()
    assert args.lora_paths == ["/some/lora/A.safetensors", "/some/lora/B.safetensors"]
    assert args.lora_scales == [pytest.approx(0.7), pytest.approx(1.0)]


@pytest.mark.fast
def test_atomic_lora_merges_with_metadata(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):  # fmt: off
    metadata_file = temp_dir / "atomic_lora.json"
    with metadata_file.open("wt") as m:
        base_metadata_dict["lora_paths"] = ["/meta/lora.safetensors"]
        base_metadata_dict["lora_scales"] = [0.3]
        json.dump(base_metadata_dict, m, indent=4)
    argv = mflux_generate_minimal_argv + [
        "--lora",
        "/cli/lora.safetensors",
        "0.9",
        "--config-from-metadata",
        metadata_file.as_posix(),
    ]
    with patch("mflux.cli.parser.parsers.LoraResolution.resolve", side_effect=lambda x: x):
        with patch("sys.argv", argv):
            args = mflux_generate_parser.parse_args()
    # metadata loras are prepended to the CLI-provided ones, scales stay aligned
    assert args.lora_paths == ["/meta/lora.safetensors", "/cli/lora.safetensors"]
    assert args.lora_scales == [pytest.approx(0.3), pytest.approx(0.9)]


@pytest.mark.fast
@pytest.mark.parametrize(
    "bad_args",
    [
        ["--lora", "A.safetensors", "0.7", "--lora-paths", "B.safetensors"],  # mixing new + legacy
        ["--lora", "A.safetensors", "not-a-number"],  # non-numeric scale
        ["--lora", "A.safetensors", "B.safetensors", "0.7"],  # too many values in one group
    ],
)
def test_atomic_lora_arg_errors(mflux_generate_parser, mflux_generate_minimal_model_argv, bad_args):
    with patch("sys.argv", mflux_generate_minimal_model_argv + bad_args):
        with pytest.raises(SystemExit):
            mflux_generate_parser.parse_args()


@pytest.mark.fast
def test_atomic_image_arg(mflux_generate_parser, mflux_generate_minimal_model_argv):
    # --image with explicit strength
    with patch("sys.argv", mflux_generate_minimal_model_argv + ["--image", "/some/photo.jpg", "0.6"]):
        args = mflux_generate_parser.parse_args()
    assert args.image_path == Path("/some/photo.jpg")
    assert args.image_strength == pytest.approx(0.6)

    # --image with omitted strength falls back to the default
    with patch("sys.argv", mflux_generate_minimal_model_argv + ["--image", "/some/photo.jpg"]):
        args = mflux_generate_parser.parse_args()
    assert args.image_path == Path("/some/photo.jpg")
    assert args.image_strength == ui_defaults.IMAGE_STRENGTH


@pytest.mark.fast
@pytest.mark.parametrize(
    "bad_args",
    [
        ["--image", "p.jpg", "--image-strength", "0.6"],  # mixing new + legacy
        ["--image", "p.jpg", "not-a-number"],  # non-numeric strength
        ["--image", "p.jpg", "0.6", "extra"],  # too many values
    ],
)
def test_atomic_image_arg_errors(mflux_generate_parser, mflux_generate_minimal_model_argv, bad_args):
    with patch("sys.argv", mflux_generate_minimal_model_argv + bad_args):
        with pytest.raises(SystemExit):
            mflux_generate_parser.parse_args()


@pytest.mark.fast
def test_required_init_image_satisfied_by_either_flag():
    # Kontext-style commands require an init image; either --image or the legacy
    # --image-path must satisfy it, and neither present is an error.
    def _required_image_parser() -> CommandLineParser:
        parser = CommandLineParser(description="x")
        parser.add_general_arguments()
        parser.add_model_arguments(require_model_arg=False)
        parser.add_image_generator_arguments(supports_metadata_config=True)
        parser.add_image_to_image_arguments(required=True)
        parser.add_output_arguments()
        return parser

    base = ["mflux-generate-kontext", "--prompt", "x", "-m", "dev"]

    with patch("sys.argv", base + ["--image", "p.jpg"]):
        assert _required_image_parser().parse_args().image_path == Path("p.jpg")

    with patch("sys.argv", base + ["--image-path", "p.jpg"]):
        assert _required_image_parser().parse_args().image_path == Path("p.jpg")

    with patch("sys.argv", base):
        with pytest.raises(SystemExit):
            _required_image_parser().parse_args()


@pytest.mark.fast
def test_image_outpaint_args(mflux_generate_parser, mflux_generate_minimal_argv, base_metadata_dict, temp_dir):  # fmt: off
    metadata_file = temp_dir / "image_outpaint.json"
    test_padding = "10,20,30,40"
    with metadata_file.open("wt") as m:
        base_metadata_dict["image_outpaint_padding"] = test_padding
        json.dump(base_metadata_dict, m, indent=4)

    # test user default value
    with patch("sys.argv", mflux_generate_minimal_argv + ["-m", "dev"]):
        args = mflux_generate_parser.parse_args()
        assert args.image_outpaint_padding is None

    # test metadata config accepted
    with patch('sys.argv', mflux_generate_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.image_outpaint_padding == BoxValues(10, 20, 30, 40)

    # test outpaint padding override in 4-value format
    with patch('sys.argv', mflux_generate_minimal_argv + ['--image-outpaint-padding', '5,15,25,35', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.image_outpaint_padding == BoxValues(5, 15, 25, 35)

    # test outpaint padding override in percentages in two-value format
    with patch('sys.argv', mflux_generate_minimal_argv + ['--image-outpaint-padding', '10%,20%', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.image_outpaint_padding == BoxValues("10%", "20%", "10%", "20%")

    # test outpaint padding override in percentages in three-value format, mixed int/percentages
    # also allowing whitespace between the box values
    with patch('sys.argv', mflux_generate_minimal_argv + ['--image-outpaint-padding', '10%, 50,   20%', '--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_parser.parse_args()
        assert args.image_outpaint_padding == BoxValues("10%", 50, "20%", 50)


@pytest.mark.fast
def test_controlnet_args(mflux_generate_controlnet_parser, mflux_generate_controlnet_minimal_argv, base_metadata_dict, temp_dir):  # fmt: off
    test_path = "/some/cnet/1.safetensors"
    metadata_file = temp_dir / "cnet_args.json"
    with metadata_file.open("wt") as m:
        base_metadata_dict["controlnet_image_path"] = test_path
        base_metadata_dict["controlnet_strength"] = 0.48
        json.dump(base_metadata_dict, m, indent=4)

    # test metadata config accepted
    with patch('sys.argv', mflux_generate_controlnet_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_controlnet_parser.parse_args()
        assert args.controlnet_image_path == test_path
        assert args.controlnet_strength == pytest.approx(0.48)
        assert args.controlnet_save_canny is False

    # test CLI override
    override_cnet = [
        "--controlnet-image-path",
        "/some/lora/2.safetensors",
        "--controlnet-strength",
        "0.85",
        "--controlnet-save-canny",
    ]
    with patch('sys.argv', mflux_generate_controlnet_minimal_argv + override_cnet + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_controlnet_parser.parse_args()
        assert args.controlnet_image_path == "/some/lora/2.safetensors"
        assert args.controlnet_strength == pytest.approx(0.85)
        assert args.controlnet_save_canny is True

    # test controlnet_save_canny is False when not specified
    with metadata_file.open("wt") as m:
        del base_metadata_dict["controlnet_save_canny"]
        json.dump(base_metadata_dict, m, indent=4)

    with patch('sys.argv', mflux_generate_controlnet_minimal_argv + ['--config-from-metadata', metadata_file.as_posix()]):  # fmt: off
        args = mflux_generate_controlnet_parser.parse_args()
        assert args.controlnet_save_canny is False


@pytest.mark.fast
def test_save_args(mflux_save_parser):
    with patch("sys.argv", ["mflux-save", "--model", "dev"]):
        # required --path not provided, exits to error
        assert pytest.raises(SystemExit, mflux_save_parser.parse_args)
    with patch("sys.argv", ["mflux-save", "--model", "dev", "--path", "/some/model/folder"]):
        # required --path not provided, exits to error
        args = mflux_save_parser.parse_args()
        assert args.path == "/some/model/folder"


@pytest.mark.fast
def test_fill_args(mflux_fill_parser, mflux_fill_minimal_argv):
    # Test required arguments
    with patch("sys.argv", mflux_fill_minimal_argv):
        args = mflux_fill_parser.parse_args()
        assert args.prompt == "meaning of life"
        assert args.image_path == Path("image.png")
        assert args.masked_image_path == Path("mask.png")
        # Default guidance for fill should be None (will be set to 30 in flux_generate_fill.py)
        assert args.guidance is None  # Parser doesn't set default, app does

    # Test with missing required arguments
    with patch("sys.argv", ["mflux-fill", "--prompt", "test"]):
        # Missing image_path and masked_image_path should raise SystemExit
        pytest.raises(SystemExit, mflux_fill_parser.parse_args)

    with patch("sys.argv", ["mflux-fill", "--image-path", "image.png", "--masked-image-path", "mask.png"]):
        # Missing prompt should raise SystemExit
        pytest.raises(SystemExit, mflux_fill_parser.parse_args)

    # Test with custom values
    custom_argv = mflux_fill_minimal_argv + ["--guidance", "30", "--steps", "20", "--height", "512", "--width", "512"]
    with patch("sys.argv", custom_argv):
        args = mflux_fill_parser.parse_args()
        assert args.guidance == pytest.approx(30.0)
        assert args.steps == 20
        assert args.height == 512
        assert args.width == 512


@pytest.mark.fast
def test_fill_args_with_metadata(mflux_fill_parser, mflux_fill_minimal_argv, base_metadata_dict, temp_dir):
    metadata_file = temp_dir / "fill_metadata.json"
    # Set up metadata with fill-related values
    with metadata_file.open("wt") as m:
        # Add masked_image_path to the metadata dictionary
        base_metadata_dict["masked_image_path"] = "metadata_mask.png"
        base_metadata_dict["prompt"] = "from metadata file"
        json.dump(base_metadata_dict, m, indent=4)

    # Test with minimal args and metadata
    # First modify the parser to support metadata config
    mflux_fill_parser.supports_metadata_config = True
    mflux_fill_parser.add_metadata_config()

    # Create a modified version of minimal_argv that includes all required arguments
    # that aren't in metadata
    minimal_metadata_argv = [
        "mflux-fill",
        "--config-from-metadata",
        metadata_file.as_posix(),
        "--prompt",
        "CLI prompt",
        "--image-path",
        "image.png",
        "--masked-image-path",
        "cli_mask.png",
    ]

    # Test command line arguments overriding metadata
    with patch("sys.argv", minimal_metadata_argv):
        args = mflux_fill_parser.parse_args()
        assert args.prompt == "CLI prompt"  # From CLI
        assert args.image_path == Path("image.png")  # From CLI
        assert args.masked_image_path == Path("cli_mask.png")  # From CLI


@pytest.mark.fast
def test_fill_default_guidance():
    # Create a parser just like in flux_generate_fill.py
    parser = CommandLineParser(description="Generate an image using the fill tool to complete masked areas.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_fill_arguments()
    parser.add_output_arguments()

    # Parse minimal arguments
    with patch(
        "sys.argv", ["mflux-fill", "--prompt", "test", "--image-path", "img.png", "--masked-image-path", "mask.png"]
    ):
        args = parser.parse_args()

        # Verify initial guidance value is None (no default set by parser)
        assert args.guidance is None

        # Simulate what happens in flux_generate_fill.py
        if args.guidance is None:
            args.guidance = ui_defaults.DEFAULT_DEV_FILL_GUIDANCE

        # Now check that guidance is correctly set to the default dev fill guidance
        assert args.guidance == ui_defaults.DEFAULT_DEV_FILL_GUIDANCE


@pytest.mark.fast
def test_scheduler_by_name(mflux_generate_parser, mflux_generate_minimal_model_argv):
    with patch("sys.argv", mflux_generate_minimal_model_argv + ["--scheduler", "linear"]):
        args = mflux_generate_parser.parse_args()
        # we just want the arg to be accepted
        assert args.scheduler == "linear"

    with patch("sys.argv", mflux_generate_minimal_model_argv + ["--scheduler", "foobar"]):
        # let the CLI will accept "foobar" - later Config will validate
        args = mflux_generate_parser.parse_args()

    with patch("sys.argv", mflux_generate_minimal_model_argv + ["--scheduler", "mflux.someplugin.SchedulerBaz"]):
        # let the CLI will accept external import paths - later Config will validate
        args = mflux_generate_parser.parse_args()


@pytest.mark.fast
def test_save_depth_args(mflux_save_depth_parser, mflux_save_depth_minimal_argv):
    # Test required arguments
    with patch("sys.argv", mflux_save_depth_minimal_argv):
        args = mflux_save_depth_parser.parse_args()
        assert args.image_path == Path("image.png")
        assert hasattr(args, "quantize")

    # Test with quantized argument
    with patch("sys.argv", mflux_save_depth_minimal_argv + ["--quantize", "4"]):
        args = mflux_save_depth_parser.parse_args()
        assert args.image_path == Path("image.png")
        assert args.quantize == 4

    # Test with output argument
    with patch("sys.argv", mflux_save_depth_minimal_argv + ["--output", "depth_map.png"]):
        args = mflux_save_depth_parser.parse_args()
        assert args.image_path == Path("image.png")
        assert args.output == "depth_map.png"


@pytest.mark.fast
def test_redux_args(mflux_redux_parser, mflux_redux_minimal_argv):
    # Test required arguments
    with patch("sys.argv", mflux_redux_minimal_argv):
        args = mflux_redux_parser.parse_args()
        assert len(args.redux_image_paths) == 2
        assert args.redux_image_paths[0] == Path("image1.png")
        assert args.redux_image_paths[1] == Path("image2.png")
        assert args.redux_image_strengths is None  # Default should be None

    # Test with more image paths
    with patch("sys.argv", ["mflux-redux", "--redux-image-paths", "image1.png", "image2.png", "image3.png"]):
        args = mflux_redux_parser.parse_args()
        assert len(args.redux_image_paths) == 3
        assert args.redux_image_paths[0] == Path("image1.png")
        assert args.redux_image_paths[1] == Path("image2.png")
        assert args.redux_image_paths[2] == Path("image3.png")

    # Test with redux_image_strengths parameter
    with patch("sys.argv", mflux_redux_minimal_argv + ["--redux-image-strengths", "0.8", "0.5"]):
        args = mflux_redux_parser.parse_args()
        assert len(args.redux_image_paths) == 2
        assert len(args.redux_image_strengths) == 2
        assert args.redux_image_strengths[0] == pytest.approx(0.8)
        assert args.redux_image_strengths[1] == pytest.approx(0.5)

    # Test with single redux_image_strength
    with patch("sys.argv", mflux_redux_minimal_argv + ["--redux-image-strengths", "0.3"]):
        args = mflux_redux_parser.parse_args()
        assert len(args.redux_image_paths) == 2
        assert len(args.redux_image_strengths) == 1
        assert args.redux_image_strengths[0] == pytest.approx(0.3)

    # Test with model argument
    with patch("sys.argv", mflux_redux_minimal_argv + ["--model", "dev"]):
        args = mflux_redux_parser.parse_args()
        assert len(args.redux_image_paths) == 2
        assert args.model == "dev"

    # Test with output argument
    with patch("sys.argv", mflux_redux_minimal_argv + ["--output", "redux_result.png"]):
        args = mflux_redux_parser.parse_args()
        assert len(args.redux_image_paths) == 2
        assert args.output == "redux_result.png"


@pytest.mark.fast
def test_concept_attention_args(mflux_concept_parser, mflux_concept_minimal_argv):
    # Test required arguments
    with patch("sys.argv", mflux_concept_minimal_argv):
        args = mflux_concept_parser.parse_args()
        assert args.prompt == "a beautiful landscape with a car"
        assert args.concept == "car"
        # Test defaults
        assert args.heatmap_layer_indices == list(range(15, 19))
        assert args.heatmap_timesteps is None

    # Test with missing required concept - should raise SystemExit
    with patch("sys.argv", ["mflux-concept", "--prompt", "test"]):
        pytest.raises(SystemExit, mflux_concept_parser.parse_args)

    # Test with missing regular prompt - should raise SystemExit
    with patch("sys.argv", ["mflux-concept", "--concept", "test concept"]):
        pytest.raises(SystemExit, mflux_concept_parser.parse_args)

    # Test with custom heatmap parameters
    custom_argv = mflux_concept_minimal_argv + [
        "--heatmap-layer-indices",
        "10",
        "11",
        "12",
        "--heatmap-timesteps",
        "0",
        "1",
        "2",
    ]
    with patch("sys.argv", custom_argv):
        args = mflux_concept_parser.parse_args()
        assert args.prompt == "a beautiful landscape with a car"
        assert args.concept == "car"
        assert args.heatmap_layer_indices == [10, 11, 12]
        assert args.heatmap_timesteps == [0, 1, 2]


@pytest.mark.fast
def test_catvton_args(mflux_catvton_parser, mflux_catvton_minimal_argv):
    # Test required arguments
    with patch("sys.argv", mflux_catvton_minimal_argv):
        args = mflux_catvton_parser.parse_args()
        assert args.person_image == "person.png"
        assert args.person_mask == "person_mask.png"
        assert args.garment_image == "garment.png"

        # Test prompt is None by default (set by the app, not parser)
        assert args.prompt is None

    # Test missing required arguments
    with patch("sys.argv", ["mflux-generate-in-context-catvton"]):
        pytest.raises(SystemExit, mflux_catvton_parser.parse_args)

    with patch("sys.argv", ["mflux-generate-in-context-catvton", "--person-image", "person.png"]):
        pytest.raises(SystemExit, mflux_catvton_parser.parse_args)

    with patch(
        "sys.argv", ["mflux-generate-in-context-catvton", "--person-image", "person.png", "--person-mask", "mask.png"]
    ):
        pytest.raises(SystemExit, mflux_catvton_parser.parse_args)

    # Test custom prompt can be set
    with patch("sys.argv", mflux_catvton_minimal_argv + ["--prompt", "custom prompt"]):
        args = mflux_catvton_parser.parse_args()
        assert args.prompt == "custom prompt"


@pytest.mark.fast
def test_in_context_edit_args(mflux_in_context_edit_parser, mflux_in_context_edit_minimal_argv):
    # Test required arguments with instruction
    with patch("sys.argv", mflux_in_context_edit_minimal_argv):
        args = mflux_in_context_edit_parser.parse_args()
        assert args.reference_image == "reference.png"
        assert args.instruction == "make the hair black"
        assert hasattr(mflux_in_context_edit_parser, "supports_in_context_edit")
        assert mflux_in_context_edit_parser.supports_in_context_edit is True

    # Test with prompt instead of instruction
    with patch(
        "sys.argv",
        [
            "mflux-generate-in-context-edit",
            "--reference-image",
            "reference.png",
            "--prompt",
            "A diptych with custom prompt",
        ],
    ):
        args = mflux_in_context_edit_parser.parse_args()
        assert args.reference_image == "reference.png"
        assert args.prompt == "A diptych with custom prompt"

    # Test missing required arguments
    with patch("sys.argv", ["mflux-generate-in-context-edit"]):
        pytest.raises(SystemExit, mflux_in_context_edit_parser.parse_args)

    with patch("sys.argv", ["mflux-generate-in-context-edit", "--reference-image", "reference.png"]):
        pytest.raises(SystemExit, mflux_in_context_edit_parser.parse_args)

    # Test both prompt and instruction provided (should error)
    with patch(
        "sys.argv",
        [
            "mflux-generate-in-context-edit",
            "--reference-image",
            "reference.png",
            "--prompt",
            "test prompt",
            "--instruction",
            "test instruction",
        ],
    ):
        pytest.raises(SystemExit, mflux_in_context_edit_parser.parse_args)


@pytest.mark.fast
def test_in_context_args(mflux_catvton_parser, mflux_catvton_minimal_argv):
    # Test save_full_image flag
    with patch("sys.argv", mflux_catvton_minimal_argv):
        args = mflux_catvton_parser.parse_args()
        assert args.save_full_image is False

    with patch("sys.argv", mflux_catvton_minimal_argv + ["--save-full-image"]):
        args = mflux_catvton_parser.parse_args()
        assert args.save_full_image is True


# ============================================================================
# Qwen Image Tests
# ============================================================================


@pytest.fixture
def mflux_qwen_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Qwen Image model.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_qwen_minimal_argv() -> list[str]:
    return ["mflux-generate-qwen", "--prompt", "a beautiful sunset"]


@pytest.mark.fast
def test_qwen_args(mflux_qwen_parser, mflux_qwen_minimal_argv):
    # Test minimal arguments
    with patch("sys.argv", mflux_qwen_minimal_argv):
        args = mflux_qwen_parser.parse_args()
        assert args.prompt == "a beautiful sunset"

    # Test with image-to-image
    with patch("sys.argv", mflux_qwen_minimal_argv + ["--image-path", "input.png", "--image-strength", "0.5"]):
        args = mflux_qwen_parser.parse_args()
        assert args.prompt == "a beautiful sunset"
        assert args.image_path == Path("input.png")
        assert args.image_strength == pytest.approx(0.5)

    # Test with quantization
    with patch("sys.argv", mflux_qwen_minimal_argv + ["--quantize", "8"]):
        args = mflux_qwen_parser.parse_args()
        assert args.quantize == 8


# ============================================================================
# Qwen Image Edit Tests
# ============================================================================


@pytest.fixture
def mflux_qwen_edit_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Qwen Image Edit.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_argument("--image-paths", type=Path, nargs="+", required=True, help="Local paths to init images")
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_qwen_edit_minimal_argv() -> list[str]:
    return ["mflux-generate-qwen-edit", "--prompt", "make it blue", "--image-paths", "image1.png"]


@pytest.mark.fast
def test_qwen_edit_args(mflux_qwen_edit_parser, mflux_qwen_edit_minimal_argv):
    # Test minimal arguments
    with patch("sys.argv", mflux_qwen_edit_minimal_argv):
        args = mflux_qwen_edit_parser.parse_args()
        assert args.prompt == "make it blue"
        assert len(args.image_paths) == 1
        assert args.image_paths[0] == Path("image1.png")

    # Test with multiple images
    with patch(
        "sys.argv",
        ["mflux-generate-qwen-edit", "--prompt", "combine these", "--image-paths", "img1.png", "img2.png", "img3.png"],
    ):
        args = mflux_qwen_edit_parser.parse_args()
        assert args.prompt == "combine these"
        assert len(args.image_paths) == 3

    # Test missing required image-paths
    with patch("sys.argv", ["mflux-generate-qwen-edit", "--prompt", "test"]):
        pytest.raises(SystemExit, mflux_qwen_edit_parser.parse_args)


# ============================================================================
# FIBO Tests
# ============================================================================


@pytest.fixture
def mflux_fibo_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using FIBO model.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.set_defaults(model="fibo")
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_fibo_minimal_argv() -> list[str]:
    return ["mflux-generate-fibo", "--prompt", '{"scene": "a forest"}']


@pytest.mark.fast
def test_fibo_args(mflux_fibo_parser, mflux_fibo_minimal_argv):
    # Test minimal arguments
    with patch("sys.argv", mflux_fibo_minimal_argv):
        args = mflux_fibo_parser.parse_args()
        assert args.prompt == '{"scene": "a forest"}'
        assert args.model == "fibo"
        assert args.steps == 50

    # Test with negative prompt
    with patch("sys.argv", mflux_fibo_minimal_argv + ["--negative-prompt", "blurry, low quality"]):
        args = mflux_fibo_parser.parse_args()
        assert args.negative_prompt == "blurry, low quality"

    # Test with image-to-image
    with patch("sys.argv", mflux_fibo_minimal_argv + ["--image-path", "input.png"]):
        args = mflux_fibo_parser.parse_args()
        assert args.image_path == Path("input.png")


@pytest.mark.fast
def test_fibo_args_restore_model_from_metadata_even_with_cli_default(mflux_fibo_parser, temp_dir):
    metadata_file = temp_dir / "fibo_metadata.json"
    metadata_file.write_text(
        json.dumps(
            {
                "model": "briaai/Fibo-lite",
                "prompt": '{"scene": "a forest"}',
                "seed": 42,
                "steps": 8,
            }
        )
    )

    with patch("sys.argv", ["mflux-generate-fibo", "--config-from-metadata", metadata_file.as_posix()]):
        args = mflux_fibo_parser.parse_args()
        assert args.model == "briaai/Fibo-lite"
        assert args.model_path == "briaai/Fibo-lite"
        assert args.steps == 8


@pytest.mark.fast
def test_fibo_args_cli_model_equals_syntax_overrides_metadata(mflux_fibo_parser, temp_dir):
    metadata_file = temp_dir / "fibo_model_equals_metadata.json"
    metadata_file.write_text(
        json.dumps(
            {
                "model": "briaai/Fibo-lite",
                "prompt": '{"scene": "a forest"}',
                "seed": 42,
                "steps": 8,
            }
        )
    )

    with patch(
        "sys.argv",
        [
            "mflux-generate-fibo",
            '--prompt={"scene":"a forest"}',
            "--model=fibo",
            "--config-from-metadata",
            metadata_file.as_posix(),
        ],
    ):
        args = mflux_fibo_parser.parse_args()
        assert args.model == "fibo"


@pytest.mark.fast
def test_fibo_args_cli_model_short_concat_syntax_overrides_metadata(mflux_fibo_parser, temp_dir):
    metadata_file = temp_dir / "fibo_model_short_concat_metadata.json"
    metadata_file.write_text(
        json.dumps(
            {
                "model": "briaai/Fibo-lite",
                "prompt": '{"scene": "a forest"}',
                "seed": 42,
                "steps": 8,
            }
        )
    )

    with patch(
        "sys.argv",
        [
            "mflux-generate-fibo",
            '--prompt={"scene":"a forest"}',
            "-mfibo",
            "--config-from-metadata",
            metadata_file.as_posix(),
        ],
    ):
        args = mflux_fibo_parser.parse_args()
        assert args.model == "fibo"


@pytest.fixture
def mflux_fibo_edit_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an edited image using Bria FIBO Edit.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.set_defaults(model="fibo-edit")
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, require_prompt=False)
    parser.add_argument("--image-path", type=Path, required=False)
    parser.add_argument("--mask-path", type=Path, default=None)
    parser.add_output_arguments()
    return parser


@pytest.mark.fast
def test_fibo_edit_args(mflux_fibo_edit_parser):
    with patch(
        "sys.argv",
        [
            "mflux-generate-fibo-edit",
            "--prompt",
            '{"short_description": "cat", "edit_instruction": "Turn the cat white"}',
            "--image-path",
            "input.png",
        ],
    ):
        args = mflux_fibo_edit_parser.parse_args()
        assert args.model == "fibo-edit"
        assert args.steps == 50
        assert args.image_path == Path("input.png")


@pytest.mark.fast
def test_fibo_edit_args_can_parse_without_prompt_for_runtime_validation(mflux_fibo_edit_parser):
    with patch(
        "sys.argv",
        [
            "mflux-generate-fibo-edit",
            "--image-path",
            "input.png",
        ],
    ):
        args = mflux_fibo_edit_parser.parse_args()
        assert args.prompt is None
        assert args.prompt_file is None
        assert args.image_path == Path("input.png")


@pytest.mark.fast
def test_fibo_edit_args_restore_image_and_mask_from_metadata(mflux_fibo_edit_parser, temp_dir):
    metadata_file = temp_dir / "fibo_edit_metadata.json"
    metadata_file.write_text(
        json.dumps(
            {
                "model": "fibo-edit",
                "prompt": '{"short_description": "cat", "edit_instruction": "Turn the cat white"}',
                "image_path": "input.png",
                "masked_image_path": "mask.png",
                "seed": 42,
                "steps": 20,
            }
        )
    )

    with patch("sys.argv", ["mflux-generate-fibo-edit", "--config-from-metadata", metadata_file.as_posix()]):
        args = mflux_fibo_edit_parser.parse_args()
        assert args.image_path == "input.png"
        assert args.mask_path == "mask.png"


@pytest.mark.fast
def test_fibo_edit_args_restore_model_from_metadata_even_with_cli_default(mflux_fibo_edit_parser, temp_dir):
    metadata_file = temp_dir / "fibo_edit_model_metadata.json"
    metadata_file.write_text(
        json.dumps(
            {
                "model": "briaai/Fibo-Edit",
                "prompt": '{"short_description": "cat", "edit_instruction": "Turn the cat white"}',
                "image_path": "input.png",
                "seed": 42,
                "steps": 20,
            }
        )
    )

    with patch("sys.argv", ["mflux-generate-fibo-edit", "--config-from-metadata", metadata_file.as_posix()]):
        args = mflux_fibo_edit_parser.parse_args()
        assert args.model == "briaai/Fibo-Edit"
        assert args.model_path == "briaai/Fibo-Edit"
        assert args.image_path == "input.png"


@pytest.mark.fast
def test_fibo_edit_rejects_non_edit_model_restored_from_metadata(mflux_fibo_edit_parser, temp_dir):
    metadata_file = temp_dir / "fibo_edit_incompatible_model_metadata.json"
    metadata_file.write_text(
        json.dumps(
            {
                "model": "briaai/FIBO",
                "prompt": '{"short_description": "cat", "edit_instruction": "Turn the cat white"}',
                "image_path": "input.png",
                "seed": 42,
                "steps": 20,
            }
        )
    )

    with patch("sys.argv", ["mflux-generate-fibo-edit", "--config-from-metadata", metadata_file.as_posix()]):
        args = mflux_fibo_edit_parser.parse_args()

    with pytest.raises(SystemExit):
        _resolve_fibo_edit_model_config(mflux_fibo_edit_parser, args)


# ============================================================================
# Z-Image Turbo Tests
# ============================================================================


@pytest.fixture
def mflux_z_image_turbo_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Z-Image Turbo.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_z_image_turbo_minimal_argv() -> list[str]:
    return ["mflux-generate-z-image-turbo", "--prompt", "a cat in space"]


@pytest.mark.fast
def test_z_image_turbo_args(mflux_z_image_turbo_parser, mflux_z_image_turbo_minimal_argv):
    # Test minimal arguments
    with patch("sys.argv", mflux_z_image_turbo_minimal_argv):
        args = mflux_z_image_turbo_parser.parse_args()
        assert args.prompt == "a cat in space"

    # Test with quantization
    with patch("sys.argv", mflux_z_image_turbo_minimal_argv + ["--quantize", "8"]):
        args = mflux_z_image_turbo_parser.parse_args()
        assert args.quantize == 8

    # Test with image-to-image
    with patch("sys.argv", mflux_z_image_turbo_minimal_argv + ["--image-path", "input.png", "--image-strength", "0.6"]):
        args = mflux_z_image_turbo_parser.parse_args()
        assert args.image_path == Path("input.png")
        assert args.image_strength == pytest.approx(0.6)

    # Mock LoraResolution.resolve to bypass file validation for test purposes
    with patch("mflux.cli.parser.parsers.LoraResolution.resolve", side_effect=lambda x: x):
        # Test with LoRA
        with patch(
            "sys.argv",
            mflux_z_image_turbo_minimal_argv + ["--lora-paths", "some/lora.safetensors", "--lora-scales", "0.8"],
        ):
            args = mflux_z_image_turbo_parser.parse_args()
            assert args.lora_paths == ["some/lora.safetensors"]
            assert args.lora_scales == [pytest.approx(0.8)]


# ============================================================================
# ERNIE-Image Tests
# ============================================================================


@pytest.fixture
def mflux_ernie_image_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using ERNIE-Image (50 steps, CFG) based on a prompt.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    parser.set_defaults(model="ernie-image", guidance=4.0)
    return parser


@pytest.fixture
def mflux_ernie_image_minimal_argv() -> list[str]:
    return ["mflux-generate-ernie-image", "--prompt", "a bicycle on a beach"]


@pytest.fixture
def mflux_ernie_image_turbo_parser() -> CommandLineParser:
    parser = CommandLineParser(
        description="Generate an image using ERNIE-Image-Turbo (distilled, 8 steps) based on a prompt."
    )
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    parser.set_defaults(model="ernie-image-turbo")
    return parser


@pytest.fixture
def mflux_ernie_image_turbo_minimal_argv() -> list[str]:
    return ["mflux-generate-ernie-image-turbo", "--prompt", "a bicycle on a beach"]


@pytest.mark.fast
def test_ernie_image_args(mflux_ernie_image_parser, mflux_ernie_image_minimal_argv):
    with patch("sys.argv", mflux_ernie_image_minimal_argv):
        args = mflux_ernie_image_parser.parse_args()
        assert args.prompt == "a bicycle on a beach"
        assert args.guidance == pytest.approx(4.0)
        assert mflux_ernie_image_parser.supports_dimension_scale_factor is True
        assert isinstance(args.width, ScaleFactor)
        assert isinstance(args.height, ScaleFactor)
        assert args.width.value == 1
        assert args.height.value == 1

    with patch("sys.argv", mflux_ernie_image_minimal_argv + ["--guidance", "3.0"]):
        args = mflux_ernie_image_parser.parse_args()
        assert args.guidance == pytest.approx(3.0)

    with patch("sys.argv", mflux_ernie_image_minimal_argv + ["--image-path", "input.png", "--image-strength", "0.7"]):
        args = mflux_ernie_image_parser.parse_args()
        assert args.image_path == Path("input.png")
        assert args.image_strength == pytest.approx(0.7)


@pytest.mark.fast
def test_ernie_image_turbo_args(mflux_ernie_image_turbo_parser, mflux_ernie_image_turbo_minimal_argv):
    with patch("sys.argv", mflux_ernie_image_turbo_minimal_argv):
        args = mflux_ernie_image_turbo_parser.parse_args()
        assert args.prompt == "a bicycle on a beach"
        assert args.guidance is None
        assert mflux_ernie_image_turbo_parser.supports_dimension_scale_factor is True
        assert isinstance(args.width, ScaleFactor)
        assert isinstance(args.height, ScaleFactor)

    with patch("sys.argv", mflux_ernie_image_turbo_minimal_argv + ["--guidance", "1.0"]):
        args = mflux_ernie_image_turbo_parser.parse_args()
        assert args.guidance == pytest.approx(1.0)


@pytest.mark.fast
def test_ernie_image_turbo_rejects_non_unit_guidance(mflux_ernie_image_turbo_minimal_argv):
    from mflux.models.ernie_image.cli import ernie_image_turbo_generate as turbo_cli

    with patch("sys.argv", mflux_ernie_image_turbo_minimal_argv + ["--guidance", "4.0"]):
        with patch.object(turbo_cli, "ErnieImage"):
            with patch.object(turbo_cli.CallbackManager, "register_callbacks", return_value=None):
                with pytest.raises(SystemExit) as exc_info:
                    turbo_cli.main()
                assert exc_info.value.code == 2


# ============================================================================
# Ideogram 4 Tests
# ============================================================================


@pytest.fixture
def mflux_ideogram4_parser() -> CommandLineParser:
    from mflux.models.ideogram4.model.ideogram4_scheduler import Ideogram4Scheduler

    parser = CommandLineParser(description="Generate an image using Ideogram 4.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_output_arguments()
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=sorted(Ideogram4Scheduler.PRESETS),
        help="Ideogram 4 sampler preset (step count, guidance schedule, and noise schedule). Default is V4_DEFAULT_20.",
    )
    parser.add_argument(
        "--strict-caption-validation",
        action="store_true",
        help="Fail when an Ideogram 4 JSON caption has schema warnings.",
    )
    return parser


@pytest.fixture
def mflux_ideogram4_minimal_argv() -> list[str]:
    return ["mflux-generate-ideogram4", "--prompt", '{"high_level_description": "a ceramic teapot"}']


@pytest.mark.fast
def test_ideogram4_args(mflux_ideogram4_parser, mflux_ideogram4_minimal_argv):
    with patch("sys.argv", mflux_ideogram4_minimal_argv):
        args = mflux_ideogram4_parser.parse_args()
        assert args.prompt == '{"high_level_description": "a ceramic teapot"}'
        assert args.preset is None
        assert args.strict_caption_validation is False
        assert args.model is None

    with patch("sys.argv", mflux_ideogram4_minimal_argv + ["--model", "ideogram4"]):
        args = mflux_ideogram4_parser.parse_args()
        assert args.model == "ideogram4"

    with patch("sys.argv", mflux_ideogram4_minimal_argv + ["--preset", "V4_TURBO_12"]):
        args = mflux_ideogram4_parser.parse_args()
        assert args.preset == "V4_TURBO_12"

    with patch("sys.argv", mflux_ideogram4_minimal_argv + ["--strict-caption-validation"]):
        args = mflux_ideogram4_parser.parse_args()
        assert args.strict_caption_validation is True


@pytest.mark.fast
@pytest.mark.parametrize("preset", ["V4_DEFAULT_20", "V4_QUALITY_48", "V4_TURBO_12"])
def test_ideogram4_preset_choices(mflux_ideogram4_parser, mflux_ideogram4_minimal_argv, preset: str):
    with patch("sys.argv", mflux_ideogram4_minimal_argv + ["--preset", preset]):
        args = mflux_ideogram4_parser.parse_args()
        assert args.preset == preset


@pytest.mark.fast
def test_ideogram4_rejects_invalid_preset(mflux_ideogram4_parser, mflux_ideogram4_minimal_argv):
    with patch("sys.argv", mflux_ideogram4_minimal_argv + ["--preset", "V4_NOT_A_PRESET"]):
        with pytest.raises(SystemExit) as exc_info:
            mflux_ideogram4_parser.parse_args()
        assert exc_info.value.code == 2


# ============================================================================
# Krea 2 Tests
# ============================================================================


@pytest.fixture
def mflux_krea2_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Krea-2 based on a prompt.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True, supports_dimension_scale_factor=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_krea2_minimal_argv() -> list[str]:
    return ["mflux-generate-krea2", "--prompt", "a red fox in a forest"]


@pytest.mark.fast
def test_krea2_args(mflux_krea2_parser, mflux_krea2_minimal_argv):
    with patch("sys.argv", mflux_krea2_minimal_argv):
        args = mflux_krea2_parser.parse_args()
        assert args.prompt == "a red fox in a forest"
        assert args.model is None
        assert args.scheduler == "linear"
        assert args.steps == 25
        assert args.guidance is None

    with patch("sys.argv", mflux_krea2_minimal_argv + ["--model", "krea2"]):
        args = mflux_krea2_parser.parse_args()
        assert args.model == "krea2"
        assert args.steps == 8

    with patch("sys.argv", mflux_krea2_minimal_argv + ["--scheduler", "euler"]):
        args = mflux_krea2_parser.parse_args()
        assert args.scheduler == "euler"

    with patch("sys.argv", mflux_krea2_minimal_argv + ["--steps", "8", "--guidance", "1.0"]):
        args = mflux_krea2_parser.parse_args()
        assert args.steps == 8
        assert args.guidance == pytest.approx(1.0)

    with patch("sys.argv", mflux_krea2_minimal_argv + ["--image-path", "input.png", "--image-strength", "0.65"]):
        args = mflux_krea2_parser.parse_args()
        assert args.image_path == Path("input.png")
        assert args.image_strength == pytest.approx(0.65)


@pytest.mark.fast
def test_krea2_lora_arguments(mflux_krea2_parser, mflux_krea2_minimal_argv):
    with patch("mflux.cli.parser.parsers.LoraResolution.resolve", side_effect=lambda x: x):
        with patch(
            "sys.argv",
            mflux_krea2_minimal_argv
            + [
                "--lora-paths",
                "krea/Krea-2-LoRA-darkbrush",
                "gokaygokay/Krea-2-Realism-LoRA",
                "--lora-scales",
                "1.0",
                "0.8",
            ],
        ):
            args = mflux_krea2_parser.parse_args()
            assert args.lora_paths == [
                "krea/Krea-2-LoRA-darkbrush",
                "gokaygokay/Krea-2-Realism-LoRA",
            ]
            assert args.lora_scales == [pytest.approx(1.0), pytest.approx(0.8)]


# ============================================================================
# Kontext Tests
# ============================================================================


@pytest.fixture
def mflux_kontext_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Flux Kontext.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments(required=True)
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_kontext_minimal_argv() -> list[str]:
    return ["mflux-generate-kontext", "--prompt", "make the background blue", "--image-path", "input.png"]


@pytest.mark.fast
def test_kontext_args(mflux_kontext_parser, mflux_kontext_minimal_argv):
    # Test minimal arguments
    with patch("sys.argv", mflux_kontext_minimal_argv):
        args = mflux_kontext_parser.parse_args()
        assert args.prompt == "make the background blue"
        assert args.image_path == Path("input.png")

    # Test missing required image-path
    with patch("sys.argv", ["mflux-generate-kontext", "--prompt", "test"]):
        pytest.raises(SystemExit, mflux_kontext_parser.parse_args)

    # Test with custom parameters
    with patch("sys.argv", mflux_kontext_minimal_argv + ["--steps", "15", "--guidance", "3.5"]):
        args = mflux_kontext_parser.parse_args()
        assert args.steps == 15
        assert args.guidance == pytest.approx(3.5)


# ============================================================================
# Depth Tests
# ============================================================================


@pytest.fixture
def mflux_depth_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image using Flux Depth.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_depth_arguments()
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_depth_minimal_argv() -> list[str]:
    return ["mflux-generate-depth", "--prompt", "a portrait", "--depth-image-path", "depth.png"]


@pytest.mark.fast
def test_depth_args(mflux_depth_parser, mflux_depth_minimal_argv):
    # Test minimal arguments with depth image
    with patch("sys.argv", mflux_depth_minimal_argv):
        args = mflux_depth_parser.parse_args()
        assert args.prompt == "a portrait"
        assert args.depth_image_path == Path("depth.png")

    # Test with source image (depth will be generated from it)
    with patch("sys.argv", ["mflux-generate-depth", "--prompt", "test", "--image-path", "source.png"]):
        args = mflux_depth_parser.parse_args()
        assert args.image_path == Path("source.png")
        assert args.depth_image_path is None

    # Test with save-depth-map flag
    with patch("sys.argv", mflux_depth_minimal_argv + ["--save-depth-map"]):
        args = mflux_depth_parser.parse_args()
        assert args.save_depth_map is True


# ============================================================================
# Info Tests
# ============================================================================


@pytest.fixture
def mflux_info_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Show information about an image.")
    parser.add_info_arguments()
    return parser


@pytest.fixture
def mflux_info_minimal_argv() -> list[str]:
    return ["mflux-info", "output.png"]  # positional argument, not --image-path


@pytest.mark.fast
def test_info_args(mflux_info_parser, mflux_info_minimal_argv):
    # Test minimal arguments (positional image_path)
    with patch("sys.argv", mflux_info_minimal_argv):
        args = mflux_info_parser.parse_args()
        assert args.image_path == "output.png"

    # Test missing required positional argument
    with patch("sys.argv", ["mflux-info"]):
        pytest.raises(SystemExit, mflux_info_parser.parse_args)


# ============================================================================
# Upscale Tests
# ============================================================================


@pytest.fixture
def mflux_upscale_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Upscale an image using Flux Controlnet.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments(required=True)
    parser.add_output_arguments()
    return parser


@pytest.fixture
def mflux_upscale_minimal_argv() -> list[str]:
    return ["mflux-upscale-controlnet", "--prompt", "enhance details", "--image-path", "low_res.png"]


@pytest.mark.fast
def test_upscale_args(mflux_upscale_parser, mflux_upscale_minimal_argv):
    # Test minimal arguments
    with patch("sys.argv", mflux_upscale_minimal_argv):
        args = mflux_upscale_parser.parse_args()
        assert args.prompt == "enhance details"
        assert args.image_path == Path("low_res.png")

    # Test missing required image-path
    with patch("sys.argv", ["mflux-upscale-controlnet", "--prompt", "test"]):
        pytest.raises(SystemExit, mflux_upscale_parser.parse_args)

    # Test with custom dimensions
    with patch("sys.argv", mflux_upscale_minimal_argv + ["--height", "1024", "--width", "1024"]):
        args = mflux_upscale_parser.parse_args()
        assert args.height == 1024
        assert args.width == 1024
