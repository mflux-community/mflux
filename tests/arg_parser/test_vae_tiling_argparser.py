from unittest.mock import patch

import pytest

from mflux.cli.parser.parsers import CommandLineParser


@pytest.fixture
def parser() -> CommandLineParser:
    parser = CommandLineParser(description="Parser for VAE tiling flag tests.")
    parser.add_general_arguments()
    return parser


@pytest.mark.fast
def test_vae_tiling_defaults_to_disabled(parser: CommandLineParser):
    with patch("sys.argv", ["mflux-generate"]):
        args = parser.parse_args()
        assert args.vae_tiling is False
        assert args.vae_tile_size is None


@pytest.mark.fast
def test_vae_tiling_flag_parses(parser: CommandLineParser):
    with patch("sys.argv", ["mflux-generate", "--vae-tiling"]):
        args = parser.parse_args()
        assert args.vae_tiling is True


@pytest.mark.fast
def test_vae_tile_size_parses(parser: CommandLineParser):
    with patch("sys.argv", ["mflux-generate", "--vae-tile-size", "256"]):
        args = parser.parse_args()
        assert args.vae_tile_size == 256
