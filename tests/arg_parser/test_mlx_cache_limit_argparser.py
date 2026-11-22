from unittest.mock import patch

import pytest

from mflux.cli.parser.parsers import CommandLineParser


@pytest.fixture
def parser() -> CommandLineParser:
    parser = CommandLineParser(description="Parser for MLX cache limit flag tests.")
    parser.add_general_arguments()
    return parser


@pytest.mark.fast
def test_mlx_cache_limit_gb_parses(parser: CommandLineParser):
    with patch("sys.argv", ["mflux-generate", "--mlx-cache-limit-gb", "8"]):
        args = parser.parse_args()
        assert args.mlx_cache_limit_gb == 8.0


@pytest.mark.fast
def test_mlx_cache_limit_gb_rejects_non_positive_value(parser: CommandLineParser):
    with patch("sys.argv", ["mflux-generate", "--mlx-cache-limit-gb", "0"]):
        with pytest.raises(SystemExit):
            parser.parse_args()


@pytest.mark.fast
@pytest.mark.parametrize("value", ["nan", "inf"])
def test_mlx_cache_limit_gb_rejects_a_value_that_is_not_finite(parser: CommandLineParser, value: str):
    with patch("sys.argv", ["mflux-generate", "--mlx-cache-limit-gb", value]):
        with pytest.raises(SystemExit):
            parser.parse_args()
