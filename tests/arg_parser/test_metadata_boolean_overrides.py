import json
import sys

import pytest

from mflux.cli.parser.parsers import CommandLineParser


def _sidecar(tmp_path, **values):
    path = tmp_path / "prior.json"
    path.write_text(json.dumps({"prompt": "x", "model": "schnell", "seed": 1, "steps": 4, **values}))
    return path


def _parse(sidecar, extra, monkeypatch):
    parser = CommandLineParser(description="t")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_pid_decode_arguments()
    parser.add_output_arguments()
    monkeypatch.setattr(sys, "argv", ["prog", "--config-from-metadata", str(sidecar)] + extra)
    return parser.parse_args()


@pytest.mark.fast
def test_metadata_supplies_the_flag_when_the_command_line_is_silent(tmp_path, monkeypatch):
    args = _parse(_sidecar(tmp_path, pid_decode=True), [], monkeypatch)
    assert args.pid_decode is True


@pytest.mark.fast
def test_the_negative_spelling_beats_the_metadata(tmp_path, monkeypatch):
    # Without this there is no way to reuse a sidecar and turn PiD off: --pid-decode is the
    # only spelling argparse knows, and it says the same thing the metadata already said.
    args = _parse(_sidecar(tmp_path, pid_decode=True), ["--no-pid-decode"], monkeypatch)
    assert args.pid_decode is False


@pytest.mark.fast
def test_the_positive_spelling_beats_a_false_in_the_metadata(tmp_path, monkeypatch):
    args = _parse(_sidecar(tmp_path, pid_decode=False), ["--pid-decode"], monkeypatch)
    assert args.pid_decode is True
