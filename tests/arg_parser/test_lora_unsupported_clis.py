# Boogu and FIBO have no LoRA mapping in mflux, but used to accept every LoRA flag anyway:
# the parser resolved the adapter — downloading it from HuggingFace when the name was a repo
# id — and the initializers then dropped it, with mflux-capabilities reporting "lora": true.

import json
import sys

import pytest

from mflux.cli import capabilities
from mflux.models.boogu.cli import boogu_image_generate
from mflux.models.fibo.cli import fibo_edit, fibo_generate
from mflux.models.flux.cli import flux_generate

LORA_LESS_CLIS = pytest.mark.parametrize(
    "module", [boogu_image_generate, fibo_generate, fibo_edit], ids=["boogu", "fibo", "fibo-edit"]
)

LORA_LESS_COMMANDS = ["mflux-generate-boogu", "mflux-generate-fibo", "mflux-generate-fibo-edit"]


@pytest.mark.fast
@LORA_LESS_CLIS
@pytest.mark.parametrize(
    "lora_argv",
    [
        ["--lora", "adapter.safetensors"],
        ["--lora-paths", "adapter.safetensors"],
        ["--lora-scales", "0.8"],
        ["--lora-style", "storyboard"],
        ["--bake-lora"],
    ],
)
def test_every_lora_spelling_is_refused(module, lora_argv, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prog", "--prompt", "x", *lora_argv])
    with pytest.raises(SystemExit) as exit_info:
        module.build_parser().parse_args()
    assert exit_info.value.code == 2
    assert lora_argv[0] in capsys.readouterr().err


@pytest.mark.fast
@LORA_LESS_CLIS
def test_the_parser_declares_no_lora_support(module):
    assert module.build_parser().supports_lora is False


@pytest.mark.fast
def test_a_lora_capable_cli_still_takes_the_flags(monkeypatch):
    # Guard on the other side of the change: --lora must keep working where it is applied.
    monkeypatch.setattr(sys, "argv", ["prog", "--model", "schnell", "--prompt", "x"])
    parser = flux_generate.build_parser()
    assert parser.supports_lora is True
    assert parser.parse_args().lora_paths is None


@pytest.mark.fast
@pytest.mark.parametrize("command", LORA_LESS_COMMANDS)
def test_capabilities_no_longer_advertises_lora(command):
    entry = next(c for c in capabilities.build_capabilities()["commands"] if c["command"] == command)
    assert entry["traits"]["lora"] is False
    assert [option["flag"] for option in entry["options"] if "lora" in option["flag"]] == []


@pytest.mark.fast
def test_a_sidecar_carrying_lora_paths_still_restores(tmp_path, monkeypatch):
    # A sidecar written by a LoRA run (or copied between models) must not resurrect the
    # discarded flags through the metadata restore path, nor make the CLI exit.
    sidecar = tmp_path / "prior.metadata.json"
    sidecar.write_text(json.dumps({"prompt": "a red cube", "seed": 7, "lora_paths": ["adapter.safetensors"]}))
    monkeypatch.setattr(sys, "argv", ["prog", "--config-from-metadata", str(sidecar)])

    args = boogu_image_generate.build_parser().parse_args()

    assert args.prompt == "a red cube"
    assert not hasattr(args, "lora_paths")
