import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import PIL.Image
import pytest
from mlx.utils import tree_flatten

from mflux.models.common.config.model_config import AVAILABLE_MODELS, ModelConfig
from mflux.models.qwen21.cli import qwen21_controlnet_generate as cli
from mflux.models.qwen21.model.qwen21_transformer.qwen21_layout import QwenImage21Layout
from mflux.models.qwen21.model.qwen21_transformer.qwen_image21_transformer import QwenImage21Transformer
from mflux.models.qwen21.qwen21_initializer import Qwen21Initializer
from mflux.models.qwen21.variants.controlnet.qwen_image21_controlnet_transformer import QwenImage21ControlNet
from mflux.models.qwen21.variants.controlnet.qwen_image_21_controlnet import QwenImage21Controlnet

pytestmark = pytest.mark.fast

RESOURCES = Path(__file__).parent.parent / "resources"
# The shape of the Qwen-Image-2.1 transformer the official ControlNet was trained beside.
OFFICIAL = dict(num_layers=32, num_attention_heads=32, attention_head_dim=128, mlp_ratio=3, eps=1e-6)
# tests/resources/qwen21_controlnet_videox_fun_tiny.safetensors: random weights, inputs and outputs of
# VideoX-Fun's QwenImage21ControlTransformer2DModel at these dimensions (aigc-apps/VideoX-Fun @ 4b7b6402,
# CPU, float32), so the comparison with the reference runs without it.
TINY = dict(
    patch_size=1, in_channels=8, out_channels=8, num_layers=4, attention_head_dim=32, num_attention_heads=1,
    context_in_dim=16, mlp_ratio=2, axes_dims_rope=(8, 12, 12), eps=1e-6, causal_condition=True, control_in_dim=17,
)  # fmt: skip
TARGET, TEXT_LENGTH = (1, 4, 6), 5


class _Reference:
    @staticmethod
    def load() -> tuple[QwenImage21Transformer, QwenImage21ControlNet, dict[str, mx.array]]:
        tensors = dict(mx.load(str(RESOURCES / "qwen21_controlnet_videox_fun_tiny.safetensors")))
        weights = {key.removeprefix("weights."): value for key, value in tensors.items() if key.startswith("weights.")}
        control = {key: value for key, value in weights.items() if key.startswith("control")}
        base = {key: value for key, value in weights.items() if key not in control}
        transformer, controlnet = QwenImage21Transformer(TINY), QwenImage21ControlNet(TINY)
        transformer.load_weights(list(Qwen21Initializer._normalize_transformer_weights(base).items()), strict=True)
        controlnet.load_weights(list(control.items()), strict=True)
        return transformer, controlnet, tensors

    @staticmethod
    def run(transformer, controlnet, tensors, scale: float | None) -> np.ndarray:
        layout = QwenImage21Layout.create(mx.array([False] * TEXT_LENGTH), [TARGET], transformer.axes)
        arguments = (tensors["inputs.hidden_states"], tensors["inputs.encoder_hidden_states"], mx.array([0.6]), layout)
        if scale is None:
            return np.array(transformer(*arguments))
        return np.array(controlnet(transformer, *arguments, tensors["inputs.control_context"], scale))


@pytest.mark.parametrize(("scale", "expected"), [(1.0, "control_1.0"), (0.6, "control_0.6"), (None, "no_control")])
def test_the_control_branch_matches_the_videox_fun_reference(scale, expected):
    transformer, controlnet, tensors = _Reference.load()

    output = _Reference.run(transformer, controlnet, tensors, scale)

    # float32 through six blocks: about 2e-5 of the largest value. Control against no control differs by its size.
    reference = np.array(tensors[f"expected.{expected}"])
    assert np.abs(output - reference).max() < 1e-4 * np.abs(reference).max()


def test_strength_zero_is_the_base_model_and_the_control_moves_it():
    transformer, controlnet, tensors = _Reference.load()
    base = _Reference.run(transformer, controlnet, tensors, None)

    assert np.abs(_Reference.run(transformer, controlnet, tensors, 0.0) - base).max() < 1e-5
    assert np.abs(_Reference.run(transformer, controlnet, tensors, 1.0) - base).max() > 1.0


def test_the_control_input_needs_one_row_per_latent_token():
    transformer, controlnet, tensors = _Reference.load()
    layout = QwenImage21Layout.create(mx.array([False] * TEXT_LENGTH), [TARGET], transformer.axes)

    with pytest.raises(ValueError, match="one row per latent token"):
        controlnet(
            transformer,
            tensors["inputs.hidden_states"],
            tensors["inputs.encoder_hidden_states"],
            mx.array([0.6]),
            layout,
            tensors["inputs.control_context"][:, :-1],
        )


def test_the_official_checkpoint_names_are_exactly_this_module():
    # alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union @ 8a4702014d4d, names read from the safetensors header.
    # The loader is strict, so a name on either side that the other lacks stops the load.
    official = set((RESOURCES / "checkpoint_keys" / "qwen21_controlnet_union.txt").read_text().split())
    controlnet = QwenImage21ControlNet(OFFICIAL)

    assert {name for name, _ in tree_flatten(controlnet.parameters())} == official
    assert controlnet._slots == {layer: slot for slot, layer in enumerate(range(0, 32, 2))}
    assert controlnet.control_img_in.weight.shape == (4096, 129)
    assert ["before_proj" in block for block in controlnet.control_blocks] == [True] + [False] * 15


class _FakeVAE:
    def __init__(self):
        self.seen = []

    def encode(self, pixels: mx.array) -> mx.array:
        self.seen.append(np.array(pixels))
        _, _, height, width = pixels.shape
        return mx.full((1, 64, 1, height // 16, width // 16), float(len(self.seen)))


class _Context:
    @staticmethod
    def build(tmp_path, control: bool, inpaint: bool) -> tuple[np.ndarray, _FakeVAE]:
        model = QwenImage21Controlnet.__new__(QwenImage21Controlnet)
        model.__dict__["vae"] = _FakeVAE()
        white = PIL.Image.new("RGB", (64, 32), (255, 255, 255))
        mask = PIL.Image.new("L", (64, 32), 0)
        mask.paste(255, (0, 0, 32, 32))  # left half is regenerated
        context = model._control_context(
            white if control else None, white if inpaint else None, mask if inpaint else None, width=64, height=32
        )
        return np.array(context).reshape(2, 4, 129), model.vae


def test_the_control_input_packs_control_latents_mask_and_masked_source(tmp_path):
    context, vae = _Context.build(tmp_path, control=True, inpaint=True)

    assert (context[..., :64] == 1.0).all()  # first encode: the control image
    assert (context[..., 65:] == 2.0).all()  # second encode: the source
    assert context[:, :2, 64].tolist() == [[0.0, 0.0], [0.0, 0.0]]  # left half: regenerate
    assert context[:, 2:, 64].tolist() == [[1.0, 1.0], [1.0, 1.0]]  # right half: keep
    control_pixels, source_pixels = vae.seen
    assert control_pixels.shape == (1, 4, 32, 64) and (control_pixels == 1.0).all()  # white, opaque alpha
    assert (source_pixels[0, :3, :, :32] == 0.0).all()  # the region to regenerate is blanked before encoding
    assert (source_pixels[0, :3, :, 32:] == 1.0).all() and (source_pixels[0, 3] == 1.0).all()


def test_pure_control_leaves_the_inpaint_channels_at_zero(tmp_path):
    context, vae = _Context.build(tmp_path, control=True, inpaint=False)

    assert (context[..., :64] == 1.0).all() and (context[..., 64:] == 0.0).all()
    assert len(vae.seen) == 1


def test_the_registry_entry_names_the_base_and_the_controlnet():
    config = ModelConfig.qwen_image_21_controlnet_union()

    assert config is AVAILABLE_MODELS["qwen-image-2.1-controlnet-union"]
    assert (config.model_name, config.controlnet_model) == (
        "Qwen/Qwen-Image-2.1",
        "alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union",
    )
    base = ModelConfig.qwen_image_21()
    assert (config.sigma_max_shift, config.sigma_max_seq_len, config.sigma_shift_terminal) == (
        base.sigma_max_shift,
        base.sigma_max_seq_len,
        base.sigma_shift_terminal,
    )


class _Command:
    @staticmethod
    def args(monkeypatch, tmp_path, *extra: str):
        control = tmp_path / "edges.png"
        PIL.Image.new("RGB", (64, 32)).save(control)
        monkeypatch.setattr(sys, "argv", ["mflux-generate-qwen-2.1-controlnet", "--prompt", "a puffin", *extra])
        return cli.build_parser().parse_args(), control


def test_the_command_passes_its_options_to_the_model(monkeypatch, tmp_path):
    args, _ = _Command.args(
        monkeypatch, tmp_path, "--controlnet-image-path", "edges.png", "--controlnet-strength", "0.7",
        "--steps", "12", "--width", "640", "--height", "384", "--seed", "3",
    )  # fmt: skip
    calls = []
    model = SimpleNamespace(generate_image=lambda **kwargs: calls.append(kwargs))

    assert cli.Qwen21ControlnetCommand.validate(args).controlnet_model is not None
    cli.Qwen21ControlnetCommand.generate(model, args, seed=3, prompt="a puffin")

    assert calls == [
        dict(
            seed=3, prompt="a puffin", controlnet_image_path="edges.png", controlnet_strength=0.7,
            num_inference_steps=12, height=384, width=640, guidance=1.0, negative_prompt=args.negative_prompt,
            output_resolution=1024, image_path=None, mask_image=None,
        )
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ((), "Give --controlnet-image-path"),
        (("--image-path", "source.png"), "needs both --image-path and --mask-image"),
        (("--controlnet-image-path", "edges.png", "--width", "500"), "multiple of 32"),
        (("--controlnet-image-path", "edges.png", "--model", "qwen-image-2.1"), "not a Qwen-Image-2.1 ControlNet"),
    ],
)
def test_the_command_rejects_a_request_it_cannot_run_before_loading(monkeypatch, tmp_path, extra, message):
    args, _ = _Command.args(monkeypatch, tmp_path, *extra)

    with pytest.raises(ValueError, match=message):
        cli.Qwen21ControlnetCommand.validate(args)


def test_the_default_strength_is_the_full_control(monkeypatch, tmp_path):
    args, _ = _Command.args(monkeypatch, tmp_path, "--controlnet-image-path", "edges.png")

    assert (args.controlnet_strength, args.model, args.steps) == (1.0, "qwen-image-2.1-controlnet", 40)
