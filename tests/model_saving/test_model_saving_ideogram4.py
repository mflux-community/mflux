import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.ideogram4.variants.txt2img.ideogram4 import Ideogram4
from mflux.utils.version_util import VersionUtil

PATH = "tests/ideogram4_q8_saved/"


def _valid_json_caption() -> dict:
    return {
        "high_level_description": "A white ceramic teapot on a simple studio table.",
        "style_description": {
            "aesthetics": "clean, calm, minimal",
            "lighting": "soft diffuse studio lighting",
            "photo": "eye-level, 50mm lens, shallow depth of field",
            "medium": "photograph",
            "color_palette": ["#FFFFFF", "#E5E0D8", "#2E2E2E"],
        },
        "compositional_deconstruction": {
            "background": "A neutral studio tabletop with a pale wall behind it.",
            "elements": [
                {
                    "type": "obj",
                    "bbox": [250, 320, 780, 690],
                    "desc": "A glossy white ceramic teapot with a curved handle and short spout.",
                }
            ],
        },
    }


class TestIdeogram4ModelSaving:
    @pytest.mark.slow
    def test_save_and_load_quantized_ideogram4(self):
        TestIdeogram4ModelSaving.delete_folder_if_exists(PATH)

        try:
            prompt = _valid_json_caption()
            model_a = Ideogram4(quantize=8, model_config=ModelConfig.ideogram4_fp8())
            image1 = model_a.generate_image(
                seed=42,
                prompt=prompt,
                num_inference_steps=4,
                height=256,
                width=256,
            )
            model_a.save_model(PATH)
            del model_a

            _, quantization_level, mflux_version = WeightLoader._try_load_mflux_format(Path(PATH) / "vae")
            assert mflux_version == VersionUtil.get_mflux_version()
            assert quantization_level == 8

            model_b = Ideogram4(model_path=PATH, model_config=ModelConfig.ideogram4_fp8())
            image2 = model_b.generate_image(
                seed=42,
                prompt=prompt,
                num_inference_steps=4,
                height=256,
                width=256,
            )
            np.testing.assert_array_equal(
                np.array(image1.image),
                np.array(image2.image),
                err_msg="Reloaded Ideogram 4 model produced a different image.",
            )

        finally:
            TestIdeogram4ModelSaving.delete_folder(PATH)

    @staticmethod
    def delete_folder(path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)

    @staticmethod
    def delete_folder_if_exists(path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)
