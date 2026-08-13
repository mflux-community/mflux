import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
from mflux.utils.version_util import VersionUtil

PATH = "tests/ernie_turbo_q8_saved/"


class TestErnieModelSaving:
    @pytest.mark.slow
    def test_save_and_load_quantized_ernie_turbo(self):
        TestErnieModelSaving.delete_folder_if_exists(PATH)

        try:
            model_a = ErnieImage(quantize=8, model_config=ModelConfig.ernie_image_turbo())
            image1 = model_a.generate_image(
                seed=7,
                prompt="A red bicycle against a cobblestone wall.",
                num_inference_steps=4,
                height=368,
                width=640,
                guidance=1.0,
            )
            model_a.save_model(PATH)
            del model_a

            _, quantization_level, mflux_version = WeightLoader._try_load_mflux_format(Path(PATH) / "vae")
            assert mflux_version == VersionUtil.get_mflux_version()
            assert quantization_level == 8

            model_b = ErnieImage(model_path=PATH, model_config=ModelConfig.ernie_image_turbo())
            image2 = model_b.generate_image(
                seed=7,
                prompt="A red bicycle against a cobblestone wall.",
                num_inference_steps=4,
                height=368,
                width=640,
                guidance=1.0,
            )
            np.testing.assert_array_equal(
                np.array(image1.image),
                np.array(image2.image),
                err_msg="Reloaded ERNIE turbo model produced a different image.",
            )

        finally:
            TestErnieModelSaving.delete_folder(PATH)

    @staticmethod
    def delete_folder(path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)

    @staticmethod
    def delete_folder_if_exists(path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)
