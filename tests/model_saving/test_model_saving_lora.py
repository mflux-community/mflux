import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from mflux.models.z_image.variants import ZImageTurbo

PATH = "tests/4bit/"
SIZE_TOLERANCE_RATIO = 0.05  # allow small metadata/header differences


class TestModelSavingLora:
    LORA_PATHS = [
        "renderartist/Technically-Color-Z-Image-Turbo",
        "renderartist/Classic-Painting-Z-Image-Turbo-LoRA",
    ]
    LORA_SCALES = [0.5, 0.5]

    @pytest.mark.slow
    def test_save_and_load_4bit_model_with_lora(self):
        # Clean up any existing temporary directories from previous test runs
        TestModelSavingLora._delete_folder_if_exists(PATH)

        try:
            lora_paths = TestModelSavingLora.LORA_PATHS

            # given a saved quantized model on disk (without LoRA)...
            modelA = ZImageTurbo(quantize=4)
            modelA.save_model(PATH)
            del modelA

            # ...and given an 'on-the-fly' quantized model which we generate an image from.
            # The adapters stay live on both models: baking folds them into the dense weights on
            # this load and onto the saved q4 grid on the next one, so only live adapters compare
            # the two quantized bases (#814).
            modelB = ZImageTurbo(
                quantize=4,
                lora_paths=lora_paths,
                lora_scales=TestModelSavingLora.LORA_SCALES,
                bake_lora=False,
            )
            image1 = modelB.generate_image(
                seed=44,
                prompt="mkym this is made of wool, pizza",
                num_inference_steps=2,
                height=368,
                width=640,
            )
            del modelB

            # when loading the quantized model from a local path (also without specifying bits) with a LoRA...
            modelC = ZImageTurbo(
                model_path=PATH,
                lora_paths=lora_paths,
                lora_scales=TestModelSavingLora.LORA_SCALES,
                bake_lora=False,
            )

            # ...and generating the identical image
            image2 = modelC.generate_image(
                seed=44,
                prompt="mkym this is made of wool, pizza",
                num_inference_steps=2,
                height=368,
                width=640,
            )

            # then we confirm that we get the exact *identical* image in both cases
            np.testing.assert_array_equal(
                np.array(image1.image),
                np.array(image2.image),
                err_msg="image2 doesn't match image1.",
            )

        finally:
            # cleanup
            TestModelSavingLora._delete_folder_if_exists(PATH)

    @pytest.mark.slow
    def test_save_baked_lora_loads_with_same_output(self):
        # Clean up any existing temporary directories from previous test runs
        TestModelSavingLora._delete_folder_if_exists(PATH)

        try:
            lora_paths = TestModelSavingLora.LORA_PATHS
            lora_scales = TestModelSavingLora.LORA_SCALES

            # Given: a model with LoRAs applied, save to disk.
            # Saving bakes LoRA into base weights for export.
            modelA = ZImageTurbo(
                quantize=None,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )
            modelA.save_model(PATH)
            del modelA

            # When: loading the baked model from disk and generating twice from fresh loads
            modelB = ZImageTurbo(model_path=PATH)
            image1 = modelB.generate_image(
                seed=44,
                prompt="mkym this is made of wool, pizza",
                num_inference_steps=9,
                height=368,
                width=640,
            )
            del modelB

            modelC = ZImageTurbo(model_path=PATH)
            image2 = modelC.generate_image(
                seed=44,
                prompt="mkym this is made of wool, pizza",
                num_inference_steps=9,
                height=368,
                width=640,
            )

            # Then: both fresh loads of the baked model are deterministic and equivalent
            np.testing.assert_array_equal(
                np.array(image1.image),
                np.array(image2.image),
                err_msg="image2 doesn't match image1 from baked model reload.",
            )
        finally:
            # cleanup
            TestModelSavingLora._delete_folder_if_exists(PATH)

    @pytest.mark.slow
    def test_save_with_lora_has_same_shard_count_as_base(self):
        base_path = "tests/4bit_base/"
        lora_path = "tests/4bit_with_lora/"

        TestModelSavingLora._delete_folder_if_exists(base_path)
        TestModelSavingLora._delete_folder_if_exists(lora_path)

        try:
            modelA = ZImageTurbo(quantize=4)
            modelA.save_model(base_path)
            del modelA

            modelB = ZImageTurbo(
                quantize=4,
                lora_paths=TestModelSavingLora.LORA_PATHS,
                lora_scales=TestModelSavingLora.LORA_SCALES,
            )
            modelB.save_model(lora_path)
            del modelB

            base_shards = list((Path(base_path) / "transformer").glob("*.safetensors"))
            lora_shards = list((Path(lora_path) / "transformer").glob("*.safetensors"))

            assert len(base_shards) == len(lora_shards), "LoRA save should not inflate transformer shard count"

            # Also assert the total saved size stays within a tight bound to catch shard bloat
            base_size = TestModelSavingLora._dir_size_bytes(base_path)
            lora_size = TestModelSavingLora._dir_size_bytes(lora_path)

            # LoRA baking should leave sizes effectively unchanged; allow a small tolerance for metadata
            max_allowed = base_size * (1 + SIZE_TOLERANCE_RATIO)
            assert lora_size <= max_allowed, f"LoRA save size grew unexpectedly: base={base_size}B vs lora={lora_size}B"
        finally:
            TestModelSavingLora._delete_folder_if_exists(base_path)
            TestModelSavingLora._delete_folder_if_exists(lora_path)

    @staticmethod
    def _delete_folder_if_exists(path: str) -> None:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Deleted folder: {path}")
        else:
            print(f"Folder does not exist: {path}")

    @staticmethod
    def _dir_size_bytes(path: str | Path) -> int:
        p = Path(path)
        if not p.exists():
            return 0
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
