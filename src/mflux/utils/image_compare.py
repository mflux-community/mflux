import os
from pathlib import Path

import numpy as np
from PIL import Image


class ImageCompare:
    DEFAULT_MISMATCH_THRESHOLD = 0.15  # Set by eye test
    ENV_MISMATCH_THRESHOLD = float(os.environ.get("MFLUX_IMAGE_MISMATCH_THRESHOLD", DEFAULT_MISMATCH_THRESHOLD))

    @staticmethod
    def check_images_close_enough(
        image1_path: str | Path,
        image2_path: str | Path,
        error_message_prefix: str,
        mismatch_threshold: float | None = None,
    ) -> float:
        if mismatch_threshold is None:
            mismatch_threshold = ImageCompare.ENV_MISMATCH_THRESHOLD

        image1_path = Path(image1_path)
        image2_path = Path(image2_path)
        image1_data = np.array(Image.open(image1_path))
        image2_data = np.array(Image.open(image2_path))
        closeness_array = np.isclose(
            image1_data,
            image2_data,
            rtol=float(os.environ.get("MFLUX_IMAGE_ALLCLOSE_RTOL", "0.1")),
            atol=float(os.environ.get("MFLUX_IMAGE_ALLCLOSE_ATOL", "2")),  # rtol alone collapses to exact-match near black
        )
        num_mismatched = np.count_nonzero(~closeness_array)
        total_elements = image1_data.size
        mismatch_ratio = num_mismatched / total_elements
        if mismatch_ratio > mismatch_threshold:
            raise AssertionError(
                f"{error_message_prefix} Check {image1_path} vs {image2_path} :: their elements are {mismatch_ratio:.1%} different. Fails assertion for {mismatch_threshold=}"
            )
        return mismatch_ratio

    @staticmethod
    def compare_images(
        image1_path: str | Path,
        image2_path: str | Path,
        mismatch_threshold: float | None = None,
    ) -> dict:
        image1_path = Path(image1_path)
        image2_path = Path(image2_path)
        image1_data = np.array(Image.open(image1_path))
        image2_data = np.array(Image.open(image2_path))

        rtol = float(os.environ.get("MFLUX_IMAGE_ALLCLOSE_RTOL", "0.1"))
        atol = float(os.environ.get("MFLUX_IMAGE_ALLCLOSE_ATOL", "2"))
        closeness_array = np.isclose(
            image1_data,
            image2_data,
            rtol=rtol,
            atol=atol,
        )

        num_mismatched = np.count_nonzero(~closeness_array)
        total_elements = image1_data.size
        mismatch_ratio = num_mismatched / total_elements

        result = {
            "mismatch_ratio": mismatch_ratio,
            "total_pixels": total_elements,
            "mismatched_pixels": num_mismatched,
            "image1_shape": image1_data.shape,
            "image2_shape": image2_data.shape,
        }

        if mismatch_threshold is not None:
            result["passes_threshold"] = mismatch_ratio <= mismatch_threshold
            result["threshold"] = mismatch_threshold

        return result
