from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_vae.vae import VAE
from mflux.models.z_image.variants.controlnet.control_types import ControlSpec, ControlType
from mflux.utils.image_util import ImageUtil

log = logging.getLogger(__name__)

# DepthPro, HED and OpenPose are separate models; load each once, and only if that control is used.
_DEPTH_PRO = None
_HED = None
_OPENPOSE = None


def _get_depth_pro():
    global _DEPTH_PRO
    if _DEPTH_PRO is None:
        from mflux.models.depth_pro.model.depth_pro import DepthPro

        _DEPTH_PRO = DepthPro()
    return _DEPTH_PRO


def _get_hed():
    global _HED
    if _HED is None:
        from mflux.models.hed import HED

        _HED = HED()
    return _HED


def _get_openpose():
    global _OPENPOSE
    if _OPENPOSE is None:
        from mflux.models.openpose import OpenPoseBody

        _OPENPOSE = OpenPoseBody()
    return _OPENPOSE


@dataclass(frozen=True)
class EncodedControls:
    control_latents: list[mx.array]
    strengths: list[float]
    types: list[ControlType]
    images: list[PIL.Image.Image]


class ZImageControlnetUtil:
    @staticmethod
    def encode_controls(
        *,
        vae: VAE,
        width: int,
        height: int,
        controls: list[ControlSpec],
        control_in_dim: int = 33,
    ) -> EncodedControls:
        if len(controls) == 0:
            raise ValueError("At least one control must be provided.")

        control_latents: list[mx.array] = []
        strengths: list[float] = []
        types: list[ControlType] = []
        images: list[PIL.Image.Image] = []

        for control in controls:
            img = ImageUtil.load_image(control.image_path)
            img = ZImageControlnetUtil._scale_image(img=img, width=width, height=height)
            img = ZImageControlnetUtil._preprocess(img, control.type)

            arr = ImageUtil.to_array(img)
            latent = vae.encode(arr)  # (1, 16, 1, H/8, W/8)
            latent = ZImageLatentCreator.pack_latents(latent, height=height, width=width)  # (16, 1, H/8, W/8)

            # Diffusers ZImageControlNetPipeline behavior: pad the 16ch base latents with zeros up
            # to the ControlNet's configured input dim (33 for Union 2.1).
            if latent.shape[0] > control_in_dim:
                raise ValueError(
                    f"Control latents have {latent.shape[0]} channels but the ControlNet expects {control_in_dim}."
                )
            if latent.shape[0] < control_in_dim:
                padding = mx.zeros((control_in_dim - latent.shape[0], *latent.shape[1:]), dtype=latent.dtype)
                latent = mx.concatenate([latent, padding], axis=0)

            control_latents.append(latent)
            strengths.append(float(control.strength))
            types.append(control.type)
            images.append(img)

        return EncodedControls(control_latents=control_latents, strengths=strengths, types=types, images=images)

    @staticmethod
    def _preprocess(img: PIL.Image.Image, control_type: ControlType) -> PIL.Image.Image:
        # Union checkpoints accept any modality as an already-preprocessed hint, and every modality is
        # now computed locally: canny/mlsd via OpenCV, depth via DepthPro, hed and pose via native-MLX
        # ports of ControlNetHED and OpenPose.
        if control_type == ControlType.canny:
            # OpenCV Canny expects an 8-bit single-channel image.
            gray_u8 = np.array(img.convert("L"), dtype=np.uint8)
            edges_u8 = cv2.Canny(gray_u8, 100, 200)
            edges_rgb = np.repeat(edges_u8[:, :, None], 3, axis=2)
            return PIL.Image.fromarray(edges_rgb)

        if control_type == ControlType.mlsd:
            return ZImageControlnetUtil._mlsd(img)

        if control_type == ControlType.depth:
            return ZImageControlnetUtil._depth(img)

        if control_type == ControlType.hed:
            return _get_hed().edge_map(img)

        if control_type == ControlType.pose:
            return _get_openpose().pose_map(img)

        return img

    @staticmethod
    def _mlsd(img: PIL.Image.Image) -> PIL.Image.Image:
        # Straight line segments as white strokes on black, approximating the MLSD hint with OpenCV's
        # LSD (no neural model). Good for architecture and interiors where the strong cues are edges.
        if not hasattr(cv2, "createLineSegmentDetector"):
            raise RuntimeError(
                "The mlsd control needs OpenCV's LineSegmentDetector, which opencv-python 4.1 through "
                "4.7 does not ship. Install opencv-python>=4.8."
            )
        gray_u8 = np.array(img.convert("L"), dtype=np.uint8)
        lines = cv2.createLineSegmentDetector().detect(gray_u8)[0]
        canvas = np.zeros((gray_u8.shape[0], gray_u8.shape[1], 3), dtype=np.uint8)
        if lines is not None:
            for line in lines:
                x0, y0, x1, y1 = (int(round(v)) for v in line[0])
                cv2.line(canvas, (x0, y0), (x1, y1), (255, 255, 255), 1)
        return PIL.Image.fromarray(canvas)

    @staticmethod
    def _depth(img: PIL.Image.Image) -> PIL.Image.Image:
        # DepthPro (native MLX) reads a file path, so round-trip the scaled control image through a
        # temp file and hand back its depth map.
        import os
        import tempfile

        depth_pro = _get_depth_pro()
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            img.convert("RGB").save(tmp_path)
            depth_image = depth_pro.create_depth_map(image_path=tmp_path).depth_image
        finally:
            os.unlink(tmp_path)
        return depth_image.convert("RGB")

    @staticmethod
    def _scale_image(*, img: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image:
        if height != img.height or width != img.width:
            log.warning(
                f"Control image {img.width}x{img.height} has different dimensions than requested. Resizing to {width}x{height}"
            )
            img = img.resize((width, height), PIL.Image.LANCZOS)
        return img
