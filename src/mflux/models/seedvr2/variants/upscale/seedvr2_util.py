from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.utils.scale_factor import ScaleFactor


class SeedVR2Util:
    @staticmethod
    def preprocess_image(
        image_path: str | Path,
        resolution: int | ScaleFactor,
        softness: float = 0.0,
    ) -> tuple[mx.array, int, int]:
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        if isinstance(resolution, ScaleFactor):
            target_res = resolution.get_scaled_value(min(w, h))
        else:
            target_res = resolution

        scale = target_res / min(w, h)
        true_w = int(w * scale)
        true_h = int(h * scale)
        true_w = (true_w // 2) * 2
        true_h = (true_h // 2) * 2

        # Map normalized 0.0-1.0 to factor 1.0-8.0
        # factor = 1.0 + normalized_value * 7.0
        factor = 1.0 + (max(0.0, min(1.0, softness)) * 7.0)

        if factor > 1.0:
            down_w = max(2, int(true_w / factor))
            down_h = max(2, int(true_h / factor))
            down = image.resize((down_w, down_h), Image.Resampling.BICUBIC)
            resized = down.resize((true_w, true_h), Image.Resampling.BICUBIC)
        else:
            resized = image.resize((true_w, true_h), Image.Resampling.BICUBIC)

        pad_w = (16 - (true_w % 16)) % 16
        pad_h = (16 - (true_h % 16)) % 16
        if pad_w or pad_h:
            padded = Image.new("RGB", (true_w + pad_w, true_h + pad_h), (0, 0, 0))
            padded.paste(resized, (0, 0))
            resized = padded

        img_mx = mx.array(np.array(resized)).astype(mx.float32) / 255.0
        img_mx = mx.clip(img_mx, 0.0, 1.0)
        img_mx = img_mx * 2.0 - 1.0
        img_mx = mx.transpose(img_mx, (2, 0, 1))
        img_mx = img_mx[None, ...]
        return img_mx, true_h, true_w

    @staticmethod
    def apply_color_correction(
        content: mx.array,
        style: mx.array,
        luminance_weight: float = 0.8,
    ) -> mx.array:
        return SeedVR2Util._lab_color_transfer_exact(content, style, luminance_weight=luminance_weight)

    @staticmethod
    def _lab_color_transfer_exact(content: mx.array, style: mx.array, luminance_weight: float = 0.8) -> mx.array:
        content_f = content.astype(mx.float32)
        style_f = style.astype(mx.float32)

        content_np = np.array(content_f, dtype=np.float32)
        style_np = np.array(style_f, dtype=np.float32)

        content_np = SeedVR2Util._wavelet_reconstruction(content_np, style_np)

        c = np.transpose(content_np, (0, 2, 3, 1))
        s = np.transpose(style_np, (0, 2, 3, 1))
        c = np.clip((c + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
        s = np.clip((s + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)

        c_lab = SeedVR2Util._rgb_to_lab(c)
        s_lab = SeedVR2Util._rgb_to_lab(s)

        matched_a = SeedVR2Util._hist_match(c_lab[..., 1], s_lab[..., 1])
        matched_b = SeedVR2Util._hist_match(c_lab[..., 2], s_lab[..., 2])

        if luminance_weight < 1.0:
            matched_L = SeedVR2Util._hist_match(c_lab[..., 0], s_lab[..., 0])
            L = luminance_weight * c_lab[..., 0] + (1.0 - luminance_weight) * matched_L
        else:
            L = c_lab[..., 0]

        out_lab = np.stack([L, matched_a, matched_b], axis=-1)
        out_rgb = SeedVR2Util._lab_to_rgb(out_lab)
        out_rgb = np.clip(out_rgb, 0.0, 1.0)

        out = out_rgb * 2.0 - 1.0
        out = mx.array(out, dtype=mx.float32)
        out = mx.transpose(out, (0, 3, 1, 2))
        return out.astype(content.dtype)

    @staticmethod
    def _wavelet_blur(image: np.ndarray, radius: int) -> np.ndarray:
        if radius < 1:
            radius = 1

        h, w = int(image.shape[-2]), int(image.shape[-1])
        max_safe_radius = max(1, min(h, w) // 8)
        if radius > max_safe_radius:
            radius = max_safe_radius

        kernel = np.array(
            [
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625],
            ],
            dtype=np.float32,
        )

        p = radius
        padded = np.pad(image, ((0, 0), (0, 0), (p, p), (p, p)), mode="edge")

        out = np.zeros_like(image, dtype=np.float32)
        H, W = image.shape[-2], image.shape[-1]

        for ky, dy in enumerate((-1, 0, 1)):
            ys = p + dy * radius
            ye = ys + H
            for kx, dx in enumerate((-1, 0, 1)):
                xs = p + dx * radius
                xe = xs + W
                out += kernel[ky, kx] * padded[:, :, ys:ye, xs:xe]

        return out

    @staticmethod
    def _wavelet_decomposition(image: np.ndarray, levels: int = 5) -> tuple[np.ndarray, np.ndarray]:
        high_freq = np.zeros_like(image, dtype=np.float32)
        cur = image.astype(np.float32)

        for i in range(levels):
            radius = 2**i
            low_freq = SeedVR2Util._wavelet_blur(cur, radius)
            high_freq += cur - low_freq
            cur = low_freq

        return high_freq, cur

    @staticmethod
    def _wavelet_reconstruction(content: np.ndarray, style: np.ndarray) -> np.ndarray:
        if content.shape != style.shape:
            raise ValueError(f"Wavelet reconstruction requires same shapes, got {content.shape} vs {style.shape}")

        content_high, _ = SeedVR2Util._wavelet_decomposition(content, levels=5)
        _, style_low = SeedVR2Util._wavelet_decomposition(style, levels=5)
        return np.clip(content_high + style_low, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)

    @staticmethod
    def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0.0031308, 1.055 * np.maximum(x, 0.0) ** (1.0 / 2.4) - 0.055, 12.92 * x)

    @staticmethod
    def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
        rgb_lin = SeedVR2Util._srgb_to_linear(rgb.astype(np.float32))
        M = np.array(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ],
            dtype=np.float32,
        )
        xyz = np.tensordot(rgb_lin, M.T, axes=([3], [0]))

        xyz[..., 0] /= 0.95047
        xyz[..., 2] /= 1.08883

        eps = 6.0 / 29.0
        eps3 = eps**3
        kappa = (29.0 / 3.0) ** 3

        f = np.where(xyz > eps3, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
        fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]

        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        return np.stack([L, a, b], axis=-1).astype(np.float32)

    @staticmethod
    def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
        L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0

        eps = 6.0 / 29.0
        kappa = (29.0 / 3.0) ** 3

        x = np.where(fx > eps, fx**3, (116.0 * fx - 16.0) / kappa)
        y = np.where(fy > eps, fy**3, (116.0 * fy - 16.0) / kappa)
        z = np.where(fz > eps, fz**3, (116.0 * fz - 16.0) / kappa)

        x *= 0.95047
        z *= 1.08883

        xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
        M_inv = np.array(
            [
                [3.2404542, -1.5371385, -0.4985314],
                [-0.9692660, 1.8760108, 0.0415560],
                [0.0556434, -0.2040259, 1.0572252],
            ],
            dtype=np.float32,
        )
        rgb_lin = np.tensordot(xyz, M_inv.T, axes=([3], [0]))
        rgb = SeedVR2Util._linear_to_srgb(rgb_lin)
        return rgb.astype(np.float32)

    @staticmethod
    def _hist_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        out = np.empty_like(source, dtype=np.float32)
        B = source.shape[0]
        for i in range(B):
            src = source[i].reshape(-1).astype(np.float32)
            ref = reference[i].reshape(-1).astype(np.float32)
            src_idx = np.argsort(src, kind="stable")
            ref_sorted = np.sort(ref, kind="stable")
            inv = np.empty_like(src_idx)
            inv[src_idx] = np.arange(src_idx.size, dtype=src_idx.dtype)
            out[i] = ref_sorted[inv].reshape(source.shape[1:]).astype(np.float32)
        return out
