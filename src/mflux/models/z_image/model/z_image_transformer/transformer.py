import math

import mlx.core as mx
from mlx import nn

from mflux.models.z_image.model.z_image_transformer.context_block import ZImageContextBlock
from mflux.models.z_image.model.z_image_transformer.final_layer import FinalLayer
from mflux.models.z_image.model.z_image_transformer.rope_embedder import RopeEmbedder
from mflux.models.z_image.model.z_image_transformer.timestep_embedder import TimestepEmbedder
from mflux.models.z_image.model.z_image_transformer.transformer_block import ZImageTransformerBlock


class ZImageTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int = 2,
        f_patch_size: int = 1,
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        cap_feat_dim: int = 2560,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: list[int] | None = None,
        axes_lens: list[int] | None = None,
    ):
        super().__init__()

        axes_dims = axes_dims or [32, 48, 48]
        axes_lens = axes_lens or [1024, 512, 512]

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.f_patch_size = f_patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.t_scale = t_scale

        key = f"{patch_size}-{f_patch_size}"
        embed_dim = f_patch_size * patch_size * patch_size * in_channels
        self.all_x_embedder = {key: nn.Linear(embed_dim, dim, bias=True)}
        self.all_final_layer = {key: FinalLayer(dim, embed_dim)}

        self.t_embedder = TimestepEmbedder(out_size=min(dim, 256), mid_size=1024)
        self.cap_embedder = [nn.RMSNorm(cap_feat_dim, eps=norm_eps), nn.Linear(cap_feat_dim, dim, bias=True)]
        self.x_pad_token = mx.zeros((1, dim))
        self.cap_pad_token = mx.zeros((1, dim))

        self.noise_refiner = [ZImageTransformerBlock(dim, n_heads, norm_eps, qk_norm) for _ in range(n_refiner_layers)]  # fmt: off
        self.context_refiner = [ZImageContextBlock(dim, n_heads, norm_eps, qk_norm) for _ in range(n_refiner_layers)]  # fmt: off
        self.layers = [ZImageTransformerBlock(dim, n_heads, norm_eps, qk_norm) for _ in range(n_layers)]  # fmt: off
        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    def __call__(
        self,
        x: mx.array,
        timestep: mx.array | float | int,
        sigmas: mx.array,
        cap_feats: mx.array,
        controlnet_block_samples: dict[int, mx.array] | list[mx.array] | None = None,
    ) -> mx.array:
        key = f"{self.patch_size}-{self.f_patch_size}"

        # Time embedding
        if not isinstance(timestep, mx.array):
            if isinstance(timestep, int):
                sigma_t = sigmas[timestep].reshape((1,))
                timestep = mx.ones_like(sigma_t) - sigma_t
            else:
                timestep = mx.array(timestep, dtype=mx.float32)
        if timestep.ndim == 0:
            timestep = timestep.reshape((1,))
        t_emb = self.t_embedder(timestep.astype(mx.float32) * self.t_scale)

        # Patchify image and caption
        x_emb, cap_emb, x_size, x_pos_ids, cap_pos_ids, x_pad_mask, cap_pad_mask = ZImageTransformer._patchify(
            image=x,
            cap_feats=cap_feats,
            patch_size=self.patch_size,
            f_patch_size=self.f_patch_size,
        )

        # Image embedding
        x_emb = self.all_x_embedder[key](x_emb)
        x_emb = mx.where(x_pad_mask[:, None], self.x_pad_token, x_emb)
        x_freqs_cis = self.rope_embedder(x_pos_ids)
        x_attn_mask = mx.ones((1, x_emb.shape[0]), dtype=mx.bool_)
        x_emb = mx.expand_dims(x_emb, axis=0)

        # Noise refiner
        for layer in self.noise_refiner:
            x_emb = layer(
                x=x_emb,
                attn_mask=x_attn_mask,
                freqs_cis=x_freqs_cis,
                t_emb=t_emb,
            )

        # Caption embedding
        cap_emb = self.cap_embedder[1](self.cap_embedder[0](cap_emb))
        cap_emb = mx.where(cap_pad_mask[:, None], self.cap_pad_token, cap_emb)
        cap_freqs_cis = self.rope_embedder(cap_pos_ids)
        cap_attn_mask = mx.ones((1, cap_emb.shape[0]), dtype=mx.bool_)
        cap_emb = mx.expand_dims(cap_emb, axis=0)

        # Context refiner
        for layer in self.context_refiner:
            cap_emb = layer(
                x=cap_emb,
                attn_mask=cap_attn_mask,
                freqs_cis=cap_freqs_cis,
            )

        # Unify and main layers
        x_len = x_emb.shape[1]
        unified = mx.concatenate([x_emb, cap_emb], axis=1)
        unified_freqs_cis = mx.concatenate([x_freqs_cis, cap_freqs_cis], axis=0)
        unified_attn_mask = mx.ones((1, unified.shape[1]), dtype=mx.bool_)

        for layer_idx, layer in enumerate(self.layers):
            unified = layer(
                x=unified,
                attn_mask=unified_attn_mask,
                freqs_cis=unified_freqs_cis,
                t_emb=t_emb,
            )
            if controlnet_block_samples is not None:
                if isinstance(controlnet_block_samples, dict):
                    if layer_idx in controlnet_block_samples:
                        unified = unified + controlnet_block_samples[layer_idx]
                else:
                    # Interval mapping fallback (like FLUX)
                    sample = ZImageTransformer._get_controlnet_sample(layer_idx, self.layers, controlnet_block_samples)
                    if sample is not None:
                        unified = unified + sample

        # Final layer and unpatchify
        unified = self.all_final_layer[key](unified, t_emb)
        output = ZImageTransformer._unpatchify(
            x=unified[0, :x_len],
            size=x_size,
            patch_size=self.patch_size,
            f_patch_size=self.f_patch_size,
            out_channels=self.out_channels,
        )
        return -output

    @staticmethod
    def _get_controlnet_sample(idx: int, blocks: list, controlnet_samples: list[mx.array] | None) -> mx.array | None:
        if controlnet_samples is None:
            return None
        if len(controlnet_samples) == 0:
            return None
        num_blocks = len(blocks)
        num_samples = len(controlnet_samples)
        interval_control = int(math.ceil(num_blocks / num_samples))
        control_index = idx // interval_control
        if control_index >= num_samples:
            control_index = num_samples - 1
        return controlnet_samples[control_index]

    @staticmethod
    def _patchify(
        image: mx.array,
        cap_feats: mx.array,
        patch_size: int,
        f_patch_size: int,
    ) -> tuple[mx.array, mx.array, tuple[int, int, int], mx.array, mx.array, mx.array, mx.array]:
        pH = pW = patch_size
        pF = f_patch_size

        # Caption padding
        cap_ori_len = cap_feats.shape[0]
        cap_padding_len = (-cap_ori_len) % 32
        cap_pos_ids = ZImageTransformer._create_coord_grid((cap_ori_len + cap_padding_len, 1, 1), (1, 0, 0)).reshape(-1, 3)  # fmt: off
        cap_pad_mask = mx.concatenate([mx.zeros((cap_ori_len,), dtype=mx.bool_), mx.ones((cap_padding_len,), dtype=mx.bool_)])  # fmt: off
        cap_padded = mx.concatenate([cap_feats, mx.repeat(cap_feats[-1:], cap_padding_len, axis=0)], axis=0) if cap_padding_len > 0 else cap_feats  # fmt: off

        # Image patchification
        C, F, H, W = image.shape
        image_size = (F, H, W)
        F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

        image = image.reshape(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        image = mx.transpose(image, axes=(1, 3, 5, 2, 4, 6, 0))
        image = image.reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

        # Image padding
        image_ori_len = image.shape[0]
        image_padding_len = (-image_ori_len) % 32
        image_pos_ids = ZImageTransformer._create_coord_grid((F_tokens, H_tokens, W_tokens), (cap_ori_len + cap_padding_len + 1, 0, 0)).reshape(-1, 3)  # fmt: off

        if image_padding_len > 0:
            image_pos_ids = mx.concatenate([image_pos_ids, mx.zeros((image_padding_len, 3), dtype=mx.int32)], axis=0)
            image = mx.concatenate([image, mx.repeat(image[-1:], image_padding_len, axis=0)], axis=0)

        image_pad_mask = mx.concatenate([mx.zeros((image_ori_len,), dtype=mx.bool_), mx.ones((image_padding_len,), dtype=mx.bool_)])  # fmt: off

        return image, cap_padded, image_size, image_pos_ids, cap_pos_ids, image_pad_mask, cap_pad_mask

    @staticmethod
    def _unpatchify(
        x: mx.array,
        size: tuple[int, int, int],
        patch_size: int,
        f_patch_size: int,
        out_channels: int,
    ) -> mx.array:
        pH = pW = patch_size
        pF = f_patch_size
        F, H, W = size
        ori_len = (F // pF) * (H // pH) * (W // pW)
        x = x[:ori_len].reshape(F // pF, H // pH, W // pW, pF, pH, pW, out_channels)
        x = mx.transpose(x, axes=(6, 0, 3, 1, 4, 2, 5))
        return x.reshape(out_channels, F, H, W)

    @staticmethod
    def _create_coord_grid(size: tuple[int, ...], start: tuple[int, ...] | None = None) -> mx.array:
        start = start or tuple(0 for _ in size)
        axes = [mx.arange(x0, x0 + span, dtype=mx.int32) for x0, span in zip(start, size)]
        grids = mx.meshgrid(*axes, indexing="ij")
        return mx.stack(grids, axis=-1)
