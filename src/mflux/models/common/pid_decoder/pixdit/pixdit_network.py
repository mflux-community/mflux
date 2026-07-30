import mlx.core as mx
from mlx import nn

from mflux.models.common.pid_decoder.pixdit.pixdit_embedders import PatchTokenEmbedder, PixelTokenEmbedder
from mflux.models.common.pid_decoder.pixdit.pixdit_final_layer import FinalLayer
from mflux.models.common.pid_decoder.pixdit.pixdit_gates import build_gate
from mflux.models.common.pid_decoder.pixdit.pixdit_lq_projection import LQProjection2D
from mflux.models.common.pid_decoder.pixdit.pixdit_mmdit_block import MMDiTBlockT2I
from mflux.models.common.pid_decoder.pixdit.pixdit_pit_block import PiTBlock
from mflux.models.common.pid_decoder.pixdit.pixdit_rms_norm import PixDiTRMSNorm
from mflux.models.common.pid_decoder.pixdit.pixdit_rope import precompute_freqs_cis_1d_text, precompute_freqs_cis_2d_ntk
from mflux.models.common.pid_decoder.pixdit.pixdit_timestep_conditioner import TimestepConditioner

# Source: pixeldit_official.py:1123-1329 (PixDiT_T2I.__init__/forward/fetch_pos*)
# and pid_net.py (PidNet, the SR/LQ-conditioned subclass). ED path and CP are
# omitted entirely (Global Constraints; confirmed safe for this checkpoint --
# enable_ed=False, no multi-GPU context-parallel needed for single-device MLX
# inference).


class PidNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_groups: int = 24,
        hidden_size: int = 1536,
        pixel_hidden_size: int = 16,
        pixel_attn_hidden_size: int | None = 1152,
        pixel_num_groups: int | None = None,
        patch_depth: int = 14,
        pixel_depth: int = 2,
        patch_size: int = 16,
        txt_embed_dim: int = 2304,
        txt_max_length: int = 300,
        use_text_rope: bool = True,
        text_rope_theta: float = 10000.0,
        rope_ref_h: int = 1024,
        rope_ref_w: int = 1024,
        lq_latent_channels: int = 16,
        lq_hidden_dim: int = 512,
        lq_num_res_blocks: int = 4,
        lq_gate_type: str = "sigma_aware_per_token",
        lq_interval: int = 2,
        lq_conv_padding_mode: str = "zeros",
        pit_lq_inject: bool = False,
        sr_scale: int = 4,
        latent_spatial_down_factor: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.patch_depth = patch_depth
        self.txt_max_length = txt_max_length
        self.out_channels = in_channels
        self.use_text_rope = use_text_rope
        self.text_rope_theta = text_rope_theta
        # NTK-aware RoPE reference resolution, in *pixels*, divided down to patch units exactly
        # as the reference does (pixeldit_official.py:1170 `self.rope_ref_grid_h = rope_ref_h //
        # self.patch_size`). Keep the argument in pixels on this side too: the released
        # checkpoint's config states 2048, and a port that stores patches instead has to
        # remember to divide -- which is how this landed on 64 (the 1024px PixDiT pretraining
        # default) instead of the 2048/16 = 128 the v1pt5 SR checkpoints were trained with.
        self.rope_ref_grid_h = rope_ref_h // patch_size
        self.rope_ref_grid_w = rope_ref_w // patch_size

        self.pixel_embedder = PixelTokenEmbedder(in_channels, pixel_hidden_size)
        self.s_embedder = PatchTokenEmbedder(in_channels * patch_size**2, hidden_size)
        self.t_embedder = TimestepConditioner(hidden_size)
        self.y_embedder = PatchTokenEmbedder(txt_embed_dim, hidden_size, norm=PixDiTRMSNorm(hidden_size))
        self.y_pos_embedding = mx.zeros((1, txt_max_length, hidden_size))

        self.patch_blocks = [MMDiTBlockT2I(hidden_size, num_groups) for _ in range(patch_depth)]

        pixel_attn_hidden_size = pixel_attn_hidden_size or hidden_size
        pixel_num_groups = pixel_num_groups or num_groups
        self.pixel_blocks = [
            PiTBlock(
                pixel_hidden_size,
                hidden_size,
                patch_size,
                attn_hidden_size=pixel_attn_hidden_size,
                attn_num_heads=pixel_num_groups,
                rope_ref_grid_h=self.rope_ref_grid_h,
                rope_ref_grid_w=self.rope_ref_grid_w,
            )
            for _ in range(pixel_depth)
        ]

        self.final_layer = FinalLayer(pixel_hidden_size, self.out_channels)

        num_lq_outputs = patch_depth // lq_interval + (1 if patch_depth % lq_interval else 0)
        self.num_lq_outputs = num_lq_outputs
        self.pit_lq_inject = pit_lq_inject
        self.lq_proj = LQProjection2D(
            latent_channels=lq_latent_channels,
            hidden_dim=lq_hidden_dim,
            out_dim=hidden_size,
            patch_size=patch_size,
            sr_scale=sr_scale,
            latent_spatial_down_factor=latent_spatial_down_factor,
            num_res_blocks=lq_num_res_blocks,
            num_outputs=num_lq_outputs,
            gate_type=lq_gate_type,
            interval=lq_interval,
            conv_padding_mode=lq_conv_padding_mode,
            pit_output=pit_lq_inject,
        )
        self.pit_lq_gate = build_gate(lq_gate_type, hidden_size) if pit_lq_inject else None

    def __call__(self, x: mx.array, t: mx.array, y: mx.array, lq_latent: mx.array, degrade_sigma: mx.array) -> mx.array:
        B, _, H, W = x.shape
        Hs, Ws = H // self.patch_size, W // self.patch_size
        L = Hs * Ws

        lq_features = self.lq_proj(lq_latent, target_pH=Hs, target_pW=Ws)
        pit_lq_feature = None
        if self.pit_lq_inject:
            pit_lq_feature = lq_features[self.num_lq_outputs]
            lq_features = lq_features[: self.num_lq_outputs]

        # Patchify: [B, 3, H, W] -> [B, L, patch_size^2 * 3] to match PyTorch's
        # `F.unfold(x, patch_size, stride=patch_size).transpose(1, 2)` channel-major
        # patch layout (channel varies fastest within a patch's flattened vector).
        x_patches = x.reshape(B, 3, Hs, self.patch_size, Ws, self.patch_size)
        x_patches = x_patches.transpose(0, 2, 4, 1, 3, 5).reshape(B, L, 3 * self.patch_size * self.patch_size)

        t_emb = self.t_embedder(t).reshape(B, 1, self.hidden_size)
        condition = nn.silu(t_emb)

        Ltxt = min(y.shape[1], self.txt_max_length)
        y = y[:, :Ltxt, :]
        y_emb = self.y_embedder(y) + self.y_pos_embedding[:, :Ltxt, :]

        pos_img = precompute_freqs_cis_2d_ntk(
            self.hidden_size // self.num_groups, Hs, Ws, self.rope_ref_grid_h, self.rope_ref_grid_w
        )
        pos_txt = None
        if self.use_text_rope:
            pos_txt = precompute_freqs_cis_1d_text(self.hidden_size // self.num_groups, Ltxt, self.text_rope_theta)

        s_main = self.s_embedder(x_patches)
        for i, block in enumerate(self.patch_blocks):
            if self.lq_proj.is_gate_active(i):
                out_idx = self.lq_proj.output_index(i)
                if out_idx < len(lq_features):
                    s_main = self.lq_proj.gate(s_main, lq_features[out_idx], degrade_sigma, out_idx)
            s_main, y_emb = block(s_main, y_emb, condition, pos_img, pos_txt=pos_txt)

        s = nn.silu(t_emb + s_main)
        s_cond_tokens = s
        if self.pit_lq_inject and pit_lq_feature is not None:
            s_cond_tokens = self.pit_lq_gate(s_cond_tokens, pit_lq_feature, sigma=degrade_sigma)
        s_cond = s_cond_tokens.reshape(B * L, self.hidden_size)

        x_pixels = self.pixel_embedder(x, img_height=H, img_width=W, patch_size=self.patch_size)
        for block in self.pixel_blocks:
            x_pixels = block(x_pixels, s_cond, H, W, self.patch_size)

        x_pixels = self.final_layer(x_pixels)  # [B*L, P2, C_out]
        x_pixels = x_pixels.reshape(B, Hs, Ws, self.patch_size, self.patch_size, self.out_channels)
        x_pixels = x_pixels.transpose(0, 1, 3, 2, 4, 5)  # [B, Hs, ps, Ws, ps, C]
        return x_pixels.reshape(B, H, W, self.out_channels).transpose(0, 3, 1, 2)  # fold -> [B, C, H, W]
