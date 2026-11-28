import mlx.core as mx
from mlx import nn

from mflux.models.seedvr2.model.seedvr2_transformer.patch_in import PatchIn
from mflux.models.seedvr2.model.seedvr2_transformer.patch_out import PatchOut
from mflux.models.seedvr2.model.seedvr2_transformer.rms_norm import RMSNorm
from mflux.models.seedvr2.model.seedvr2_transformer.time_embedding import TimeEmbedding
from mflux.models.seedvr2.model.seedvr2_transformer.transformer_block import TransformerBlock


class SeedVR2Transformer(nn.Module):
    def __init__(
        self,
        vid_in_channels: int = 33,
        vid_out_channels: int = 16,
        vid_dim: int = 2560,
        txt_in_dim: int = 5120,
        txt_dim: int | None = None,
        emb_dim: int | None = None,
        heads: int = 20,
        head_dim: int = 128,
        expand_ratio: int = 4,
        mlp_type: str = "swiglu",
        rope_on_text: bool = True,
        rope_freqs_for: str = "lang",
        use_output_ada: bool = True,
        last_layer_vid_only: bool = True,
        norm_eps: float = 1e-5,
        patch_size: tuple = (1, 2, 2),
        num_layers: int = 32,
        mm_layers: int = 10,
        rope_dim: int = 128,
        window: tuple[int, int, int] = (4, 3, 3),
    ):
        super().__init__()

        txt_dim = txt_dim if txt_dim is not None else vid_dim
        emb_dim = emb_dim if emb_dim is not None else 6 * vid_dim

        self.vid_dim = vid_dim
        self.txt_dim = txt_dim
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.mm_layers = mm_layers
        self.use_output_ada = use_output_ada
        self.last_layer_vid_only = last_layer_vid_only

        self.vid_in = PatchIn(
            in_channels=vid_in_channels,
            patch_size=patch_size,
            dim=vid_dim,
        )

        self.txt_in = nn.Linear(txt_in_dim, txt_dim)

        self.emb_in = TimeEmbedding(
            sinusoidal_dim=256,
            hidden_dim=max(vid_dim, txt_dim),
            output_dim=emb_dim,
        )

        self.blocks = []
        for i in range(num_layers):
            shared_weights = i >= mm_layers
            is_last_layer = self.last_layer_vid_only and i == num_layers - 1
            shift = i % 2 == 1

            self.blocks.append(
                TransformerBlock(
                    vid_dim=vid_dim,
                    txt_dim=txt_dim,
                    heads=heads,
                    head_dim=head_dim,
                    expand_ratio=expand_ratio,
                    mlp_type=mlp_type,
                    norm_eps=norm_eps,
                    qk_bias=False,
                    rope_dim=rope_dim,
                    rope_freqs_for=rope_freqs_for,
                    rope_on_text=rope_on_text,
                    shared_weights=shared_weights,
                    is_last_layer=is_last_layer,
                    window=window,
                    shift=shift,
                )
            )

        if use_output_ada:
            self.vid_out_norm = RMSNorm(vid_dim, eps=norm_eps)
            self.out_shift = mx.zeros((vid_dim,))
            self.out_scale = mx.ones((vid_dim,))

        self.vid_out = PatchOut(
            out_channels=vid_out_channels,
            patch_size=patch_size,
            dim=vid_dim,
        )

    def __call__(
        self,
        vid: mx.array,
        txt: mx.array,
        timestep: mx.array,
    ) -> mx.array:
        dtype = self.vid_in.proj.bias.dtype
        vid = vid.astype(dtype)
        txt = txt.astype(dtype)
        txt = self.txt_in(txt)
        txt_shape = mx.full((txt.shape[0], 1), txt.shape[1], dtype=mx.int32)
        vid, vid_shape = self.vid_in(vid)
        emb = self.emb_in(timestep)
        emb = emb.reshape(-1, self.vid_dim, 2, 3)

        for block in self.blocks:
            vid, txt = block(
                vid=vid,
                txt=txt,
                emb=emb,
                vid_shape=vid_shape,
                txt_shape=txt_shape,
            )

        if self.use_output_ada:
            vid = self.vid_out_norm(vid)
            vid = self._apply_out_ada(vid, emb=emb)
        vid, vid_shape = self.vid_out(vid, vid_shape)
        return vid

    def _apply_out_ada(self, hidden: mx.array, emb: mx.array) -> mx.array:
        shift_a = emb[:, :, 0, 0][:, None, :]
        scale_a = emb[:, :, 0, 1][:, None, :]
        return hidden * (scale_a + self.out_scale) + (shift_a + self.out_shift)
