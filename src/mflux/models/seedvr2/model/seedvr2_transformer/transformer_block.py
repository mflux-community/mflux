import mlx.core as mx
from mlx import nn

from mflux.models.seedvr2.model.seedvr2_transformer.ada_modulation import AdaModulation
from mflux.models.seedvr2.model.seedvr2_transformer.attention import MMAttention
from mflux.models.seedvr2.model.seedvr2_transformer.mm_swiglu import MMSwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        vid_dim: int = 2560,
        txt_dim: int = 2560,
        heads: int = 20,
        head_dim: int = 128,
        expand_ratio: int = 4,
        mlp_type: str = "swiglu",
        norm_eps: float = 1e-5,
        qk_bias: bool = False,
        rope_dim: int = 128,
        rope_freqs_for: str = "lang",
        rope_on_text: bool = True,
        shared_weights: bool = False,
        is_last_layer: bool = False,
        window: tuple[int, int, int] = (4, 3, 3),
        shift: bool = False,
    ):
        super().__init__()
        self.shared_weights = shared_weights
        self.is_last_layer = is_last_layer
        self.vid_dim = vid_dim
        self.txt_dim = txt_dim
        self.norm_eps = norm_eps

        self.attn = MMAttention(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            heads=heads,
            head_dim=head_dim,
            qk_bias=qk_bias,
            qk_norm_eps=norm_eps,
            rope_dim=rope_dim,
            rope_freqs_for=rope_freqs_for,
            rope_on_text=rope_on_text,
            shared_weights=shared_weights,
            window=window,
            shift=shift,
        )

        self.mlp = MMSwiGLU(
            vid_dim=vid_dim,
            txt_dim=txt_dim,
            expand_ratio=expand_ratio,
            shared_weights=shared_weights,
            is_last_layer=is_last_layer,
            mlp_type=mlp_type,
        )

        self.ada = AdaModulation(
            dim=vid_dim,
            shared_weights=shared_weights,
            is_last_layer=is_last_layer,
        )

    def __call__(
        self,
        vid: mx.array,
        txt: mx.array,
        emb: mx.array,
        vid_shape: mx.array,
        txt_shape: mx.array,
    ) -> tuple[mx.array, mx.array]:
        vid_attn = TransformerBlock._rms_norm(vid, self.norm_eps)
        txt_attn = TransformerBlock._rms_norm(txt, self.norm_eps)

        vid_attn = self.ada.modulate_vid(vid_attn, emb, layer="attn", mode="in")
        txt_attn = self.ada.modulate_txt(txt_attn, emb, layer="attn", mode="in")

        vid_attn, txt_attn = self.attn(vid_attn, txt_attn, vid_shape, txt_shape)

        vid_attn = self.ada.modulate_vid(vid_attn, emb, layer="attn", mode="out")
        txt_attn = self.ada.modulate_txt(txt_attn, emb, layer="attn", mode="out")

        vid = vid + vid_attn
        if not self.is_last_layer:
            txt = txt + txt_attn

        vid_mlp = TransformerBlock._rms_norm(vid, self.norm_eps)
        if self.is_last_layer:
            txt_mlp = txt
        else:
            txt_mlp = TransformerBlock._rms_norm(txt, self.norm_eps)

        vid_mlp = self.ada.modulate_vid(vid_mlp, emb, layer="mlp", mode="in")
        txt_mlp = self.ada.modulate_txt(txt_mlp, emb, layer="mlp", mode="in")

        vid_mlp, txt_mlp = self.mlp(vid_mlp, txt_mlp)

        vid_mlp = self.ada.modulate_vid(vid_mlp, emb, layer="mlp", mode="out")
        txt_mlp = self.ada.modulate_txt(txt_mlp, emb, layer="mlp", mode="out")

        vid = vid + vid_mlp
        if not self.is_last_layer:
            txt = txt + txt_mlp

        return vid, txt

    @staticmethod
    def _rms_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
        return mx.fast.rms_norm(x, mx.ones(x.shape[-1], dtype=x.dtype), eps)
