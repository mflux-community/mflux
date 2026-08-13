import mlx.core as mx
from mlx import nn

from mflux.models.krea2.model.krea2_transformer.final_layer import LastLayer
from mflux.models.krea2.model.krea2_transformer.rope_embedder import Krea2RopeEmbedder
from mflux.models.krea2.model.krea2_transformer.text_fusion import TextFusionTransformer
from mflux.models.krea2.model.krea2_transformer.text_mlp import Krea2TextMLP
from mflux.models.krea2.model.krea2_transformer.timestep_embedder import Krea2TimestepMLP, Krea2TimestepProj
from mflux.models.krea2.model.krea2_transformer.transformer_block import SingleStreamBlock


class Krea2Transformer(nn.Module):
    def __init__(
        self,
        features: int = 6144,
        tdim: int = 256,
        txtdim: int = 2560,
        heads: int = 48,
        kvheads: int = 12,
        multiplier: int = 4,
        layers: int = 28,
        patch: int = 2,
        channels: int = 16,
        bias: bool = False,
        theta: int = 1000,
        txtlayers: int = 12,
        txtheads: int = 20,
        txtkvheads: int = 20,
    ):
        super().__init__()
        self.patch = patch
        self.channels = channels
        self.tdim = tdim
        self.heads = heads
        self.txtdim = txtdim
        self.txtlayers = txtlayers

        head_dim = features // heads
        axes = [head_dim - 12 * (head_dim // 16), 6 * (head_dim // 16), 6 * (head_dim // 16)]
        self.pe_embedder = Krea2RopeEmbedder(head_dim=head_dim, theta=theta, axes_dim=axes)

        self.first = nn.Linear(channels * patch**2, features, bias=True)
        self.blocks = [SingleStreamBlock(features, heads, multiplier, bias, kvheads) for _ in range(layers)]
        self.tmlp = Krea2TimestepMLP(tdim, features)
        self.tproj = Krea2TimestepProj(features)
        self.txtfusion = TextFusionTransformer(txtlayers, txtdim, txtheads, multiplier, bias, txtkvheads)
        self.txtmlp = Krea2TextMLP(txtdim, features)
        self.last = LastLayer(features, patch, channels)

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        context: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        bs, c, H_orig, W_orig = hidden_states.shape
        patch = self.patch

        x = Krea2Transformer._pad_to_multiple(hidden_states, patch)
        H, W = x.shape[-2], x.shape[-1]
        h_, w_ = H // patch, W // patch

        context = self._unpack_context(context)

        # Patchify: (b, c, h*ph, w*pw) -> (b, h*w, c*ph*pw)
        img = x.reshape(bs, c, h_, patch, w_, patch).transpose(0, 2, 4, 1, 3, 5).reshape(bs, h_ * w_, c * patch * patch)
        img = self.first(img)

        t = self.tmlp(Krea2TimestepMLP.timestep_embedding(timestep, self.tdim)[:, None, :].astype(img.dtype))
        tvec = self.tproj(t)

        context = self.txtfusion(context, mask=None)
        context = self.txtmlp(context)

        txtlen, imglen = context.shape[1], img.shape[1]
        combined = mx.concatenate([context, img], axis=1)

        # Position ids: text at (0,0,0); image at (0, h_idx, w_idx).
        txtpos = mx.zeros((bs, txtlen, 3), dtype=mx.float32)
        gh, gw = mx.meshgrid(mx.arange(h_, dtype=mx.float32), mx.arange(w_, dtype=mx.float32), indexing="ij")
        imgids = mx.stack([mx.zeros_like(gh), gh, gw], axis=-1).reshape(1, h_ * w_, 3)
        imgpos = mx.broadcast_to(imgids, (bs, h_ * w_, 3))
        pos = mx.concatenate([txtpos, imgpos], axis=1)
        freqs = self.pe_embedder(pos)

        # Optional gradient checkpointing: recompute each block in backward instead of storing its
        # activations, trading compute for a large drop in peak memory during training. Off by
        # default, so inference is unaffected; the training adapter turns it on.
        gradient_checkpointing = getattr(self, "gradient_checkpointing", False)
        for block in self.blocks:
            run = nn.utils.checkpoint(block) if gradient_checkpointing else block
            combined = run(combined, tvec, freqs, attention_mask)

        final = self.last(combined, t)
        out = final[:, txtlen : txtlen + imglen, :]
        # Unpatchify: (b, h*w, c*ph*pw) -> (b, c, h*ph, w*pw)
        out = out.reshape(bs, h_, w_, self.channels, patch, patch).transpose(0, 3, 1, 4, 2, 5)
        out = out.reshape(bs, self.channels, H, W)
        return out[:, :, :H_orig, :W_orig]

    def _unpack_context(self, context: mx.array) -> mx.array:
        # (B, seq, txtlayers*txtdim) -> (B, seq, txtlayers, txtdim)
        b, seq, fused = context.shape
        if fused != self.txtlayers * self.txtdim:
            raise ValueError(
                f"Krea2 expects conditioning with {self.txtlayers}x{self.txtdim}="
                f"{self.txtlayers * self.txtdim} features (a {self.txtlayers}-layer Qwen3-VL stack) but got {fused}."
            )
        return context.reshape(b, seq, self.txtlayers, self.txtdim)

    @staticmethod
    def _pad_to_multiple(x: mx.array, patch: int) -> mx.array:
        H, W = x.shape[-2], x.shape[-1]
        pad_h = (patch - H % patch) % patch
        pad_w = (patch - W % patch) % patch
        if pad_h == 0 and pad_w == 0:
            return x
        return mx.pad(x, [(0, 0), (0, 0), (0, pad_h), (0, pad_w)])
