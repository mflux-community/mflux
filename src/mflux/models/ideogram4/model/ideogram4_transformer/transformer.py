import mlx.core as mx
from mlx import nn

from mflux.models.common.config import ModelConfig
from mflux.models.ideogram4.model.ideogram4_transformer.config import Ideogram4Config
from mflux.models.ideogram4.model.ideogram4_transformer.fp8_linear import Fp8Linear
from mflux.models.ideogram4.model.ideogram4_transformer.modulation import (
    Ideogram4EmbedScalar,
    Ideogram4FinalLayer,
    Ideogram4RMSNorm,
)
from mflux.models.ideogram4.model.ideogram4_transformer.rope_embedder import Ideogram4MRoPE
from mflux.models.ideogram4.model.ideogram4_transformer.transformer_block import Ideogram4TransformerBlock


class Ideogram4Transformer(nn.Module):
    def __init__(self, config: Ideogram4Config | None = None) -> None:
        super().__init__()
        self.config = config or Ideogram4Config()
        head_dim = self.config.emb_dim // self.config.num_heads
        self.input_proj = Fp8Linear(self.config.in_channels, self.config.emb_dim, bias=True)
        self.llm_cond_norm = Ideogram4RMSNorm(self.config.llm_features_dim, eps=1e-6)
        self.llm_cond_proj = Fp8Linear(self.config.llm_features_dim, self.config.emb_dim, bias=True)
        self.t_embedding = Ideogram4EmbedScalar(self.config.emb_dim, input_range=(0.0, 1.0))
        self.adaln_proj = Fp8Linear(self.config.emb_dim, self.config.adanln_dim)
        self.embed_image_indicator = nn.Embedding(2, self.config.emb_dim)
        self.rotary_emb = Ideogram4MRoPE(
            head_dim=head_dim,
            base=self.config.rope_theta,
            mrope_section=self.config.mrope_section,
        )
        self.layers = [
            Ideogram4TransformerBlock(
                hidden_size=self.config.emb_dim,
                intermediate_size=self.config.intermediate_size,
                num_heads=self.config.num_heads,
                norm_eps=self.config.norm_eps,
                adanln_dim=self.config.adanln_dim,
            )
            for _ in range(self.config.num_layers)
        ]
        self.final_layer = Ideogram4FinalLayer(
            hidden_size=self.config.emb_dim,
            out_channels=self.config.in_channels,
            adanln_dim=self.config.adanln_dim,
        )

    def __call__(
        self,
        *,
        llm_features: mx.array,
        x: mx.array,
        t: mx.array,
        position_ids: mx.array,
        segment_ids: mx.array,
        indicator: mx.array,
    ) -> mx.array:
        _, _, in_channels = x.shape
        if in_channels != self.config.in_channels:
            raise ValueError(f"x has {in_channels} channels, expected {self.config.in_channels}")
        x = x.astype(ModelConfig.precision)
        t = t.astype(ModelConfig.precision)
        llm_features = llm_features.astype(ModelConfig.precision)

        llm_token_mask = (indicator == 3).astype(x.dtype)[..., None]
        output_image_mask = (indicator == 2).astype(x.dtype)[..., None]

        llm_features = llm_features * llm_token_mask
        x = x * output_image_mask
        x = self.input_proj(x) * output_image_mask

        t_cond = self.t_embedding(t)
        if t.ndim == 1:
            t_cond = t_cond[:, None, :]
        adaln_input = nn.silu(self.adaln_proj(t_cond))

        llm_features = self.llm_cond_norm(llm_features)
        llm_features = self.llm_cond_proj(llm_features) * llm_token_mask
        h = x + llm_features
        h = h + self.embed_image_indicator((indicator == 2).astype(mx.int32))

        cos, sin = self.rotary_emb(position_ids)
        cos = cos.astype(h.dtype)
        sin = sin.astype(h.dtype)
        for layer in self.layers:
            h = layer(
                h,
                segment_ids=segment_ids,
                cos=cos,
                sin=sin,
                adaln_input=adaln_input,
            )
        return self.final_layer(h, c=adaln_input).astype(mx.float32)
