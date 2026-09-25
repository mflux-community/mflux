import mlx.core as mx
from mlx import nn

from mflux.models.ming_image.model.ming_text_encoder.ling_moe_encoder import HIDDEN_SIZE, NUM_LAYERS, LingMoeEncoder
from mflux.models.ming_image.model.ming_text_encoder.ming_connector import (
    HIDDEN_SIZE as CONNECTOR_SIZE,
    MingConnector,
)

IMAGE_START_ID = 157158
IMAGE_PATCH_ID = 157157
IMAGE_END_ID = 157159
QUERY_GRID = 16  # img_gen_scales = [16] -> 256 learned query tokens
DIRECTVLM_LAYERS = (5, 12, 20)
CAP_FEAT_DIM = 2560
DIT_DIM = 3840

# Rendered by the checkpoint's chat template for a single HUMAN turn with generation prompt.
PROMPT_TEMPLATE = (
    "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off<|role_end|>"
    "<role>HUMAN</role>{prompt}<|role_end|><role>ASSISTANT</role>"
)


class MingHeads(nn.Module):
    """The mlp/ folder: learned query tokens plus the projections around the connector and the
    direct-VLM head that turns three LLM hidden states into DiT-width caption tokens."""

    def __init__(self):
        super().__init__()
        self.query_tokens = mx.zeros((QUERY_GRID * QUERY_GRID, HIDDEN_SIZE))
        self.proj_in = nn.Linear(HIDDEN_SIZE, CONNECTOR_SIZE, bias=True)
        self.proj_out = nn.Linear(CONNECTOR_SIZE, CAP_FEAT_DIM, bias=True)
        directvlm_dim = HIDDEN_SIZE * len(DIRECTVLM_LAYERS)
        self.proj_directvlm = [nn.RMSNorm(directvlm_dim, eps=1e-5), nn.Linear(directvlm_dim, DIT_DIM, bias=True)]


class MingConditionEncoder:
    @staticmethod
    def build_inputs(prompt_ids: list[int]) -> tuple[mx.array, mx.array, mx.array]:
        """Append the query-token block and build the 3D video_rope positions that
        get_t_scale_rope_index assigns: text counts up on all three axes, the 1x1x256 query grid
        shares t = h = n_text with w centred on it, and the closing token resumes at t + 1."""
        n_q = QUERY_GRID * QUERY_GRID
        ids = prompt_ids + [IMAGE_START_ID] + [IMAGE_PATCH_ID] * n_q + [IMAGE_END_ID]
        n_text = len(prompt_ids) + 1
        text = mx.broadcast_to(mx.arange(n_text)[None], (3, n_text))
        base = mx.full((n_q,), n_text)
        w = mx.arange(n_q) - (n_q - 1) // 2 + n_text
        query = mx.stack([base, base, w])
        end = mx.full((3, 1), n_text + 1)
        position_ids = mx.concatenate([text, query, end], axis=1).astype(mx.int32)
        input_ids = mx.array(ids, dtype=mx.int32)
        image_mask = (input_ids == IMAGE_PATCH_ID)[None]
        return input_ids, position_ids, image_mask

    @staticmethod
    def encode(
        prompt_ids: list[int],
        text_encoder: LingMoeEncoder,
        connector: MingConnector,
        heads: MingHeads,
    ) -> tuple[mx.array, mx.array]:
        """Returns (cap_feats (256, 2560), cap_feats_2 (n_prompt, 3840))."""
        input_ids, position_ids, image_mask = MingConditionEncoder.build_inputs(prompt_ids)
        embeds = text_encoder.word_embeddings(input_ids)
        q_start = len(prompt_ids) + 1
        n_q = heads.query_tokens.shape[0]
        embeds = mx.concatenate(
            [embeds[:q_start], heads.query_tokens.astype(embeds.dtype), embeds[q_start + n_q :]], axis=0
        )[None]
        hidden = text_encoder(embeds, position_ids, image_mask, output_layers=(*DIRECTVLM_LAYERS, NUM_LAYERS))

        n_prompt = len(prompt_ids)
        directvlm = mx.concatenate([hidden[i][0, :n_prompt] for i in DIRECTVLM_LAYERS], axis=-1)
        cap_feats_2 = heads.proj_directvlm[1](heads.proj_directvlm[0](directvlm))

        query_states = hidden[NUM_LAYERS][:, q_start : q_start + n_q]
        cap_feats = heads.proj_out(connector(heads.proj_in(query_states)))[0]
        return cap_feats, cap_feats_2
