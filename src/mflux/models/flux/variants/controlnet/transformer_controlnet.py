import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux.model.flux_transformer.embed_nd import EmbedND
from mflux.models.flux.model.flux_transformer.joint_transformer_block import JointTransformerBlock
from mflux.models.flux.model.flux_transformer.single_transformer_block import SingleTransformerBlock
from mflux.models.flux.model.flux_transformer.time_text_embed import TimeTextEmbed
from mflux.models.flux.model.flux_transformer.transformer import Transformer


class TransformerControlnet(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        num_transformer_blocks: int = 5,
        num_single_transformer_blocks: int = 0,
    ):
        super().__init__()
        self.pos_embed = EmbedND()
        self.x_embedder = nn.Linear(64, 3072)
        self.time_text_embed = TimeTextEmbed(model_config=model_config)
        self.context_embedder = nn.Linear(4096, 3072)
        self.transformer_blocks = [JointTransformerBlock(i) for i in range(num_transformer_blocks)]
        self.single_transformer_blocks = [SingleTransformerBlock(i) for i in range(num_single_transformer_blocks)]

        self.controlnet_x_embedder = nn.Linear(64, 3072).apply(nn.init.constant(0))
        self.controlnet_blocks = [nn.Linear(3072, 3072).apply(nn.init.constant(0)) for _ in range(num_transformer_blocks)]  # fmt: off
        self.controlnet_single_blocks = [nn.Linear(3072, 3072).apply(nn.init.constant(0)) for _ in range(num_single_transformer_blocks)]  # fmt: off

    def __call__(
        self,
        t: int,
        config: Config,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        controlnet_condition: mx.array,
    ) -> tuple[list[mx.array], list[mx.array]]:
        # 1. Create embeddings
        hidden_states = self.x_embedder(hidden_states) + self.controlnet_x_embedder(controlnet_condition)
        encoder_hidden_states = self.context_embedder(prompt_embeds)
        text_embeddings = Transformer.compute_text_embeddings(t, pooled_prompt_embeds, self.time_text_embed, config)
        image_rotary_embeddings = Transformer.compute_rotary_embeddings(prompt_embeds, self.pos_embed, config, None)

        # 2. Run the joint flux_transformer blocks
        controlnet_block_samples = []
        for idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = self._apply_joint_transformer_block(
                idx=idx,
                block=block,
                config=config,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                image_rotary_embeddings=image_rotary_embeddings,
                controlnet_block_samples=controlnet_block_samples,
            )

        # 3. Concat the hidden states
        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

        # 4. Run the single flux_transformer blocks
        controlnet_single_block_samples = []
        for idx, block in enumerate(self.single_transformer_blocks):
            hidden_states = self._apply_single_transformer_block(
                idx=idx,
                block=block,
                config=config,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                image_rotary_embeddings=image_rotary_embeddings,
                controlnet_single_block_samples=controlnet_single_block_samples,
            )

        return controlnet_block_samples, controlnet_single_block_samples

    def _apply_single_transformer_block(
        self,
        idx: int,
        config: Config,
        block: SingleTransformerBlock,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        image_rotary_embeddings: mx.array,
        controlnet_single_block_samples: list[mx.array],
    ) -> mx.array:
        # 1. Apply single flux_transformer block
        hidden_states = block(
            hidden_states=hidden_states,
            text_embeddings=text_embeddings,
            rotary_embeddings=image_rotary_embeddings,
        )

        # 2. Apply controlnet block
        states = hidden_states[:, encoder_hidden_states.shape[1] :]
        controlnet_sample = self.controlnet_single_blocks[idx](states)
        scaled_controlnet_sample = controlnet_sample * config.controlnet_strength
        controlnet_single_block_samples.append(scaled_controlnet_sample)

        return hidden_states

    def _apply_joint_transformer_block(
        self,
        idx: int,
        config: Config,
        block: JointTransformerBlock,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        image_rotary_embeddings: mx.array,
        controlnet_block_samples: list[mx.array],
    ) -> tuple[mx.array, mx.array]:
        # 1. Apply joint flux_transformer block
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            text_embeddings=text_embeddings,
            rotary_embeddings=image_rotary_embeddings,
        )

        # 2. Apply controlnet block
        controlnet_sample = self.controlnet_blocks[idx](hidden_states)
        scaled_controlnet_example = controlnet_sample * config.controlnet_strength
        controlnet_block_samples.append(scaled_controlnet_example)

        return encoder_hidden_states, hidden_states
