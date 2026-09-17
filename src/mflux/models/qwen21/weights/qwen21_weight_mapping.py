from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget
from mflux.models.common.weights.mapping.weight_transforms import WeightTransforms


class Qwen21WeightMapping(WeightMapping):
    NUM_ENCODER_RESNETS = 2
    NUM_DECODER_RESNETS = 3
    NUM_TRANSFORMER_BLOCKS = 32
    NUM_TEXT_LAYERS = 36

    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        targets = [
            WeightTarget(to_pattern="img_in.weight", from_pattern=["img_in.weight"]),
            WeightTarget(to_pattern="proj_out.weight", from_pattern=["proj_out.weight"]),
            WeightTarget(to_pattern="modulation.layers.1.weight", from_pattern=["modulation.1.weight"]),
            WeightTarget(to_pattern="norm_out.linear.weight", from_pattern=["norm_out.linear.weight"]),
            WeightTarget(to_pattern="txt_in.text_norm.weight", from_pattern=["txt_in.text_norm.weight"]),
            WeightTarget(to_pattern="txt_in.in_layer.weight", from_pattern=["txt_in.in_layer.weight"]),
            WeightTarget(to_pattern="txt_in.out_layer.weight", from_pattern=["txt_in.out_layer.weight"]),
            WeightTarget(
                to_pattern="time_text_embed.timestep_embedder.linear_1.weight",
                from_pattern=["time_text_embed.timestep_embedder.linear_1.weight"],
            ),
            WeightTarget(
                to_pattern="time_text_embed.timestep_embedder.linear_2.weight",
                from_pattern=["time_text_embed.timestep_embedder.linear_2.weight"],
            ),
        ]
        for block in range(Qwen21WeightMapping.NUM_TRANSFORMER_BLOCKS):
            targets.extend(
                WeightTarget(
                    to_pattern=f"transformer_blocks.{block}.attn.{param}.weight",
                    from_pattern=[f"transformer_blocks.{block}.attn.{param}.weight"],
                )
                for param in ["to_q", "to_k", "to_v", "norm_q", "norm_k"]
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"transformer_blocks.{block}.attn.to_out.0.weight",
                    from_pattern=[f"transformer_blocks.{block}.attn.to_out.0.weight"],
                )
            )
            targets.extend(
                WeightTarget(
                    to_pattern=f"transformer_blocks.{block}.img_mlp.{param}.weight",
                    from_pattern=[f"transformer_blocks.{block}.img_mlp.{param}.weight"],
                )
                for param in ["proj", "out", "gate_layer"]
            )
        return targets

    @staticmethod
    def get_text_encoder_mapping() -> List[WeightTarget]:
        targets = [
            WeightTarget(
                to_pattern="embed_tokens.weight",
                from_pattern=["model.language_model.embed_tokens.weight"],
            ),
            WeightTarget(to_pattern="norm.weight", from_pattern=["model.language_model.norm.weight"]),
        ]
        for layer in range(Qwen21WeightMapping.NUM_TEXT_LAYERS):
            prefix = f"model.language_model.layers.{layer}"
            to_prefix = f"layers.{layer}"
            targets.extend(
                WeightTarget(
                    to_pattern=f"{to_prefix}.{param}.weight",
                    from_pattern=[f"{prefix}.{param}.weight"],
                )
                for param in ["input_layernorm", "post_attention_layernorm"]
            )
            targets.extend(
                WeightTarget(
                    to_pattern=f"{to_prefix}.self_attn.{param}.weight",
                    from_pattern=[f"{prefix}.self_attn.{param}.weight"],
                )
                for param in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"]
            )
            targets.extend(
                WeightTarget(
                    to_pattern=f"{to_prefix}.mlp.{param}.weight",
                    from_pattern=[f"{prefix}.mlp.{param}.weight"],
                )
                for param in ["gate_proj", "up_proj", "down_proj"]
            )
        return targets

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        targets = []
        for conv in [
            "encoder.conv_in",
            "encoder.conv_out",
            "decoder.conv_in",
            "decoder.conv_out",
            "quant_conv",
            "post_quant_conv",
        ]:
            targets.append(
                WeightTarget(
                    to_pattern=f"{conv}.conv.weight",
                    from_pattern=[f"{conv}.weight"],
                    transform=WeightTransforms.transpose_conv2d_weight,
                )
            )
            targets.append(WeightTarget(to_pattern=f"{conv}.conv.bias", from_pattern=[f"{conv}.bias"]))

        # encoder dims run [96, 96, 192, 384, 768, 768]: five residual down blocks where the
        # first four own a downsampler and only the middle three change channel count
        for block in range(5):
            targets.extend(
                Qwen21WeightMapping._resnet_group(
                    prefix=f"encoder.down_blocks.{block}",
                    num_resnets=Qwen21WeightMapping.NUM_ENCODER_RESNETS,
                    with_shortcut=block in (1, 2, 3),
                )
            )
            if block < 4:
                targets.append(
                    WeightTarget(
                        to_pattern=f"encoder.down_blocks.{block}.downsampler.conv.weight",
                        from_pattern=[f"encoder.down_blocks.{block}.downsampler.resample.1.weight"],
                        transform=WeightTransforms.transpose_conv2d_weight,
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"encoder.down_blocks.{block}.downsampler.conv.bias",
                        from_pattern=[f"encoder.down_blocks.{block}.downsampler.resample.1.bias"],
                    )
                )

        # decoder dims run [1152, 1152, 1152, 576, 288, 144]: five residual up blocks where
        # only the first four own an upsampler and only the last three change channel count
        for block in range(5):
            targets.extend(
                Qwen21WeightMapping._resnet_group(
                    prefix=f"decoder.up_blocks.{block}",
                    num_resnets=Qwen21WeightMapping.NUM_DECODER_RESNETS,
                    with_shortcut=block >= 2,
                )
            )
            if block < 4:
                targets.append(
                    WeightTarget(
                        to_pattern=f"decoder.up_blocks.{block}.upsampler.conv.weight",
                        from_pattern=[f"decoder.up_blocks.{block}.upsampler.resample.1.weight"],
                        transform=WeightTransforms.transpose_conv2d_weight,
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"decoder.up_blocks.{block}.upsampler.conv.bias",
                        from_pattern=[f"decoder.up_blocks.{block}.upsampler.resample.1.bias"],
                    )
                )

        for side in ["encoder", "decoder"]:
            targets.extend(
                Qwen21WeightMapping._resnet_group(
                    prefix=f"{side}.mid_block",
                    num_resnets=2,
                    with_shortcut=False,
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"{side}.mid_block.attentions.0.norm.weight",
                    from_pattern=[f"{side}.mid_block.attentions.0.norm.gamma"],
                    transform=WeightTransforms.reshape_gamma_to_1d,
                )
            )
            for param in ["to_qkv", "proj"]:
                targets.append(
                    WeightTarget(
                        to_pattern=f"{side}.mid_block.attentions.0.{param}.weight",
                        from_pattern=[f"{side}.mid_block.attentions.0.{param}.weight"],
                        transform=WeightTransforms.transpose_conv2d_weight,
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{side}.mid_block.attentions.0.{param}.bias",
                        from_pattern=[f"{side}.mid_block.attentions.0.{param}.bias"],
                    )
                )
            targets.append(
                WeightTarget(
                    to_pattern=f"{side}.norm_out.weight",
                    from_pattern=[f"{side}.norm_out.gamma"],
                    transform=WeightTransforms.reshape_gamma_to_1d,
                )
            )
        return targets

    @staticmethod
    def _resnet_group(prefix: str, num_resnets: int, with_shortcut: bool) -> List[WeightTarget]:
        targets = []
        for i in range(num_resnets):
            targets.extend(
                WeightTarget(
                    to_pattern=f"{prefix}.resnets.{i}.{norm}.weight",
                    from_pattern=[f"{prefix}.resnets.{i}.{norm}.gamma"],
                    transform=WeightTransforms.reshape_gamma_to_1d,
                )
                for norm in ["norm1", "norm2"]
            )
            for conv in ["conv1", "conv2"]:
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix}.resnets.{i}.{conv}.conv.weight",
                        from_pattern=[f"{prefix}.resnets.{i}.{conv}.weight"],
                        transform=WeightTransforms.transpose_conv2d_weight,
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix}.resnets.{i}.{conv}.conv.bias",
                        from_pattern=[f"{prefix}.resnets.{i}.{conv}.bias"],
                    )
                )
            if with_shortcut and i == 0:
                # only the first resnet changes channel count; later ones are out -> out
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix}.resnets.{i}.conv_shortcut.conv.weight",
                        from_pattern=[f"{prefix}.resnets.{i}.conv_shortcut.weight"],
                        transform=WeightTransforms.transpose_conv2d_weight,
                    )
                )
                targets.append(
                    WeightTarget(
                        to_pattern=f"{prefix}.resnets.{i}.conv_shortcut.conv.bias",
                        from_pattern=[f"{prefix}.resnets.{i}.conv_shortcut.bias"],
                    )
                )
        return targets
