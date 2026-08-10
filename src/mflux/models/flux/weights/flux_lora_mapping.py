from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget
from mflux.models.common.lora.mapping.lora_transforms import LoraTransforms


class FluxLoRAMapping(LoRAMapping):
    @staticmethod
    def _dora_scale_patterns(*base_patterns: str) -> list[str]:
        patterns: list[str] = []
        for base_pattern in base_patterns:
            patterns.append(f"{base_pattern}.dora_scale")
            if base_pattern.startswith(("lycoris_", "lycoris__")):
                patterns.append(f"{FluxLoRAMapping._lycoris_to_lora_unet_base(base_pattern)}.dora_scale")
        return patterns

    @staticmethod
    def _lycoris_to_lora_unet_base(lycoris_base: str) -> str:
        if lycoris_base.startswith("lycoris__"):
            return f"lora_unet_{lycoris_base[len('lycoris__') :]}"
        if lycoris_base.startswith("lycoris_"):
            return f"lora_unet_{lycoris_base[len('lycoris_') :]}"
        return f"lora_unet_{lycoris_base}"

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets = []

        targets.extend(FluxLoRAMapping._get_observed_lycoris_global_targets())
        targets.extend(FluxLoRAMapping._get_standard_transformer_block_targets())
        targets.extend(FluxLoRAMapping._get_standard_single_transformer_block_targets())

        targets.extend(FluxLoRAMapping._get_bfl_transformer_block_targets())
        targets.extend(FluxLoRAMapping._get_bfl_single_transformer_block_targets())

        FluxLoRAMapping._add_observed_lycoris_lokr_patterns(targets)
        return targets

    @staticmethod
    def _get_observed_lycoris_global_targets() -> list[LoRATarget]:
        return [
            FluxLoRAMapping._observed_lycoris_lokr_target("x_embedder", "lycoris__x_embedder"),
            FluxLoRAMapping._observed_lycoris_lokr_target("context_embedder", "lycoris__context_embedder"),
            FluxLoRAMapping._observed_lycoris_lokr_target("norm_out.linear", "lycoris__norm_out_linear"),
            FluxLoRAMapping._observed_lycoris_lokr_target("proj_out", "lycoris__proj_out"),
            FluxLoRAMapping._observed_lycoris_lokr_target(
                "time_text_embed.timestep_embedder.linear_1",
                "lycoris__time_text_embed_timestep_embedder_linear_1",
            ),
            FluxLoRAMapping._observed_lycoris_lokr_target(
                "time_text_embed.timestep_embedder.linear_2",
                "lycoris__time_text_embed_timestep_embedder_linear_2",
            ),
            FluxLoRAMapping._observed_lycoris_lokr_target(
                "time_text_embed.text_embedder.linear_1",
                "lycoris__time_text_embed_text_embedder_linear_1",
            ),
            FluxLoRAMapping._observed_lycoris_lokr_target(
                "time_text_embed.text_embedder.linear_2",
                "lycoris__time_text_embed_text_embedder_linear_2",
            ),
            FluxLoRAMapping._observed_lycoris_lokr_target(
                "time_text_embed.guidance_embedder.linear_1",
                "lycoris__time_text_embed_guidance_embedder_linear_1",
            ),
            FluxLoRAMapping._observed_lycoris_lokr_target(
                "time_text_embed.guidance_embedder.linear_2",
                "lycoris__time_text_embed_guidance_embedder_linear_2",
            ),
        ]

    @staticmethod
    def _observed_lycoris_lokr_target(model_path: str, source_base: str) -> LoRATarget:
        lora_unet_base = FluxLoRAMapping._lycoris_to_lora_unet_base(source_base)
        return LoRATarget(
            model_path=model_path,
            possible_up_patterns=[],
            possible_down_patterns=[],
            possible_alpha_patterns=[f"{source_base}.alpha"],
            possible_lokr_w1_patterns=[f"{source_base}.lokr_w1", f"{lora_unet_base}.lokr_w1"],
            possible_lokr_w2_patterns=[f"{source_base}.lokr_w2", f"{lora_unet_base}.lokr_w2"],
            possible_dora_scale_patterns=FluxLoRAMapping._dora_scale_patterns(source_base),
        )

    @staticmethod
    def _add_observed_lycoris_lokr_patterns(targets: list[LoRATarget]) -> None:
        for target in targets:
            if not target.model_path.startswith(("transformer_blocks.", "single_transformer_blocks.")):
                continue

            source_bases = {target.model_path}
            for alpha_pattern in target.possible_alpha_patterns:
                if not alpha_pattern.endswith(".alpha"):
                    continue
                source_base = alpha_pattern.removesuffix(".alpha")
                if source_base.startswith("transformer."):
                    source_base = source_base.removeprefix("transformer.")
                if source_base.startswith(("transformer_blocks.", "single_transformer_blocks.")):
                    source_bases.add(source_base)

            observed_bases = []
            for source_base in source_bases:
                source_key = source_base.replace(".", "_")
                observed_bases.extend(
                    [
                        f"lycoris_{source_key}",
                        f"lycoris__{source_key}",
                        f"lora_unet_{source_key}",
                    ]
                )
            target.possible_alpha_patterns.extend(f"{base}.alpha" for base in observed_bases)
            target.possible_lokr_w1_patterns.extend(f"{base}.lokr_w1" for base in observed_bases)
            target.possible_lokr_w2_patterns.extend(f"{base}.lokr_w2" for base in observed_bases)
            target.possible_dora_scale_patterns.extend(FluxLoRAMapping._dora_scale_patterns(*observed_bases))

    @staticmethod
    def _get_standard_transformer_block_targets() -> list[LoRATarget]:
        return [
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_q",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_q.lora_B.weight",
                    "transformer.transformer_blocks.{block}.attn.to_q.lora_B",
                    "transformer_blocks.{block}.attn.to_q.lora_up.weight",
                    "transformer_blocks.{block}.attn.to_q.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_q.lora_A.weight",
                    "transformer.transformer_blocks.{block}.attn.to_q.lora_A",
                    "transformer_blocks.{block}.attn.to_q.lora_down.weight",
                    "transformer_blocks.{block}.attn.to_q.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_q.alpha",
                    "transformer_blocks.{block}.attn.to_q.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_k",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_k.lora_B.weight",
                    "transformer.transformer_blocks.{block}.attn.to_k.lora_B",
                    "transformer_blocks.{block}.attn.to_k.lora_up.weight",
                    "transformer_blocks.{block}.attn.to_k.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_k.lora_A.weight",
                    "transformer.transformer_blocks.{block}.attn.to_k.lora_A",
                    "transformer_blocks.{block}.attn.to_k.lora_down.weight",
                    "transformer_blocks.{block}.attn.to_k.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_k.alpha",
                    "transformer_blocks.{block}.attn.to_k.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_v",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_v.lora_B.weight",
                    "transformer.transformer_blocks.{block}.attn.to_v.lora_B",
                    "transformer_blocks.{block}.attn.to_v.lora_up.weight",
                    "transformer_blocks.{block}.attn.to_v.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_v.lora_A.weight",
                    "transformer.transformer_blocks.{block}.attn.to_v.lora_A",
                    "transformer_blocks.{block}.attn.to_v.lora_down.weight",
                    "transformer_blocks.{block}.attn.to_v.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_v.alpha",
                    "transformer_blocks.{block}.attn.to_v.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_out.0",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_out.0.lora_B.weight",
                    "transformer.transformer_blocks.{block}.attn.to_out.0.lora_B",
                    "transformer_blocks.{block}.attn.to_out.0.lora_up.weight",
                    "transformer_blocks.{block}.attn.to_out.0.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_out.0.lora_A.weight",
                    "transformer.transformer_blocks.{block}.attn.to_out.0.lora_A",
                    "transformer_blocks.{block}.attn.to_out.0.lora_down.weight",
                    "transformer_blocks.{block}.attn.to_out.0.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_out.0.alpha",
                    "transformer_blocks.{block}.attn.to_out.0.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.add_q_proj",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.attn.add_q_proj.lora_B.weight",
                    "transformer.transformer_blocks.{block}.attn.add_q_proj.lora_B",
                    "transformer_blocks.{block}.attn.add_q_proj.lora_up.weight",
                    "transformer_blocks.{block}.attn.add_q_proj.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.attn.add_q_proj.lora_A.weight",
                    "transformer.transformer_blocks.{block}.attn.add_q_proj.lora_A",
                    "transformer_blocks.{block}.attn.add_q_proj.lora_down.weight",
                    "transformer_blocks.{block}.attn.add_q_proj.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.attn.add_q_proj.alpha",
                    "transformer_blocks.{block}.attn.add_q_proj.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.add_k_proj",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.attn.add_k_proj.lora_B.weight",
                    "transformer.transformer_blocks.{block}.attn.add_k_proj.lora_B",
                    "transformer_blocks.{block}.attn.add_k_proj.lora_up.weight",
                    "transformer_blocks.{block}.attn.add_k_proj.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.attn.add_k_proj.lora_A.weight",
                    "transformer.transformer_blocks.{block}.attn.add_k_proj.lora_A",
                    "transformer_blocks.{block}.attn.add_k_proj.lora_down.weight",
                    "transformer_blocks.{block}.attn.add_k_proj.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.attn.add_k_proj.alpha",
                    "transformer_blocks.{block}.attn.add_k_proj.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.add_v_proj",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.attn.add_v_proj.lora_B.weight",
                    "transformer.transformer_blocks.{block}.attn.add_v_proj.lora_B",
                    "transformer_blocks.{block}.attn.add_v_proj.lora_up.weight",
                    "transformer_blocks.{block}.attn.add_v_proj.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.attn.add_v_proj.lora_A.weight",
                    "transformer.transformer_blocks.{block}.attn.add_v_proj.lora_A",
                    "transformer_blocks.{block}.attn.add_v_proj.lora_down.weight",
                    "transformer_blocks.{block}.attn.add_v_proj.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.attn.add_v_proj.alpha",
                    "transformer_blocks.{block}.attn.add_v_proj.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_add_out",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_add_out.lora_B.weight",
                    "transformer.transformer_blocks.{block}.attn.to_add_out.lora_B",
                    "transformer_blocks.{block}.attn.to_add_out.lora_up.weight",
                    "transformer_blocks.{block}.attn.to_add_out.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_add_out.lora_A.weight",
                    "transformer.transformer_blocks.{block}.attn.to_add_out.lora_A",
                    "transformer_blocks.{block}.attn.to_add_out.lora_down.weight",
                    "transformer_blocks.{block}.attn.to_add_out.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.attn.to_add_out.alpha",
                    "transformer_blocks.{block}.attn.to_add_out.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff.linear1",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.ff.net.0.proj.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff.net.0.proj.lora_B",
                    "transformer.transformer_blocks.{block}.ff.linear1.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff.linear1.lora_B",
                    "transformer_blocks.{block}.ff.net.0.proj.lora_up.weight",
                    "transformer_blocks.{block}.ff.linear1.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.ff.net.0.proj.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff.net.0.proj.lora_A",
                    "transformer.transformer_blocks.{block}.ff.linear1.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff.linear1.lora_A",
                    "transformer_blocks.{block}.ff.net.0.proj.lora_down.weight",
                    "transformer_blocks.{block}.ff.linear1.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.ff.net.0.proj.alpha",
                    "transformer.transformer_blocks.{block}.ff.linear1.alpha",
                    "transformer_blocks.{block}.ff.net.0.proj.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff.linear2",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.ff.net.2.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff.net.2.lora_B",
                    "transformer.transformer_blocks.{block}.ff.linear2.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff.linear2.lora_B",
                    "transformer_blocks.{block}.ff.net.2.lora_up.weight",
                    "transformer_blocks.{block}.ff.linear2.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.ff.net.2.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff.net.2.lora_A",
                    "transformer.transformer_blocks.{block}.ff.linear2.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff.linear2.lora_A",
                    "transformer_blocks.{block}.ff.net.2.lora_down.weight",
                    "transformer_blocks.{block}.ff.linear2.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.ff.net.2.alpha",
                    "transformer.transformer_blocks.{block}.ff.linear2.alpha",
                    "transformer_blocks.{block}.ff.net.2.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff_context.linear1",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.ff_context.net.0.proj.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff_context.net.0.proj.lora_B",
                    "transformer.transformer_blocks.{block}.ff_context.linear1.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff_context.linear1.lora_B",
                    "transformer_blocks.{block}.ff_context.net.0.proj.lora_up.weight",
                    "transformer_blocks.{block}.ff_context.linear1.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.ff_context.net.0.proj.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff_context.net.0.proj.lora_A",
                    "transformer.transformer_blocks.{block}.ff_context.linear1.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff_context.linear1.lora_A",
                    "transformer_blocks.{block}.ff_context.net.0.proj.lora_down.weight",
                    "transformer_blocks.{block}.ff_context.linear1.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.ff_context.net.0.proj.alpha",
                    "transformer.transformer_blocks.{block}.ff_context.linear1.alpha",
                    "transformer_blocks.{block}.ff_context.net.0.proj.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff_context.linear2",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.ff_context.net.2.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff_context.net.2.lora_B",
                    "transformer.transformer_blocks.{block}.ff_context.linear2.lora_B.weight",
                    "transformer.transformer_blocks.{block}.ff_context.linear2.lora_B",
                    "transformer_blocks.{block}.ff_context.net.2.lora_up.weight",
                    "transformer_blocks.{block}.ff_context.linear2.lora_up.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.ff_context.net.2.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff_context.net.2.lora_A",
                    "transformer.transformer_blocks.{block}.ff_context.linear2.lora_A.weight",
                    "transformer.transformer_blocks.{block}.ff_context.linear2.lora_A",
                    "transformer_blocks.{block}.ff_context.net.2.lora_down.weight",
                    "transformer_blocks.{block}.ff_context.linear2.lora_down.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.ff_context.net.2.alpha",
                    "transformer.transformer_blocks.{block}.ff_context.linear2.alpha",
                    "transformer_blocks.{block}.ff_context.net.2.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.norm1.linear",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.norm1.linear.lora_B.weight",
                    "transformer.transformer_blocks.{block}.norm1.linear.lora_B",
                    "transformer_blocks.{block}.norm1.linear.lora_up.weight",
                    "transformer_blocks.{block}.norm1.linear.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.norm1.linear.lora_A.weight",
                    "transformer.transformer_blocks.{block}.norm1.linear.lora_A",
                    "transformer_blocks.{block}.norm1.linear.lora_down.weight",
                    "transformer_blocks.{block}.norm1.linear.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.norm1.linear.alpha",
                    "transformer_blocks.{block}.norm1.linear.alpha",
                ],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.norm1_context.linear",
                possible_up_patterns=[
                    "transformer.transformer_blocks.{block}.norm1_context.linear.lora_B.weight",
                    "transformer.transformer_blocks.{block}.norm1_context.linear.lora_B",
                    "transformer_blocks.{block}.norm1_context.linear.lora_up.weight",
                    "transformer_blocks.{block}.norm1_context.linear.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.transformer_blocks.{block}.norm1_context.linear.lora_A.weight",
                    "transformer.transformer_blocks.{block}.norm1_context.linear.lora_A",
                    "transformer_blocks.{block}.norm1_context.linear.lora_down.weight",
                    "transformer_blocks.{block}.norm1_context.linear.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.transformer_blocks.{block}.norm1_context.linear.alpha",
                    "transformer_blocks.{block}.norm1_context.linear.alpha",
                ],
            ),
        ]

    @staticmethod
    def _get_standard_single_transformer_block_targets() -> list[LoRATarget]:
        return [
            LoRATarget(
                model_path="single_transformer_blocks.{block}.attn.to_q",
                possible_up_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_q.lora_B.weight",
                    "transformer.single_transformer_blocks.{block}.attn.to_q.lora_B",
                    "single_transformer_blocks.{block}.attn.to_q.lora_up.weight",
                    "single_transformer_blocks.{block}.attn.to_q.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_q.lora_A.weight",
                    "transformer.single_transformer_blocks.{block}.attn.to_q.lora_A",
                    "single_transformer_blocks.{block}.attn.to_q.lora_down.weight",
                    "single_transformer_blocks.{block}.attn.to_q.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_q.alpha",
                    "single_transformer_blocks.{block}.attn.to_q.alpha",
                ],
            ),
            LoRATarget(
                model_path="single_transformer_blocks.{block}.attn.to_k",
                possible_up_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_k.lora_B.weight",
                    "transformer.single_transformer_blocks.{block}.attn.to_k.lora_B",
                    "single_transformer_blocks.{block}.attn.to_k.lora_up.weight",
                    "single_transformer_blocks.{block}.attn.to_k.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_k.lora_A.weight",
                    "transformer.single_transformer_blocks.{block}.attn.to_k.lora_A",
                    "single_transformer_blocks.{block}.attn.to_k.lora_down.weight",
                    "single_transformer_blocks.{block}.attn.to_k.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_k.alpha",
                    "single_transformer_blocks.{block}.attn.to_k.alpha",
                ],
            ),
            LoRATarget(
                model_path="single_transformer_blocks.{block}.attn.to_v",
                possible_up_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_v.lora_B.weight",
                    "transformer.single_transformer_blocks.{block}.attn.to_v.lora_B",
                    "single_transformer_blocks.{block}.attn.to_v.lora_up.weight",
                    "single_transformer_blocks.{block}.attn.to_v.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_v.lora_A.weight",
                    "transformer.single_transformer_blocks.{block}.attn.to_v.lora_A",
                    "single_transformer_blocks.{block}.attn.to_v.lora_down.weight",
                    "single_transformer_blocks.{block}.attn.to_v.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.single_transformer_blocks.{block}.attn.to_v.alpha",
                    "single_transformer_blocks.{block}.attn.to_v.alpha",
                ],
            ),
            LoRATarget(
                model_path="single_transformer_blocks.{block}.proj_mlp",
                possible_up_patterns=[
                    "transformer.single_transformer_blocks.{block}.proj_mlp.lora_B.weight",
                    "transformer.single_transformer_blocks.{block}.proj_mlp.lora_B",
                    "single_transformer_blocks.{block}.proj_mlp.lora_up.weight",
                    "single_transformer_blocks.{block}.proj_mlp.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.single_transformer_blocks.{block}.proj_mlp.lora_A.weight",
                    "transformer.single_transformer_blocks.{block}.proj_mlp.lora_A",
                    "single_transformer_blocks.{block}.proj_mlp.lora_down.weight",
                    "single_transformer_blocks.{block}.proj_mlp.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.single_transformer_blocks.{block}.proj_mlp.alpha",
                    "single_transformer_blocks.{block}.proj_mlp.alpha",
                ],
            ),
            LoRATarget(
                model_path="single_transformer_blocks.{block}.proj_out",
                possible_up_patterns=[
                    "transformer.single_transformer_blocks.{block}.proj_out.lora_B.weight",
                    "transformer.single_transformer_blocks.{block}.proj_out.lora_B",
                    "single_transformer_blocks.{block}.proj_out.lora_up.weight",
                    "single_transformer_blocks.{block}.proj_out.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.single_transformer_blocks.{block}.proj_out.lora_A.weight",
                    "transformer.single_transformer_blocks.{block}.proj_out.lora_A",
                    "single_transformer_blocks.{block}.proj_out.lora_down.weight",
                    "single_transformer_blocks.{block}.proj_out.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.single_transformer_blocks.{block}.proj_out.alpha",
                    "single_transformer_blocks.{block}.proj_out.alpha",
                ],
            ),
            LoRATarget(
                model_path="single_transformer_blocks.{block}.norm.linear",
                possible_up_patterns=[
                    "transformer.single_transformer_blocks.{block}.norm.linear.lora_B.weight",
                    "transformer.single_transformer_blocks.{block}.norm.linear.lora_B",
                    "single_transformer_blocks.{block}.norm.linear.lora_up.weight",
                    "single_transformer_blocks.{block}.norm.linear.lora_B.weight",
                ],
                possible_down_patterns=[
                    "transformer.single_transformer_blocks.{block}.norm.linear.lora_A.weight",
                    "transformer.single_transformer_blocks.{block}.norm.linear.lora_A",
                    "single_transformer_blocks.{block}.norm.linear.lora_down.weight",
                    "single_transformer_blocks.{block}.norm.linear.lora_A.weight",
                ],
                possible_alpha_patterns=[
                    "transformer.single_transformer_blocks.{block}.norm.linear.alpha",
                    "single_transformer_blocks.{block}.norm.linear.alpha",
                ],
            ),
        ]

    @staticmethod
    def _get_bfl_transformer_block_targets() -> list[LoRATarget]:
        return [
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_q",
                possible_up_patterns=["lora_unet_double_blocks_{block}_img_attn_qkv.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_img_attn_qkv.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_img_attn_qkv.alpha"],
                up_transform=LoraTransforms.split_q_up,
                down_transform=LoraTransforms.split_q_down,
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_k",
                possible_up_patterns=["lora_unet_double_blocks_{block}_img_attn_qkv.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_img_attn_qkv.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_img_attn_qkv.alpha"],
                up_transform=LoraTransforms.split_k_up,
                down_transform=LoraTransforms.split_k_down,
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_v",
                possible_up_patterns=["lora_unet_double_blocks_{block}_img_attn_qkv.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_img_attn_qkv.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_img_attn_qkv.alpha"],
                up_transform=LoraTransforms.split_v_up,
                down_transform=LoraTransforms.split_v_down,
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_out.0",
                possible_up_patterns=["lora_unet_double_blocks_{block}_img_attn_proj.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_img_attn_proj.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_img_attn_proj.alpha"],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.add_q_proj",
                possible_up_patterns=["lora_unet_double_blocks_{block}_txt_attn_qkv.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_txt_attn_qkv.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_txt_attn_qkv.alpha"],
                up_transform=LoraTransforms.split_q_up,
                down_transform=LoraTransforms.split_q_down,
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.add_k_proj",
                possible_up_patterns=["lora_unet_double_blocks_{block}_txt_attn_qkv.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_txt_attn_qkv.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_txt_attn_qkv.alpha"],
                up_transform=LoraTransforms.split_k_up,
                down_transform=LoraTransforms.split_k_down,
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.add_v_proj",
                possible_up_patterns=["lora_unet_double_blocks_{block}_txt_attn_qkv.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_txt_attn_qkv.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_txt_attn_qkv.alpha"],
                up_transform=LoraTransforms.split_v_up,
                down_transform=LoraTransforms.split_v_down,
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.attn.to_add_out",
                possible_up_patterns=["lora_unet_double_blocks_{block}_txt_attn_proj.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_txt_attn_proj.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_txt_attn_proj.alpha"],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff.linear1",
                possible_up_patterns=["lora_unet_double_blocks_{block}_img_mlp_0.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_img_mlp_0.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_img_mlp_0.alpha"],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff.linear2",
                possible_up_patterns=["lora_unet_double_blocks_{block}_img_mlp_2.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_img_mlp_2.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_img_mlp_2.alpha"],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff_context.linear1",
                possible_up_patterns=["lora_unet_double_blocks_{block}_txt_mlp_0.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_txt_mlp_0.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_txt_mlp_0.alpha"],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.ff_context.linear2",
                possible_up_patterns=["lora_unet_double_blocks_{block}_txt_mlp_2.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_txt_mlp_2.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_txt_mlp_2.alpha"],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.norm1.linear",
                possible_up_patterns=["lora_unet_double_blocks_{block}_img_mod_lin.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_img_mod_lin.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_img_mod_lin.alpha"],
            ),
            LoRATarget(
                model_path="transformer_blocks.{block}.norm1_context.linear",
                possible_up_patterns=["lora_unet_double_blocks_{block}_txt_mod_lin.lora_up.weight"],
                possible_down_patterns=["lora_unet_double_blocks_{block}_txt_mod_lin.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_double_blocks_{block}_txt_mod_lin.alpha"],
            ),
        ]

    @staticmethod
    def _get_bfl_single_transformer_block_targets() -> list[LoRATarget]:
        return [
            LoRATarget(
                model_path="single_transformer_blocks.{block}.attn.to_q",
                possible_up_patterns=["lora_unet_single_blocks_{block}_linear1.lora_up.weight"],
                possible_down_patterns=["lora_unet_single_blocks_{block}_linear1.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_single_blocks_{block}_linear1.alpha"],
                up_transform=LoraTransforms.split_single_q_up,
                down_transform=LoraTransforms.split_single_q_down,
            ),
            LoRATarget(
                model_path="single_transformer_blocks.{block}.attn.to_k",
                possible_up_patterns=["lora_unet_single_blocks_{block}_linear1.lora_up.weight"],
                possible_down_patterns=["lora_unet_single_blocks_{block}_linear1.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_single_blocks_{block}_linear1.alpha"],
                up_transform=LoraTransforms.split_single_k_up,
                down_transform=LoraTransforms.split_single_k_down,
            ),
            LoRATarget(
                model_path="single_transformer_blocks.{block}.attn.to_v",
                possible_up_patterns=["lora_unet_single_blocks_{block}_linear1.lora_up.weight"],
                possible_down_patterns=["lora_unet_single_blocks_{block}_linear1.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_single_blocks_{block}_linear1.alpha"],
                up_transform=LoraTransforms.split_single_v_up,
                down_transform=LoraTransforms.split_single_v_down,
            ),
            LoRATarget(
                model_path="single_transformer_blocks.{block}.proj_mlp",
                possible_up_patterns=["lora_unet_single_blocks_{block}_linear1.lora_up.weight"],
                possible_down_patterns=["lora_unet_single_blocks_{block}_linear1.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_single_blocks_{block}_linear1.alpha"],
                up_transform=LoraTransforms.split_single_mlp_up,
                down_transform=LoraTransforms.split_single_mlp_down,
            ),
            LoRATarget(
                model_path="single_transformer_blocks.{block}.proj_out",
                possible_up_patterns=["lora_unet_single_blocks_{block}_linear2.lora_up.weight"],
                possible_down_patterns=["lora_unet_single_blocks_{block}_linear2.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_single_blocks_{block}_linear2.alpha"],
            ),
            LoRATarget(
                model_path="single_transformer_blocks.{block}.norm.linear",
                possible_up_patterns=["lora_unet_single_blocks_{block}_modulation_lin.lora_up.weight"],
                possible_down_patterns=["lora_unet_single_blocks_{block}_modulation_lin.lora_down.weight"],
                possible_alpha_patterns=["lora_unet_single_blocks_{block}_modulation_lin.alpha"],
            ),
        ]
