from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget
from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping


class Krea2WeightMapping(WeightMapping):
    @staticmethod
    def _identity(*keys: str) -> List[WeightTarget]:
        return [WeightTarget(to_pattern=k, from_pattern=[k]) for k in keys]

    @staticmethod
    def _attn(prefix: str) -> List[WeightTarget]:
        keys = [
            f"{prefix}.attn.wq.weight",
            f"{prefix}.attn.wk.weight",
            f"{prefix}.attn.wv.weight",
            f"{prefix}.attn.wo.weight",
            f"{prefix}.attn.gate.weight",
            f"{prefix}.attn.qknorm.qnorm.scale",
            f"{prefix}.attn.qknorm.knorm.scale",
        ]
        return Krea2WeightMapping._identity(*keys)

    @staticmethod
    def _mlp(prefix: str) -> List[WeightTarget]:
        return Krea2WeightMapping._identity(
            f"{prefix}.mlp.gate.weight", f"{prefix}.mlp.up.weight", f"{prefix}.mlp.down.weight"
        )

    @staticmethod
    def _norm_pair(prefix: str) -> List[WeightTarget]:
        return Krea2WeightMapping._identity(f"{prefix}.prenorm.scale", f"{prefix}.postnorm.scale")

    @staticmethod
    def _text_fusion_block(prefix: str) -> List[WeightTarget]:
        return (
            Krea2WeightMapping._norm_pair(prefix) + Krea2WeightMapping._attn(prefix) + Krea2WeightMapping._mlp(prefix)
        )

    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        targets: List[WeightTarget] = []

        # Patch embed
        targets += Krea2WeightMapping._identity("first.weight", "first.bias")

        # Main single-stream blocks (expanded over num_layers via {layer})
        targets += Krea2WeightMapping._norm_pair("blocks.{layer}")
        targets += Krea2WeightMapping._attn("blocks.{layer}")
        targets += Krea2WeightMapping._mlp("blocks.{layer}")
        targets += Krea2WeightMapping._identity("blocks.{layer}.mod.lin")

        # Timestep path: tmlp (Sequential 0/2), tproj (Sequential 1)
        targets += [
            WeightTarget(to_pattern="tmlp.linear_in.weight", from_pattern=["tmlp.0.weight"]),
            WeightTarget(to_pattern="tmlp.linear_in.bias", from_pattern=["tmlp.0.bias"]),
            WeightTarget(to_pattern="tmlp.linear_out.weight", from_pattern=["tmlp.2.weight"]),
            WeightTarget(to_pattern="tmlp.linear_out.bias", from_pattern=["tmlp.2.bias"]),
            WeightTarget(to_pattern="tproj.linear.weight", from_pattern=["tproj.1.weight"]),
            WeightTarget(to_pattern="tproj.linear.bias", from_pattern=["tproj.1.bias"]),
        ]

        # Text fusion: 2 layerwise + projector + 2 refiner blocks
        for i in range(2):
            targets += Krea2WeightMapping._text_fusion_block(f"txtfusion.layerwise_blocks.{i}")
        targets += Krea2WeightMapping._identity("txtfusion.projector.weight")
        for i in range(2):
            targets += Krea2WeightMapping._text_fusion_block(f"txtfusion.refiner_blocks.{i}")

        # Text MLP: Sequential (0=RMSNorm, 1=Linear, 3=Linear)
        targets += [
            WeightTarget(to_pattern="txtmlp.norm.scale", from_pattern=["txtmlp.0.scale"]),
            WeightTarget(to_pattern="txtmlp.linear_in.weight", from_pattern=["txtmlp.1.weight"]),
            WeightTarget(to_pattern="txtmlp.linear_in.bias", from_pattern=["txtmlp.1.bias"]),
            WeightTarget(to_pattern="txtmlp.linear_out.weight", from_pattern=["txtmlp.3.weight"]),
            WeightTarget(to_pattern="txtmlp.linear_out.bias", from_pattern=["txtmlp.3.bias"]),
        ]

        # Final layer (last.up / last.down deliberately omitted)
        targets += Krea2WeightMapping._identity(
            "last.norm.scale",
            "last.linear.weight",
            "last.linear.bias",
            "last.modulation.lin",
        )

        return targets

    # -- Diffusers-format transformer ---------------------------------------------------
    # The official Krea 2 repo ships the transformer in diffusers layout (sharded
    # transformer/*.safetensors with different key names). These helpers mirror the
    # native mapping above but read the diffusers keys. The full diffusers<->MLX name
    # correspondence is also recorded in krea2_lora_mapping.py.

    @staticmethod
    def _diffusers_norm_pair(mlx: str, hf: str) -> List[WeightTarget]:
        return [
            WeightTarget(to_pattern=f"{mlx}.prenorm.scale", from_pattern=[f"{hf}.norm1.weight"]),
            WeightTarget(to_pattern=f"{mlx}.postnorm.scale", from_pattern=[f"{hf}.norm2.weight"]),
        ]

    @staticmethod
    def _diffusers_attn(mlx: str, hf: str) -> List[WeightTarget]:
        return [
            WeightTarget(to_pattern=f"{mlx}.attn.wq.weight", from_pattern=[f"{hf}.attn.to_q.weight"]),
            WeightTarget(to_pattern=f"{mlx}.attn.wk.weight", from_pattern=[f"{hf}.attn.to_k.weight"]),
            WeightTarget(to_pattern=f"{mlx}.attn.wv.weight", from_pattern=[f"{hf}.attn.to_v.weight"]),
            WeightTarget(to_pattern=f"{mlx}.attn.wo.weight", from_pattern=[f"{hf}.attn.to_out.0.weight"]),
            WeightTarget(to_pattern=f"{mlx}.attn.gate.weight", from_pattern=[f"{hf}.attn.to_gate.weight"]),
            WeightTarget(to_pattern=f"{mlx}.attn.qknorm.qnorm.scale", from_pattern=[f"{hf}.attn.norm_q.weight"]),
            WeightTarget(to_pattern=f"{mlx}.attn.qknorm.knorm.scale", from_pattern=[f"{hf}.attn.norm_k.weight"]),
        ]

    @staticmethod
    def _diffusers_mlp(mlx: str, hf: str) -> List[WeightTarget]:
        return [
            WeightTarget(to_pattern=f"{mlx}.mlp.gate.weight", from_pattern=[f"{hf}.ff.gate.weight"]),
            WeightTarget(to_pattern=f"{mlx}.mlp.up.weight", from_pattern=[f"{hf}.ff.up.weight"]),
            WeightTarget(to_pattern=f"{mlx}.mlp.down.weight", from_pattern=[f"{hf}.ff.down.weight"]),
        ]

    @staticmethod
    def _diffusers_text_fusion_block(mlx: str, hf: str) -> List[WeightTarget]:
        return (
            Krea2WeightMapping._diffusers_norm_pair(mlx, hf)
            + Krea2WeightMapping._diffusers_attn(mlx, hf)
            + Krea2WeightMapping._diffusers_mlp(mlx, hf)
        )

    @staticmethod
    def get_transformer_mapping_diffusers() -> List[WeightTarget]:
        targets: List[WeightTarget] = []

        # Patch embed
        targets += [
            WeightTarget(to_pattern="first.weight", from_pattern=["img_in.weight"]),
            WeightTarget(to_pattern="first.bias", from_pattern=["img_in.bias"]),
        ]

        # Main single-stream blocks (expanded over num_layers via {layer})
        mlx_block, hf_block = "blocks.{layer}", "transformer_blocks.{layer}"
        targets += Krea2WeightMapping._diffusers_norm_pair(mlx_block, hf_block)
        targets += Krea2WeightMapping._diffusers_attn(mlx_block, hf_block)
        targets += Krea2WeightMapping._diffusers_mlp(mlx_block, hf_block)
        # Diffusers stores modulation as a (6, dim) scale_shift_table; the MLX module holds
        # it flat as (6*dim,) and adds it to the timestep vector before splitting into 6.
        targets += [
            WeightTarget(
                to_pattern=f"{mlx_block}.mod.lin",
                from_pattern=[f"{hf_block}.scale_shift_table"],
                transform=lambda t: t.reshape(-1),
            )
        ]

        # Timestep path: tmlp (in/out) + tproj
        targets += [
            WeightTarget(to_pattern="tmlp.linear_in.weight", from_pattern=["time_embed.linear_1.weight"]),
            WeightTarget(to_pattern="tmlp.linear_in.bias", from_pattern=["time_embed.linear_1.bias"]),
            WeightTarget(to_pattern="tmlp.linear_out.weight", from_pattern=["time_embed.linear_2.weight"]),
            WeightTarget(to_pattern="tmlp.linear_out.bias", from_pattern=["time_embed.linear_2.bias"]),
            WeightTarget(to_pattern="tproj.linear.weight", from_pattern=["time_mod_proj.weight"]),
            WeightTarget(to_pattern="tproj.linear.bias", from_pattern=["time_mod_proj.bias"]),
        ]

        # Text fusion: 2 layerwise + projector + 2 refiner blocks
        for i in range(2):
            targets += Krea2WeightMapping._diffusers_text_fusion_block(
                f"txtfusion.layerwise_blocks.{i}", f"text_fusion.layerwise_blocks.{i}"
            )
        targets += [WeightTarget(to_pattern="txtfusion.projector.weight", from_pattern=["text_fusion.projector.weight"])]
        for i in range(2):
            targets += Krea2WeightMapping._diffusers_text_fusion_block(
                f"txtfusion.refiner_blocks.{i}", f"text_fusion.refiner_blocks.{i}"
            )

        # Text MLP (RMSNorm + two Linears)
        targets += [
            WeightTarget(to_pattern="txtmlp.norm.scale", from_pattern=["txt_in.norm.weight"]),
            WeightTarget(to_pattern="txtmlp.linear_in.weight", from_pattern=["txt_in.linear_1.weight"]),
            WeightTarget(to_pattern="txtmlp.linear_in.bias", from_pattern=["txt_in.linear_1.bias"]),
            WeightTarget(to_pattern="txtmlp.linear_out.weight", from_pattern=["txt_in.linear_2.weight"]),
            WeightTarget(to_pattern="txtmlp.linear_out.bias", from_pattern=["txt_in.linear_2.bias"]),
        ]

        # Final layer (final_layer.scale_shift_table is already (2, dim), a direct copy)
        targets += [
            WeightTarget(to_pattern="last.norm.scale", from_pattern=["final_layer.norm.weight"]),
            WeightTarget(to_pattern="last.linear.weight", from_pattern=["final_layer.linear.weight"]),
            WeightTarget(to_pattern="last.linear.bias", from_pattern=["final_layer.linear.bias"]),
            WeightTarget(to_pattern="last.modulation.lin", from_pattern=["final_layer.scale_shift_table"]),
        ]

        return targets

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        return QwenWeightMapping.get_vae_mapping()
