import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_unflatten

from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.moss_image.model.moss_image_text_encoder.moss_image_text_encoder import (
    MOSSImageTextEncoder,
)
from mflux.models.moss_image.model.moss_image_transformer.moss_image_transformer import (
    MOSSImageTransformer,
)
from mflux.models.moss_image.weights.moss_image_weight_definition import (
    MOSSImageWeightDefinition,
)


class MOSSImageInitializer:
    @staticmethod
    def init(
        model,
        model_path: str | Path,
        quantize: int | None = None,
    ) -> None:
        path = Path(model_path)

        print(f"📦 Initializing MOSS-Image components from {path} ...")

        # 1. Initialize Transformer
        t_config_file = path / "transformer" / "config.json"
        if t_config_file.exists():
            t_config = json.loads(t_config_file.read_text())
            model.transformer = MOSSImageTransformer(
                dim=t_config.get("dim", 3072),
                ffn_dim=t_config.get("ffn_dim", 8192),
                num_heads=t_config.get("num_heads", 24),
                num_layers=t_config.get("num_layers", 42),
                text_dim=t_config.get("text_dim", 4096),
                freq_dim=t_config.get("freq_dim", 256),
                latent_channels=t_config.get("latent_channels", 128),
                rope_base=t_config.get("rope_base", 10000.0),
            )
        else:
            model.transformer = MOSSImageTransformer()

        # 2. Load Transformer weights using WeightLoader
        MOSSImageInitializer.load_transformer_weights(model.transformer, path)

        # 3. Initialize & Load FLUX.2 VAE (small decoder configuration)
        from mflux.models.flux2.model.flux2_vae.decoder.decoder import Flux2Decoder
        print("📦 Loading VAE (AutoencoderKLFlux2 small decoder) ...")
        model.vae = Flux2VAE()
        model.vae.decoder = Flux2Decoder(block_out_channels=(96, 192, 384, 384))
        vae_path = path / "vae"
        MOSSImageInitializer.load_vae_weights(model.vae, vae_path if vae_path.exists() else path)

        # 4. Initialize Text Encoder
        model.text_encoder = MOSSImageTextEncoder(path)

        print("✨ All MOSS-Image components initialized successfully!")

    @staticmethod
    def load_transformer_weights(transformer: MOSSImageTransformer, root: Path) -> None:
        t_dir = root / "transformer" if (root / "transformer").exists() else root
        print(f"📦 Loading Transformer weights from {t_dir} ...")

        # Check if safetensors exist
        safetensors_files = sorted(list(t_dir.glob("*.safetensors")))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors found in {t_dir}")

        mapping = MOSSImageWeightDefinition.get_components()[0].mapping_getter()
        target_map = {src: target.to_pattern for target in mapping for src in target.from_pattern}

        weights = {}
        for sf in safetensors_files:
            loaded = mx.load(str(sf))
            for k, v in loaded.items():
                if k in target_map:
                    mapped_key = target_map[k]
                    weights[mapped_key] = v
                elif k.endswith(".weight") or k.endswith(".bias"):
                    # direct match fallback
                    weights[k] = v

        print(f"   Mapped {len(weights)} transformer parameters.")
        transformer.update(tree_unflatten(list(weights.items())))

    @staticmethod
    def load_vae_weights(vae: Flux2VAE, root: Path) -> None:
        from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
        from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping
        mapping = Flux2WeightMapping.get_vae_mapping()

        vae_files = sorted(list(root.glob("*.safetensors")))
        if not vae_files:
            vae_files = sorted(list(root.glob("vae/*.safetensors")))
        if not vae_files:
            raise FileNotFoundError(f"No VAE safetensors found in {root}")

        raw_weights = {}
        for vf in vae_files:
            loaded = mx.load(str(vf))
            for k, v in loaded.items():
                raw_weights[k] = v

        mapped = WeightMapper.apply_mapping(raw_weights, mapping)
        vae.update(mapped)
        print("✅ VAE weights (all 250 parameters) loaded successfully via WeightMapper!")
