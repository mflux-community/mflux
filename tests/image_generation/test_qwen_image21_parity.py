import mlx.core as mx
import numpy as np
import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLModel

from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.qwen21.reference.model.qwen_image21_text_encoder.text_encoder import QwenImage21TextEncoder
from mflux.models.qwen21.reference.model.qwen_image21_transformer.layout import QwenImage21Layout
from mflux.models.qwen21.reference.model.qwen_image21_transformer.transformer import QwenImage21Transformer
from mflux.models.qwen21.reference.model.qwen_image21_vae.vae import QwenImage21VAE
from mflux.models.qwen21.reference.weights.qwen_image21_weight_definition import QwenImage21WeightDefinition

pytestmark = pytest.mark.fast


class TestQwenImage21Reference:
    @staticmethod
    def reference_module(name):
        module = pytest.importorskip(
            name, reason="Install a diffusers revision with QwenImage21 for reference comparisons"
        )
        if not hasattr(module, "QwenImage21Transformer2DModel") and name == "diffusers":
            pytest.skip("The installed diffusers predates QwenImage21")
        return module

    @staticmethod
    def transfer(model, reference, transform=None):
        weights = []
        for key, tensor in reference.state_dict().items():
            value = mx.array(tensor.detach().float().cpu().numpy())
            weights.append((key, transform(key, value) if transform else value))
        model.load_weights(weights)

    @staticmethod
    def assert_close(actual, expected):
        np.testing.assert_allclose(np.array(actual), expected.detach().float().cpu().numpy(), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize(
        "slots,shapes",
        [
            ([False] * 5, [(1, 2, 2)]),
            ([False, True, False, True, False], [(1, 2, 2)] * 3),
            ([False, True, True, False], [(1, 2, 2)] * 3),
        ],
    )
    def test_transformer_prefill_and_cached_decode(self, slots, shapes):
        diffusers = self.reference_module("diffusers")
        module = self.reference_module("diffusers.models.transformers.transformer_qwenimage21")
        config = dict(
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=16,
            axes_dims_rope=(4, 6, 6),
            context_in_dim=32,
            in_channels=4,
            out_channels=4,
            mlp_ratio=3,
            eps=1e-6,
            causal_condition=True,
        )
        torch.manual_seed(42)
        reference = diffusers.QwenImage21Transformer2DModel(**config).to("mps").eval()
        model = QwenImage21Transformer(config)
        self.transfer(model, reference)
        layout = QwenImage21Layout.create(mx.array(slots), shapes, (4, 6, 6))
        latents = torch.randn(1, len(shapes) * 4, 4, device="mps")
        text = torch.randn(1, len(slots), 32, device="mps")
        mask = torch.tensor([slots + [True]], device="mps")
        cache = []
        ref_cache = module.QwenImage21KVCache(2)
        for index, timestep in enumerate([0.8, 0.5]):
            t = torch.tensor([timestep], device="mps")
            with torch.no_grad():
                expected = reference(
                    latents,
                    text,
                    t,
                    [shapes],
                    mask,
                    kv_cache=ref_cache,
                    kv_cache_mode="extract" if index == 0 else "cached",
                ).sample[:, -4:]
            actual = model(
                mx.array(latents.cpu().numpy()), mx.array(text.cpu().numpy()), mx.array([timestep]), layout, cache
            )
            self.assert_close(actual, expected)

    def test_vae_encode_and_decode(self):
        diffusers = self.reference_module("diffusers")
        config = dict(
            base_dim=4,
            decoder_base_dim=4,
            z_dim=4,
            dim_mult=[1, 2, 4, 8, 8],
            num_res_blocks=1,
            attn_scales=[],
            temporal_downsample=[False, True, True, True],
            dropout=0.0,
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
            is_residual=True,
            in_channels=4,
            out_channels=4,
            patch_size=None,
            scale_factor_spatial=16,
            scale_factor_temporal=8,
        )
        torch.manual_seed(42)
        reference = diffusers.AutoencoderKLQwenImage21(**config).to("mps").eval()
        model = QwenImage21VAE(config)
        self.transfer(model, reference, QwenImage21WeightDefinition.vae_weight)
        pixels = torch.randn(1, 4, 1, 32, 32, device="mps")
        latents = torch.randn(1, 4, 1, 2, 2, device="mps")
        with torch.no_grad():
            self.assert_close(model.encode(mx.array(pixels.cpu().numpy())), reference.encode(pixels).latent_dist.mode())
            self.assert_close(model.decode(mx.array(latents.cpu().numpy())), reference.decode(latents).sample)

    @pytest.mark.parametrize(
        "ids,grids",
        [
            ([1, 2, 3, 4, 5], None),
            ([1, 98, 99, 99, 99, 99, 97, 5], [[1, 4, 4]]),
            ([1, 98, 99, 99, 99, 99, 97, 98, 99, 99, 97, 5], [[1, 4, 4], [1, 2, 4]]),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_text_encoder_and_deepstack(self, ids, grids, dtype):
        config = dict(
            image_token_id=99,
            vision_start_token_id=98,
            vision_end_token_id=97,
            text_config=dict(
                vocab_size=128,
                hidden_size=32,
                num_hidden_layers=3,
                num_attention_heads=2,
                num_key_value_heads=1,
                intermediate_size=64,
                head_dim=16,
                max_position_embeddings=2048,
                rope_theta=5000000.0,
                rms_norm_eps=1e-6,
                attention_bias=False,
                rope_scaling=dict(rope_type="default", mrope_section=[4, 2, 2], mrope_interleaved=True),
            ),
            vision_config=dict(
                depth=3,
                hidden_size=32,
                intermediate_size=64,
                num_heads=2,
                patch_size=2,
                temporal_patch_size=2,
                in_channels=3,
                spatial_merge_size=2,
                out_hidden_size=32,
                num_position_embeddings=16,
                deepstack_visual_indexes=[0, 1, 2],
                hidden_act="gelu_pytorch_tanh",
            ),
        )
        torch.manual_seed(42)
        reference = Qwen3VLModel(Qwen3VLConfig(**config)).to(device="mps", dtype=dtype).eval()
        reference.language_model.norm = torch.nn.Identity()
        if dtype == torch.float32:
            # Exercise exact GELU versus its tanh approximation outside the near-linear initialization range.
            with torch.no_grad():
                for merger in [reference.visual.merger, *reference.visual.deepstack_merger_list]:
                    merger.linear_fc1.weight.mul_(10)
        model = QwenImage21TextEncoder(config)
        weights = [
            (
                key,
                QwenImage21WeightDefinition.text_weight(
                    key,
                    mx.array(value.detach().float().cpu().numpy()).astype(
                        mx.float32 if dtype == torch.float32 else mx.bfloat16
                    ),
                ),
            )
            for key, value in reference.state_dict().items()
        ]
        model.load_weights(weights, strict=False)
        ids = torch.tensor([ids], device="mps")
        kwargs, mlx_kwargs = {}, {}
        if grids is not None:
            grids = torch.tensor(grids, device="mps")
            pixels = torch.randn(int(grids.prod(-1).sum()), 24, device="mps", dtype=dtype)
            kwargs = dict(pixel_values=pixels, image_grid_thw=grids, mm_token_type_ids=(ids == 99).int())
            mlx_kwargs = dict(
                pixel_values=mx.array(pixels.float().cpu().numpy()), image_grid_thw=mx.array(grids.cpu().numpy())
            )
        with torch.no_grad():
            expected = reference(input_ids=ids, use_cache=False, **kwargs).last_hidden_state
        actual = model(mx.array(ids.cpu().numpy()), **mlx_kwargs)
        tolerance = 1e-4 if dtype == torch.float32 else 1e-2
        np.testing.assert_allclose(
            np.array(actual.astype(mx.float32)), expected.float().cpu().numpy(), atol=tolerance, rtol=tolerance
        )

    @pytest.mark.parametrize("width,height", [(1024, 1024), (2048, 2048), (1536, 2752)])
    def test_scheduler_matches_reference(self, width, height):
        diffusers = self.reference_module("diffusers")
        config = Config(ModelConfig.qwen_image_21(), num_inference_steps=40, width=width, height=height)
        reference = diffusers.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            use_dynamic_shifting=True,
            base_image_seq_len=256,
            max_image_seq_len=8192,
            base_shift=0.5,
            max_shift=0.9,
            shift_terminal=0.02,
        )
        mu = 0.5 + (config.image_seq_len - 256) * (0.9 - 0.5) / (8192 - 256)
        reference.set_timesteps(sigmas=np.linspace(1, 1 / 40, 40), mu=mu, device="mps")
        self.assert_close(config.scheduler.sigmas, reference.sigmas)
