import gc
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.pid_decoder.caption_encoder import PidCaptionEncoder
from mflux.models.common.pid_decoder.gemma2.gemma2_config import Gemma2Config
from mflux.models.common.pid_decoder.gemma2.gemma2_model import Gemma2Model
from mflux.models.common.pid_decoder.gemma2.gemma2_weight_mapping import Gemma2WeightDefinition
from mflux.models.common.pid_decoder.pid_sampler import sample
from mflux.models.common.pid_decoder.pid_weight_mapping import convert_checkpoint
from mflux.models.common.pid_decoder.pixdit.pixdit_network import PidNet
from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader
from mflux.models.common.weights.loading.weight_definition import TokenizerDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader

PID_REPO_DEFAULT = "nvidia/PiD"
GEMMA2_REPO = "google/gemma-2-2b-it"

# PiD's LQ conditioning was trained on latents noised at sigma ~ U[0.0, 0.8]
# (add_sigma_max in nvidia/PiD's v1pt5 teacher configs). Past that the decoder has
# never seen the input distribution.
PID_MAX_DEGRADE_SIGMA = 0.8

# Per-latent-space checkpoint variants: (checkpoint path in nvidia/PiD, lq_latent_channels).
# PiD's LQ conditioning is the base model's VAE latent, so a checkpoint is selected by the
# *latent space* it was distilled against, not by the base model -- every mflux model sharing
# a VAE shares a variant (see each VAE class's `pid_variant`).
#
# Only the v1pt5 res2kto4k family is wired: those three checkpoints share one PidNet
# architecture. Verified against the real repo listing (2026-07-26) -- flux and qwenimage are
# both exactly 2_800_546_587 bytes, and flux2 is 2_800_841_499, i.e. +294_912 bytes = the 16
# extra input channels of PidLqProjection.latent_conv1 (16 * 1024 * 3*3 * 2 bytes, bf16). So
# lq_latent_channels is the only hyperparameter that differs. `_assert_full_weight_coverage`
# fails loudly if that ever stops holding.
#
# The res2k and non-v1pt5 res2kto4k checkpoints (sd3/sdxl/dinov2/siglip, plus res2k flux/flux2)
# are a *different* architecture -- res2k flux is 2_724_842_961 bytes, 75MB off v1pt5 -- whose
# config cannot be derived without reading the checkpoints. Not wired.
#
# A green-channel bias (-2 to -3 levels) was reported specific to the flux2 (32ch) checkpoint,
# absent on flux/qwen-image (16ch) -- checked whether LQProjection2D's channel-folding permute
# (pixdit_lq_projection.py::_align_latent_to_patch_grid, the one place a wrong-but-same-shape
# channel order could hide from _assert_full_weight_coverage) could be the cause: it can't be,
# because that branch only runs when z_to_patch_ratio < 1, and every wired variant here has
# z_to_patch_ratio = (sr_scale * latent_spatial_down_factor) / patch_size = (4*8)/16 = 2 --
# all three take the plain nearest-upsample path instead, with latent_proj_in_ch passed straight
# through unpermuted. lq_latent_channels also only ever reaches latent_conv1's in_channels
# (pixdit_network.py); the RGB output head is identical across variants. If the bias is real,
# it isn't an mflux-side wiring bug reachable from this code -- it would have to come from the
# checkpoint's own flux2 distillation or a latent-channel-order mismatch against whatever VAE
# NVIDIA trained it with, neither of which is checkable without the (gated) reference model.
#
# Reproduced and left uncorrected (measured 2026-07-30, ERNIE Turbo q8, 512x640 and 640x512,
# per-channel mean of the PiD output minus a Lanczos-4x VAE decode of the same latent/seed):
#   bright portrait  R -1.06  G -3.08  B -3.29   (matches the reported signature closely)
#   dark scene       R -1.89  G -1.62  B +1.77
# The drift is real but not a fixed per-channel offset -- B flips sign with the scene -- so the
# per-backbone linear correction in kijai/ComfyUI-KJNodes' PiDColorBiasCorrection does not
# transfer: subtracting its predicted bias at the first sampling step (its own convention) took
# the portrait to G -6.78 / B -7.76 and the dark scene to G -5.44, i.e. worse on both. Adding it
# instead lands the portrait at G +0.77 / B +1.39 but overshoots the dark scene to B +5.49. Two
# subjects, one model: not enough to ship a correction on, so nothing is applied here.
PID_CHECKPOINT_VARIANTS: dict[str, tuple[str, int]] = {
    "flux": ("checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth", 16),
    "flux2": ("checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth", 32),
    "qwen-image": ("checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth", 16),
}


@lru_cache(maxsize=1)
def _load_decoder(variant: str) -> "PidDecoder":
    return PidDecoder.from_pretrained(variant=variant)


def release_pid_decoder() -> None:
    """Drop the cached PidNet (~8GB of weights).

    The cache is what keeps a multi-seed run from reloading the checkpoint per seed, so nothing
    on the CLI path calls this -- the process exits instead. It exists because `MemorySaver` has
    no visibility into a module-level cache, so a long-lived host importing mflux would otherwise
    have no way to get that memory back.
    """
    _load_decoder.cache_clear()
    gc.collect()
    mx.clear_cache()


def pid_decode_latents(
    *,
    vae: nn.Module,
    latent: mx.array,
    caption: str,
    seed: int,
    degrade_sigma: float = 0.0,
) -> mx.array:
    """Decode `latent` with PiD instead of `vae`, picking the checkpoint from the VAE's own
    `pid_variant`. `latent` must be the unpacked VAE latent -- [B, C, H/8, W/8], the exact
    tensor `vae.decode` would receive.

    `degrade_sigma` deliberately noises the latent before decoding -- see PidDecoder.decode.

    Cached across calls (and across model instances) because loading costs ~8GB of downloads;
    the base pipeline's MLX buffers are released first, since PidNet's working set is much
    larger than the diffusion loop's."""
    if not 0.0 <= degrade_sigma <= PID_MAX_DEGRADE_SIGMA:
        raise ValueError(f"pid_degrade_sigma must be between 0.0 and {PID_MAX_DEGRADE_SIGMA}, got {degrade_sigma}")
    variant = getattr(vae, "pid_variant", None)
    if variant is None:
        raise ValueError(
            f"--pid-decode: no PiD checkpoint covers {type(vae).__name__}'s latent space. "
            f"Supported: {sorted(PID_CHECKPOINT_VARIANTS)}."
        )
    gc.collect()
    mx.clear_cache()
    decoder = _load_decoder(variant)
    return decoder.decode(latent=latent, caption=caption, seed=seed, degrade_sigma=degrade_sigma)


def _assert_full_weight_coverage(module: nn.Module, supplied: dict, label: str) -> None:
    """`Module.update(..., strict=False)` silently leaves any parameter with no matching
    key in `supplied` at its random-init value instead of raising (Finding 1, final
    integration review). Derive the real parameter-path set from the constructed module
    itself (not a hardcoded count) and fail loudly if any path got no value.

    `supplied` may be either a flat dict of dotted paths or a nested dict (both shapes are
    used by the two callers below) -- tree_flatten normalizes either into the same dotted
    path format `module.parameters()` uses, so the two sides are directly comparable.
    """
    expected_paths = {path for path, _ in tree_flatten(module.parameters())}
    supplied_paths = {path for path, _ in tree_flatten(supplied)}
    missing = sorted(expected_paths - supplied_paths)
    if missing:
        raise ValueError(
            f"{label}: {len(missing)} of {len(expected_paths)} parameter(s) got no value from the "
            f"converted checkpoint (still at random init) -- weight mapping is incomplete: {missing}"
        )


class PidDecoder:
    """NVIDIA PiD's pixel-diffusion super-resolving decoder (opt-in alternative to the VAE).

    Two independent weight sources (confirmed Task 10 -- the released `nvidia/PiD`
    checkpoint holds only `PidNet`'s 461 weights, no text encoder):
      - PidNet: nvidia/PiD's checkpoint via pid_weight_mapping.convert_checkpoint.
      - Gemma2Model (caption encoder): google/gemma-2-2b-it, a separate frozen HF repo.
    """

    SR_SCALE = 4
    VAE_COMPRESSION = 8  # matches every currently-supported base VAE's spatial_scale (see PID_CHECKPOINT_VARIANTS).

    def __init__(self, pid_net: PidNet, caption_encoder: PidCaptionEncoder):
        self.pid_net = pid_net
        self.caption_encoder = caption_encoder

    def decode(self, latent: mx.array, caption: str, seed: int = 0, degrade_sigma: float = 0.0) -> mx.array:
        """`degrade_sigma` deliberately noises `latent` to the given flow-matching noise level
        before decoding (0.0, the default, decodes it as-is). PiD's LQ gate was distilled
        against latents noised at sigma ~ U[0.0, 0.8] (see PID_MAX_DEGRADE_SIGMA); handing it a
        clean latent while still telling the gate sigma=0 is the one input configuration it was
        trained on the most, which is also why a clean latent can read as over-textured (e.g.
        invented skin detail) -- both halves have to move together, so we noise the latent by
        exactly the sigma we report, rather than merely relabeling a clean one."""
        expected_channels = self.pid_net.lq_proj.latent_channels
        if latent.shape[1] != expected_channels:
            raise ValueError(f"pid decode: latent has {latent.shape[1]} channels, expected {expected_channels}")

        _, _, zH, zW = latent.shape
        target_h = zH * self.VAE_COMPRESSION * self.SR_SCALE
        target_w = zW * self.VAE_COMPRESSION * self.SR_SCALE

        caption_embs = self.caption_encoder(caption)

        if degrade_sigma:
            # Independent RNG stream: drawing this from the seed the sampler itself uses would
            # also perturb sample()'s own pixel noise, reshuffling the whole image and making
            # it impossible to tell a texture change from a different sample (mirrors
            # Krea2Sampler's seed ^ 0x5DE, same reasoning).
            eps = mx.random.normal(latent.shape, key=mx.random.key(seed ^ 0x91D))
            latent = LatentCreator.add_noise_by_interpolation(clean=latent, noise=eps, sigma=degrade_sigma)

        sigma = mx.full((latent.shape[0],), degrade_sigma, dtype=mx.float32)

        # PidNet's checkpoint runs bf16 (pid_weight_mapping.py); the LQ-adapter convs need their
        # input in that dtype to matmul against the bf16 weights -- a no-op cast when latent is
        # already bf16. Matches the reference decode path, which casts only this one input (x/t/
        # caption/sigma stay float32 -- see pid_sampler.py).
        latent = latent.astype(mx.bfloat16)

        # sample() already returns x0 clipped to [-1, 1] -- the same raw range
        # VAEUtil.decode's own output has (QwenVAE.decode has no final tanh/clamp;
        # the [-1,1] -> [0,1] conversion happens downstream in
        # ImageUtil.to_image._denormalize, applied uniformly to whatever
        # `decoded` is regardless of which decoder produced it). Converting
        # here too would double-normalize.
        return sample(
            self.pid_net,
            caption_embs,
            latent,
            sigma,
            target_h=target_h,
            target_w=target_w,
            seed=seed,
        )

    @classmethod
    def from_pretrained(cls, model_path: str = PID_REPO_DEFAULT, variant: str = "qwen-image") -> "PidDecoder":
        if variant not in PID_CHECKPOINT_VARIANTS:
            raise ValueError(f"Unknown PiD variant {variant!r}; choose one of {sorted(PID_CHECKPOINT_VARIANTS)}")
        _, lq_latent_channels = PID_CHECKPOINT_VARIANTS[variant]

        # 1. PidNet: real confirmed config (Tasks 1/5/7/8/10 -- NOT PidNet's own
        #    constructor defaults, which differ), weights from `model_path`
        #    (nvidia/PiD's released checkpoint; PidNet weights only).
        pid_net = PidNet(
            patch_depth=14,
            pixel_depth=2,
            hidden_size=1536,
            pixel_hidden_size=16,
            pixel_attn_hidden_size=1152,
            num_groups=24,
            pixel_num_groups=16,
            patch_size=16,
            lq_latent_channels=lq_latent_channels,
            lq_gate_type="sigma_aware_per_token",
            lq_interval=2,
            lq_hidden_dim=1024,
            lq_conv_padding_mode="replicate",
            pit_lq_inject=True,
            sr_scale=4,
            latent_spatial_down_factor=8,
            txt_embed_dim=2304,
            txt_max_length=300,
            use_text_rope=True,
            lq_num_res_blocks=4,
            # PID_SR4X_V1PT5 (pid/_src/configs/common/defaults/net.py:66-67) -- the config the
            # released res2kto4k checkpoints were trained with. NOT PidNet's own 1024 default.
            rope_ref_h=2048,
            rope_ref_w=2048,
        )
        pth_path = PidDecoder._load_or_raise_friendly(
            "PidNet checkpoint download",
            model_path,
            lambda: PidDecoder._resolve_pid_checkpoint(model_path, variant),
        )
        pid_weights = convert_checkpoint(str(pth_path))
        # convert_checkpoint's targets are "pid_net.*" (as if nested under a parent
        # module) -- strip that prefix since `pid_net` here IS the root module.
        stripped = {key.removeprefix("pid_net."): value for key, value in pid_weights.items()}
        pid_net.update(tree_unflatten(list(stripped.items())), strict=False)
        _assert_full_weight_coverage(pid_net, stripped, label="PidNet")

        # 2. Gemma-2 caption encoder: independently-pretrained, frozen, sourced from
        #    its own HF repo (not nvidia/PiD's checkpoint -- see Task 10 finding).
        #    Reuses mflux's existing declarative WeightTarget/WeightLoader HF-safetensors
        #    loading infra (see gemma2_weight_mapping.py) rather than a bespoke loader.
        gemma2 = Gemma2Model(Gemma2Config())
        gemma2_weights = PidDecoder._load_or_raise_friendly(
            "Gemma-2 caption encoder weights download",
            GEMMA2_REPO,
            lambda: WeightLoader.load(weight_definition=Gemma2WeightDefinition, model_path=GEMMA2_REPO),
        )
        gemma2.update(gemma2_weights.components["gemma2"], strict=False)
        _assert_full_weight_coverage(gemma2, gemma2_weights.components["gemma2"], label="Gemma2Model")

        tokenizer = PidDecoder._load_or_raise_friendly(
            "Gemma-2 tokenizer download",
            GEMMA2_REPO,
            lambda: TokenizerLoader.load(
                definition=TokenizerDefinition(
                    name="gemma2",
                    hf_subdir="",
                    tokenizer_class="GemmaTokenizerFast",
                    download_patterns=[
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "special_tokens_map.json",
                        "tokenizer.model",
                    ],  # fmt: skip
                ),
                model_path=GEMMA2_REPO,
            ),
        )
        caption_encoder = PidCaptionEncoder(gemma2=gemma2, tokenizer=tokenizer)

        return cls(pid_net=pid_net, caption_encoder=caption_encoder)

    @staticmethod
    def _load_or_raise_friendly(step_label: str, repo_id: str, fn):
        """Wrap a Hugging Face download/load call with an actionable error (Finding 3,
        final integration review). `google/gemma-2-2b-it` is a *gated* repo -- a
        first-time user with no accepted license / no HF auth hits a raw GatedRepoError
        (401) traceback otherwise. Re-raises naming both required repos, their combined
        size, and (for the gated case) the exact fix."""
        try:
            return fn()
        except GatedRepoError as e:
            raise RuntimeError(
                f"{step_label} failed: '{repo_id}' is gated on Hugging Face. Accept its license at "
                f"https://huggingface.co/{repo_id}, then run `hf auth login` to authenticate, and retry.\n"
                "PidDecoder needs two Hugging Face repos, ~8GB combined: 'nvidia/PiD' (PidNet weights) "
                "and 'google/gemma-2-2b-it' (caption encoder -- gated, requires license acceptance + "
                "`hf auth login`)."
            ) from e
        except RepositoryNotFoundError as e:
            raise RuntimeError(
                f"{step_label} failed: Hugging Face repo '{repo_id}' was not found, or you don't have "
                "access to it.\nPidDecoder needs two Hugging Face repos, ~8GB combined: 'nvidia/PiD' "
                "(PidNet weights) and 'google/gemma-2-2b-it' (caption encoder -- gated, accept its license "
                "at https://huggingface.co/google/gemma-2-2b-it and run `hf auth login`)."
            ) from e
        except (HfHubHTTPError, OSError) as e:
            raise RuntimeError(
                f"{step_label} failed: could not download '{repo_id}' from Hugging Face ({e}).\n"
                "PidDecoder needs two Hugging Face repos, ~8GB combined: 'nvidia/PiD' (PidNet weights) "
                "and 'google/gemma-2-2b-it' (caption encoder -- gated, accept its license at "
                "https://huggingface.co/google/gemma-2-2b-it and run `hf auth login`). Check your network "
                "connection and retry."
            ) from e

    @staticmethod
    def _resolve_pid_checkpoint(model_path: str, variant: str) -> Path:
        local_path = Path(model_path).expanduser()
        if local_path.is_file():
            return local_path
        return Path(hf_hub_download(repo_id=model_path, filename=PID_CHECKPOINT_VARIANTS[variant][0]))
