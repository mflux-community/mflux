from __future__ import annotations

import gc
import random
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.optimizers import clip_grad_norm
from mlx.utils import tree_map, tree_unflatten
from tqdm import tqdm

from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.lora.layer.fused_linear_lora_layer import FusedLoRALinear
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.training.adapters.base import TrainingAdapter
from mflux.models.common.training.dataset.batch import Batch, DataItem
from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.state.training_state import TrainingState
from mflux.models.common.training.statistics.plotter import Plotter
from mflux.models.common.training.utils import TrainingUtil
from mflux.utils.exif_orientation import oriented_size


class TrainingTrainer:
    @staticmethod
    def _sample_timestep_index(timestep_type: str, low: int, high: int, rng) -> int:
        # Map a [0,1) draw shaped by the distribution to a timestep index in [low, high).
        import math

        span = high - low
        if span <= 0:
            return low
        if timestep_type == "sigmoid":  # mid-concentrated (ai-toolkit default; best for identity)
            frac = 1.0 / (1.0 + math.exp(-rng.gauss(0.0, 1.0)))
        elif timestep_type == "content":  # cubic, favors low noise (fine detail)
            frac = rng.random() ** 3
        elif timestep_type == "style":  # favors high noise (coarse style)
            frac = 1.0 - rng.random() ** 3
        else:
            frac = rng.random()
        return min(max(low + int(frac * span), low), high - 1)

    @staticmethod
    def compute_loss(
        adapter: TrainingAdapter,
        training_spec: TrainingSpec,
        base_config,
        batch: Batch,
    ) -> mx.float16:
        losses = [
            TrainingTrainer._single_example_loss(adapter, training_spec, base_config, item, batch.rng)
            for item in batch.data
        ]
        return mx.mean(mx.array(losses))

    @staticmethod
    def _single_example_loss(
        adapter: TrainingAdapter,
        training_spec: TrainingSpec,
        base_config,
        item: DataItem,
        rng: random.Random,
    ) -> mx.float16:
        # Create a config matching this item's spatial dimensions.
        # Flux uses config.width/height for rotary embeddings, so this must match the latent layout.
        config = adapter.create_config(training_spec, width=item.width, height=item.height)

        # Reuse the base scheduler only when compatible with the item's dimensions.
        # Some schedulers depend on image seq len when sigma shift is enabled.
        if not config.model_config.requires_sigma_shift or config.image_seq_len == base_config.image_seq_len:
            config._scheduler = base_config.scheduler  # type: ignore[attr-defined]
        else:
            _ = config.scheduler

        time_seed = rng.randint(0, 2**32 - 1)
        noise_seed = rng.randint(0, 2**32 - 1)

        low = int(training_spec.training_loop.timestep_low)
        high = int(
            config.num_inference_steps
            if training_spec.training_loop.timestep_high is None
            else training_spec.training_loop.timestep_high
        )

        timestep_type = training_spec.training_loop.timestep_type
        if timestep_type and timestep_type != "uniform":
            # Non-uniform timestep-index sampling (sigmoid/content/style). Identity learning
            # lives in the mid/low-noise band that flat sampling under-weights.
            t = TrainingTrainer._sample_timestep_index(timestep_type, low, high, rng)
        else:
            t = int(
                mx.random.randint(
                    low=low,
                    high=high,
                    shape=[],
                    key=mx.random.key(time_seed),
                )
            )

        clean_image = item.clean_latents
        pure_noise = mx.random.normal(
            shape=clean_image.shape,
            dtype=config.precision,
            key=mx.random.key(noise_seed),
        )

        latents_t = LatentCreator.add_noise_by_interpolation(
            clean=clean_image,
            noise=pure_noise,
            sigma=config.scheduler.sigmas[t],
        )

        predicted_noise = adapter.predict_noise(
            t=t,
            latents_t=latents_t,
            sigmas=config.scheduler.sigmas,
            cond=item.cond,
            config=config,
        )

        error = (clean_image + predicted_noise - pure_noise).square()
        return error.mean()

    @staticmethod
    def train(
        *,
        adapter: TrainingAdapter,
        training_spec: TrainingSpec,
        training_state: TrainingState,
    ) -> None:
        first_preview = None
        if training_spec.monitoring is not None and training_spec.monitoring.preview_images:
            first_preview = training_spec.monitoring.preview_images[0]
        preview_width, preview_height = TrainingTrainer._preview_dimensions(training_spec, preview_image=first_preview)
        base_config = adapter.create_config(training_spec, width=preview_width, height=preview_height)
        # Ensure scheduler is initialized once and can be reused in per-item configs.
        _ = base_config.scheduler

        # Freeze base weights and unfreeze LoRA weights
        adapter.freeze_base()
        TrainingTrainer._unfreeze_lora_layers(adapter.transformer())

        train_step_function = nn.value_and_grad(
            model=adapter.model(),
            fn=lambda b: TrainingTrainer.compute_loss(adapter, training_spec, base_config, b),
        )

        if training_spec.monitoring is not None and training_state.iterator.num_iterations == 0:
            TrainingTrainer._generate_previews_with_optimizer_offload(adapter, training_spec, training_state)
            validation_batch = training_state.iterator.get_validation_batch()
            validation_loss = TrainingTrainer.compute_loss(adapter, training_spec, base_config, validation_batch)
            training_state.statistics.append_values(step=training_state.iterator.num_iterations, loss=float(validation_loss))  # fmt: off
            Plotter.update_loss_plot(training_spec=training_spec, training_state=training_state)
            del validation_loss
            training_state.save(adapter, training_spec)

        batches = tqdm(
            training_state.iterator,
            total=training_state.iterator.total_number_of_steps(),
            initial=training_state.iterator.num_iterations,
        )

        max_grad_norm = training_spec.optimizer.max_grad_norm
        accum_steps = max(1, training_spec.optimizer.gradient_accumulation_steps)
        accumulated_grads = None
        accumulated_count = 0
        nonfinite_skips = 0
        for batch in batches:
            loss, grads = train_step_function(batch)
            if not TrainingTrainer._step_is_finite(loss):
                del loss, grads
                nonfinite_skips += 1
                # Drop any partial accumulation window: a skipped micro-batch (especially on a
                # window boundary) must not carry its accumulated grads into the next window,
                # which would apply an oversized optimizer step.
                accumulated_grads, accumulated_count = None, 0
                if training_spec.low_ram:
                    mx.clear_cache()
                continue
            del loss

            # Gradient accumulation: average grads across accum_steps micro-batches and only step
            # the optimizer on the window boundary, for an effective batch of batch_size *
            # accum_steps.
            at_step_boundary = True
            if accum_steps > 1:
                grads, accumulated_count, at_step_boundary = TrainingTrainer._fold_into_window(
                    grads, accumulated_grads, accum_steps, accumulated_count
                )
                accumulated_grads = None if at_step_boundary else grads

            if at_step_boundary:
                if max_grad_norm is not None:
                    grads, _ = clip_grad_norm(grads, max_grad_norm)
                training_state.optimizer.optimizer.update(model=adapter.model(), gradients=grads)
                mx.eval(adapter.model().parameters(), training_state.optimizer.optimizer.state)
            else:
                # Keep the partial sum materialized so the graph doesn't grow across the window.
                mx.eval(accumulated_grads)
            del grads

            if training_state.should_plot_loss(training_spec):
                validation_batch = training_state.iterator.get_validation_batch()
                validation_loss = TrainingTrainer.compute_loss(adapter, training_spec, base_config, validation_batch)
                training_state.statistics.append_values(step=training_state.iterator.num_iterations, loss=float(validation_loss))  # fmt: off
                Plotter.update_loss_plot(training_spec=training_spec, training_state=training_state)
                del validation_loss

            if training_state.should_generate_image(training_spec):
                TrainingTrainer._generate_previews_with_optimizer_offload(adapter, training_spec, training_state)

            if training_state.should_save(training_spec):
                training_state.save(adapter, training_spec)

            if training_spec.low_ram:
                mx.clear_cache()

        if nonfinite_skips:
            print(f"Skipped {nonfinite_skips} non-finite (NaN/Inf) training step(s).")
        training_state.save(adapter, training_spec)

    @staticmethod
    def _unfreeze_lora_layers(module: nn.Module) -> None:
        for _, child in module.named_modules():
            if isinstance(child, LoRALinear):
                if getattr(child, "_mflux_lora_role", None) == "train":
                    child.unfreeze(keys=["lora_A", "lora_B"], strict=False)
            elif isinstance(child, FusedLoRALinear):
                for lora in child.loras:
                    if getattr(lora, "_mflux_lora_role", None) == "train":
                        lora.unfreeze(keys=["lora_A", "lora_B"], strict=False)

    @staticmethod
    def _preview_dimensions(training_spec: TrainingSpec, *, preview_image: Path | None = None) -> tuple[int, int]:
        if training_spec.monitoring is None:
            return 1024, 1024
        if preview_image is not None:
            width, height = oriented_size(preview_image)
        else:
            width = int(training_spec.monitoring.preview_width)
            height = int(training_spec.monitoring.preview_height)

        return TrainingUtil.resolve_dimensions(
            width=width,
            height=height,
            max_resolution=None,
            error_template=(
                f"Preview image too small for training (needs >=16px). Got {{width}}x{{height}} from {preview_image}"
            ),
        )

    @staticmethod
    def _generate_previews(
        adapter: TrainingAdapter,
        training_spec: TrainingSpec,
        training_state: TrainingState,
    ) -> None:
        if training_spec.monitoring is None:
            return
        preview_prompts = training_spec.monitoring.preview_prompts
        preview_names = training_spec.monitoring.preview_prompt_names
        preview_images = training_spec.monitoring.preview_images
        for idx, prompt in enumerate(preview_prompts):
            image_paths = None
            if training_spec.is_edit:
                if not preview_images or idx >= len(preview_images):
                    raise ValueError("Edit training requires data/preview.* for each preview prompt.")
                image_paths = [preview_images[idx]]
                preview_width, preview_height = TrainingTrainer._preview_dimensions(
                    training_spec, preview_image=preview_images[idx]
                )
            else:
                preview_width, preview_height = TrainingTrainer._preview_dimensions(training_spec)
            image = adapter.generate_preview_image(
                seed=training_spec.seed,
                prompt=prompt,
                width=preview_width,
                height=preview_height,
                steps=training_spec.steps,
                image_paths=image_paths,
            )
            preview_name = preview_names[idx] if idx < len(preview_names) else None
            image.save(
                training_state.get_current_preview_image_path(
                    training_spec,
                    preview_index=idx,
                    preview_name=preview_name,
                )
            )
            del image

    @staticmethod
    def _fold_into_window(grads, accumulated, accum_steps: int, count: int):
        """Fold one valid micro-batch into the accumulation window.

        The window closes after accum_steps VALID micro-batches rather than after
        accum_steps iterations. Counting iterations closes it early whenever a
        non-finite step reset the window mid-way, and the optimizer then steps on a
        partial sum that was still divided by the full accum_steps: a smaller update
        than either the accumulated or the unaccumulated setting asks for.
        """
        scaled = tree_map(lambda g: g / accum_steps, grads)
        if accumulated is not None:
            scaled = tree_map(lambda a, g: a + g, accumulated, scaled)
        count += 1
        if count >= accum_steps:
            return scaled, 0, True
        return scaled, count, False

    @staticmethod
    def _step_is_finite(loss) -> bool:
        """A non-finite loss (bf16 activation spikes, a NaN from one bad batch) must never
        reach optimizer.update: the gradients it came with poison the LoRA weights and the
        optimizer moments in a single step. The caller skips the step and the run continues
        from the last good state. clip_grad_norm handles ordinary spikes; this catches
        Inf/NaN."""
        return bool(mx.isfinite(loss).item())

    @staticmethod
    def _generate_previews_with_optimizer_offload(
        adapter: TrainingAdapter,
        training_spec: TrainingSpec,
        training_state: TrainingState,
    ) -> None:
        optimizer = training_state.optimizer
        with tempfile.TemporaryDirectory() as tmp_dir:
            offload_path = Path(tmp_dir) / "optimizer_offload.safetensors"
            optimizer.save(offload_path)
            optimizer.optimizer.state = []

            gc.collect()
            mx.clear_cache()
            try:
                TrainingTrainer._generate_previews(adapter, training_spec, training_state)
            finally:
                restored_state = tree_unflatten(list(mx.load(str(offload_path)).items()))
                optimizer.optimizer.state = restored_state
                gc.collect()
                mx.clear_cache()
