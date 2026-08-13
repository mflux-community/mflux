from argparse import Namespace

import mlx.core as mx

from mflux.callbacks.instances.battery_saver import BatterySaver
from mflux.callbacks.instances.canny_saver import CannyImageSaver
from mflux.callbacks.instances.depth_saver import DepthImageSaver
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler


class CallbackManager:
    @staticmethod
    def register_callbacks(
        args: Namespace,
        model,
        latent_creator,
        enable_canny_saver: bool = False,
        enable_depth_saver: bool = False,
    ) -> MemorySaver | None:
        # Battery saver (always enabled)
        CallbackManager._register_battery_saver(args, model)

        # Specialized savers (based on flags)
        if enable_canny_saver:
            CallbackManager._register_canny_saver(args, model)

        if enable_depth_saver:
            CallbackManager._register_depth_saver(args, model)

        # Stepwise handler (if requested)
        CallbackManager._register_stepwise_handler(args, model, latent_creator)

        # Memory saver (if requested)
        return CallbackManager._register_memory_saver(args, model)

    @staticmethod
    def _register_battery_saver(args: Namespace, model) -> None:
        battery_saver = BatterySaver(battery_percentage_stop_limit=args.battery_percentage_stop_limit)
        model.callbacks.register(battery_saver)

    @staticmethod
    def _register_canny_saver(args: Namespace, model) -> None:
        if hasattr(args, "controlnet_save_canny") and args.controlnet_save_canny:
            canny_image_saver = CannyImageSaver(path=args.output)
            model.callbacks.register(canny_image_saver)

    @staticmethod
    def _register_depth_saver(args: Namespace, model) -> None:
        if hasattr(args, "save_depth_map") and args.save_depth_map:
            depth_image_saver = DepthImageSaver(path=args.output)
            model.callbacks.register(depth_image_saver)

    @staticmethod
    def _register_stepwise_handler(args: Namespace, model, latent_creator) -> None:
        if args.stepwise_image_output_dir:
            handler = StepwiseHandler(
                model=model,
                latent_creator=latent_creator,
                output_dir=args.stepwise_image_output_dir,
            )
            model.callbacks.register(handler)

    @staticmethod
    def _register_memory_saver(args: Namespace, model) -> MemorySaver | None:
        cache_limit_bytes = CallbackManager._resolve_cache_limit_bytes(getattr(args, "mlx_cache_limit_gb", None))
        seeds = getattr(args, "seed", []) or []
        num_seeds = len(seeds) if seeds else 1
        if args.low_ram:
            images = getattr(args, "image_path", [])
            if not isinstance(images, list):
                images = [images] if images is not None else []
            keep_transformer = num_seeds > 1 or len(images) > 1
            memory_saver = MemorySaver(
                model=model,
                keep_transformer=keep_transformer,
                cache_limit_bytes=cache_limit_bytes or 1000**3,
                args=args,
                num_seeds=num_seeds,
            )
        else:
            # Always evict text encoders after encoding — they are never needed post-encode
            # and keeping them wastes 8-12 GB throughout the denoising loop.
            #
            # With --pid-decode the transformer is dead weight the moment the loop ends: PidNet
            # decodes, so neither the transformer nor the VAE is touched again. Holding it is not
            # a speed/memory trade-off, it is a straight loss -- measured on Krea 2 at 512 -> 2048,
            # holding vs evicting costs +349 MB of swap and takes PiD decode from 19.47s to 25.51s,
            # because MLX's 48 GB peak demand no longer fits and the kernel starts paging. So don't
            # make it wait for --low-ram, whose other effects (VAE tiling, 1 GB cache limit) buy
            # nothing on a path that never calls the VAE.
            keep_transformer = num_seeds > 1 or not getattr(args, "pid_decode", False)
            memory_saver = MemorySaver(
                model=model, keep_transformer=keep_transformer, cache_limit_bytes=None, num_seeds=num_seeds
            )
            if cache_limit_bytes is not None:
                mx.set_cache_limit(cache_limit_bytes)
                mx.clear_cache()
                mx.reset_peak_memory()
        model.callbacks.register(memory_saver)
        return memory_saver

    @staticmethod
    def _resolve_cache_limit_bytes(mlx_cache_limit_gb: float | None) -> int | None:
        if mlx_cache_limit_gb is None:
            return None
        return int(mlx_cache_limit_gb * (1000**3))
