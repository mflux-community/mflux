import time
from pathlib import Path

import mlx.core as mx
from PIL import Image

from mflux.models.common.step_cache.step_cache import StepCache
from mflux.models.moss_image.latent_creator.moss_image_latent_creator import (
    MOSSImageLatentCreator,
)
from mflux.models.moss_image.model.moss_image_scheduler.moss_image_scheduler import (
    MOSSImageScheduler,
)
from mflux.models.moss_image.moss_image_initializer import MOSSImageInitializer
from mflux.utils.image_util import ImageUtil


class MOSSImage:
    def __init__(
        self,
        model_path: str | Path,
        quantize: int | None = None,
    ):
        self.model_path = Path(model_path)
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.scheduler = MOSSImageScheduler()

        MOSSImageInitializer.init(
            self,
            model_path=self.model_path,
            quantize=quantize,
        )

    def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        seed: int = 42,
        negative_prompt: str = "",
        uncond_skip_layers: tuple[int, ...] = (22,),
        step_cache_ratio: float | None = 0.4,
    ) -> Image.Image:
        t0 = time.time()
        cache_str = f"StepCache ratio={step_cache_ratio}" if step_cache_ratio else "no cache"
        print(f"\n🎨 [MOSS-Image] 开始生图: 尺寸={width}x{height}, 步数={num_inference_steps}, CFG={guidance_scale}, 种子={seed}, 加速={cache_str}", flush=True)

        # 1. Encode text
        print("📝 [文本编码] 正在提取提示词特征 ...", flush=True)
        t_enc = time.time()
        positive_embeds = self.text_encoder.encode_prompt(prompt)

        do_cfg = guidance_scale != 1.0
        negative_embeds = None
        if do_cfg:
            negative_embeds = self.text_encoder.encode_prompt(negative_prompt)
        print(f"✅ [文本编码完成] 耗时: {time.time()-t_enc:.2f}s", flush=True)

        # 2. Setup Scheduler & Latents
        timesteps = self.scheduler.set_timesteps(num_inference_steps)
        latents = MOSSImageLatentCreator.create_latents(1, height, width, seed)

        num_steps = len(timesteps)

        # Build StepCache
        skip_steps = frozenset()
        if step_cache_ratio is not None and 0 < step_cache_ratio < 1 and num_steps >= 10:
            skip_steps = StepCache.select_skip_steps(
                steps=list(range(num_steps)),
                sigmas=self.scheduler.sigmas,
                ratio=step_cache_ratio
            )
            print(f"⚡ [StepCache 就绪] 共 {num_steps} 步，已启用特征复用，将自动跳过 {len(skip_steps)} 步计算！", flush=True)
        step_cache = StepCache(skip_steps)

        # 3. Denoising loop
        t_loop_start = time.time()
        for step_idx in range(num_steps):
            step_t0 = time.time()
            t = timesteps[step_idx]

            reused_pred = step_cache.reuse(step_idx)
            if reused_pred is not None:
                pred = reused_pred
                is_skipped = True
            else:
                is_skipped = False
                # Positive forward
                pred = self.transformer(latents, t, positive_embeds)

                if do_cfg:
                    # Negative forward with skip layer 22
                    uncond = self.transformer(latents, t, negative_embeds, skip_layers=uncond_skip_layers)
                    pred = uncond + guidance_scale * (pred - uncond)

                step_cache.store(pred)

            latents = self.scheduler.step(pred, step_idx, latents)
            mx.eval(latents)

            step_sec = time.time() - step_t0
            status = "⏩ [跳步复用]" if is_skipped else f"{step_sec:.2f}s"
            if (step_idx + 1) % 5 == 0 or step_idx == num_steps - 1 or step_idx < 3:
                print(f"   第 {step_idx + 1:2d}/{num_steps} 步完成 ({status})", flush=True)

        loop_time = time.time() - t_loop_start
        print(f"✅ [去噪完成] 纯采样总耗时: {loop_time:.2f}s (平均 {loop_time/num_steps:.2f} s/步)", flush=True)

        # 4. Decode with FLUX.2 VAE
        print("🖼️ [VAE解码] 正在将潜变量解码为高质量图像 ...", flush=True)
        t_vae = time.time()
        decoded = self.vae.decode_packed_latents(latents)
        mx.eval(decoded)
        print(f"✅ [VAE解码完成] 耗时: {time.time() - t_vae:.2f}s", flush=True)

        image = ImageUtil.to_pil(decoded)
        total_time = time.time() - t0
        print(f"🎉 [生图完成] 总耗时: {total_time:.2f}s!", flush=True)
        return image
