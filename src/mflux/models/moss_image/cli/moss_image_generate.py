import argparse
from pathlib import Path

from mflux.models.moss_image.variants.txt2img.moss_image import MOSSImage


def main():
    parser = argparse.ArgumentParser(description="Generate an image with MOSS-Image (MLX).")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation.")
    parser.add_argument("--model", type=str, default="/Users/vanch/models/mlx/MOSS-Image-SFT-mlx-bf16", help="Path to MOSS-Image model directory.")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps (default: 50).")
    parser.add_argument("--guidance", type=float, default=3.0, help="CFG Guidance scale (default: 3.0).")
    parser.add_argument("--width", type=int, default=1024, help="Output image width (default: 1024).")
    parser.add_argument("--height", type=int, default=1024, help="Output image height (default: 1024).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt.")
    parser.add_argument("--step-cache-ratio", type=float, default=0.4, help="StepCache skip ratio (0 to disable, default: 0.4).")
    parser.add_argument("--output", type=str, default="output_moss.png", help="Output image path.")
    args = parser.parse_args()

    model = MOSSImage(model_path=args.model)
    ratio = args.step_cache_ratio if args.step_cache_ratio > 0 else None
    image = model.generate_image(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        step_cache_ratio=ratio,
    )
    out_path = Path(args.output)
    image.save(out_path)
    print(f"💾 Image successfully saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
