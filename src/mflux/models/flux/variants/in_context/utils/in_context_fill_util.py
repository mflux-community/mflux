
from mflux.utils.exif_orientation import oriented_size
from mflux.utils.prompt_util import PromptUtil


class FluxInContextFillUtil:
    @staticmethod
    def get_effective_ic_edit_prompt(args):
        if hasattr(args, "instruction") and args.instruction:
            return f"A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {args.instruction}"
        else:
            return PromptUtil.read_prompt(args)

    @staticmethod
    def resize_for_ic_edit_optimal_width(args):
        actual_width, actual_height = oriented_size(args.reference_image)
        aspect_ratio = actual_height / actual_width
        original_args_width = args.width
        original_args_height = args.height
        optimal_width = 512
        optimal_height = int(512 * aspect_ratio)
        optimal_height = (optimal_height // 8) * 8
        print(f"[INFO] IC-Edit LoRA trained on 512px width. Auto-resizing from actual image {actual_width}x{actual_height} to {optimal_width}x{optimal_height}")  # fmt:off
        print(f"[INFO] Aspect ratio maintained: {aspect_ratio:.3f}")
        if original_args_width != actual_width or original_args_height != actual_height:
            print(f"[INFO] Note: Command line args specified {original_args_width}x{original_args_height}, but using actual image dimensions for scaling")  # fmt:off
        return optimal_width, optimal_height
