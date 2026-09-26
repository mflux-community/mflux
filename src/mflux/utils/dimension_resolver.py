from pathlib import Path

from mflux.cli.defaults import defaults as ui_defaults
from mflux.utils.exif_orientation import oriented_size
from mflux.utils.scale_factor import ScaleFactor


class DimensionResolver:
    @staticmethod
    def resolve_output_dimensions(
        width: int | ScaleFactor,
        height: int | ScaleFactor,
        reference_image_path: str,
        dims_specified: bool,
    ) -> tuple[int | None, int | None]:
        """Translate the edit CLI's dimension flags into generate_image arguments.

        With no dimension flag on the command line, both map to (None, None) so the
        variant derives its ~1MP target from the last condition image's aspect ratio,
        like the reference pipeline. Once any dimension flag is given, the shared
        flags' semantics apply: scale factors resolve against the reference image
        ("1x" is the source's own displayed size) and plain integers pass through;
        an axis still on its ScaleFactor(1) default then means 1x the source size.
        """
        if not dims_specified:
            return None, None
        return DimensionResolver.resolve(width=width, height=height, reference_image_path=reference_image_path)

    @staticmethod
    def resolve(
        height: int | ScaleFactor,
        width: int | ScaleFactor,
        reference_image_path: Path | str | None = None,
    ) -> tuple[int, int]:
        height_is_scale = isinstance(height, ScaleFactor)
        width_is_scale = isinstance(width, ScaleFactor)

        # If neither dimension uses ScaleFactor, just return as-is
        if not height_is_scale and not width_is_scale:
            return int(width), int(height)

        # ScaleFactor requires a reference image - fall back to defaults if not provided
        if reference_image_path is None:
            resolved_width = ui_defaults.WIDTH if width_is_scale else int(width)
            resolved_height = ui_defaults.HEIGHT if height_is_scale else int(height)
            return resolved_width, resolved_height

        # Header-only read, and the displayed size rather than the stored one so these
        # dimensions describe the same picture ImageUtil.load_image hands the model.
        orig_width, orig_height = oriented_size(reference_image_path)

        # Resolve height
        if height_is_scale:
            resolved_height = height.get_scaled_value(orig_height)
        else:
            resolved_height = int(height)

        # Resolve width
        if width_is_scale:
            resolved_width = width.get_scaled_value(orig_width)
        else:
            resolved_width = int(width)

        return resolved_width, resolved_height
