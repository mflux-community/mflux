"""Lazy loader for the native ANE extension (_ane_linear, a nanobind module).

The extension is not part of the wheel. Set MFLUX_ANE_EXT_DIR to the directory
holding the built .so; the default points at the spike build in this checkout.
"""

import os
import sys
from pathlib import Path


class AneExtension:
    _module = None

    @staticmethod
    def load():
        if AneExtension._module is not None:
            return AneExtension._module
        env = os.environ.get("MFLUX_ANE_EXT_DIR")
        default = Path(__file__).resolve().parents[5] / "spikes" / "ane_matmul" / "ane_linear"
        directory = Path(env) if env else default
        if not (directory / "_ane_linear.cpython-313-darwin.so").exists() and not any(
            directory.glob("_ane_linear*.so")
        ):
            raise RuntimeError(
                f"ANE extension not found in {directory}. Build it (spikes/ane_matmul/stage3b/ops/build_extension.sh) "
                "or point MFLUX_ANE_EXT_DIR at a built copy."
            )
        sys.path.insert(0, str(directory))
        import _ane_linear  # ty: ignore[unresolved-import]

        AneExtension._module = _ane_linear
        return _ane_linear

    @staticmethod
    def available() -> bool:
        env = os.environ.get("MFLUX_ANE_EXT_DIR")
        directory = Path(env) if env else Path(__file__).resolve().parents[5] / "spikes" / "ane_matmul" / "ane_linear"
        return any(directory.glob("_ane_linear*.so"))
