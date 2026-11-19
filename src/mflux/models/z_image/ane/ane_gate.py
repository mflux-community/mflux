"""--ane run gating: OS and chip support, weight source, and the Background-QoS warning."""

import ctypes
import ctypes.util
import platform
import re
import warnings


class AneGate:
    QOS_BACKGROUND = 0x09  # anything at or below this number is background/maintenance class

    @staticmethod
    def check_or_raise(
        bits: int | None,
        precision_name: str | None,
        float32: bool,
        allow_m5: bool = False,
        model_type: str | None = None,
    ) -> None:
        if platform.system() != "Darwin":
            raise ValueError("--ane runs Core ML models on the Apple Neural Engine and needs macOS.")
        major = AneGate._macos_major()
        if major is not None and major < 26:
            raise ValueError(f"--ane needs macOS 26 or newer (found {platform.mac_ver()[0]}).")
        if model_type is not None and "z-image" not in model_type and "Z-Image" not in model_type:
            raise ValueError("--ane is implemented for the Z-Image transformer only.")
        if float32:
            raise ValueError("--ane cannot be combined with --float32 (the split halves run in bfloat16).")
        if precision_name is not None:
            raise ValueError(
                "--ane cannot be combined with --compute-precision yet (the split halves run in bfloat16)."
            )
        if bits not in (None, 8):
            raise ValueError("ANE mode needs a bf16 or 8-bit model: use no --quantize or -q 8.")

        chip = AneGate._chip_name().lower()
        gen = re.search(r"apple m(\d+)", chip)
        if gen:
            number = int(gen.group(1))
            if number <= 2:
                raise ValueError(f"--ane refuses this machine ({AneGate._chip_name()}): the ANE path is M3 or newer.")
            if number == 5 and not allow_m5:
                raise ValueError(
                    f"{AneGate._chip_name()} detected: the ANE split is tuned on M3/M4. "
                    "Pass --ane-allow-m5 once you have confirmed it works on your M5."
                )

    @staticmethod
    def warn_if_background_qos() -> None:
        # A launchd Background session runs the ANE downclocked (Stage 3a: 2.5 W vs 9.3 W, ~1.6x slower).
        try:
            lib = ctypes.CDLL(ctypes.util.find_library("system"))
            lib.qos_class_self.restype = ctypes.c_uint
            qos = lib.qos_class_self()
        except (OSError, AttributeError):
            return
        if 0 < qos <= AneGate.QOS_BACKGROUND:
            warnings.warn(
                "--ane: this process runs at background QoS (a launchd or daemon session). macOS throttles the "
                "Apple Neural Engine for such sessions; run mflux in a terminal foreground for full speed.",
                stacklevel=3,
            )

    @staticmethod
    def _macos_major() -> int | None:
        try:
            return int(platform.mac_ver()[0].split(".")[0])
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _chip_name() -> str:
        import subprocess

        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            return ""
