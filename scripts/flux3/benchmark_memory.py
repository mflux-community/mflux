import argparse
import functools
import importlib.metadata
import json
import platform
import resource
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

START = time.perf_counter()


class MemoryBenchmark:
    def __init__(self, mx):
        self.mx = mx
        self.events = []
        self.phases = []
        self.peaks = []
        self.saved_images = 0
        self.expected_images = None

    def parse_wrapper(self, original):
        @functools.wraps(original)
        def parse(*args, **kwargs):
            parsed = original(*args, **kwargs)
            self.expected_images = len(parsed.seed)
            return parsed

        return parse

    def check_complete(self) -> None:
        if not self.expected_images or self.saved_images != self.expected_images:
            raise RuntimeError(f"CLI saved {self.saved_images} of {self.expected_images} expected images")

    def snapshot(self, label: str) -> None:
        self.mx.synchronize()
        peak = self.mx.get_peak_memory()
        self.peaks.append(peak)
        self.events.append(
            {
                "label": label,
                "elapsed_s": time.perf_counter() - START,
                "mlx_active_bytes": self.mx.get_active_memory(),
                "mlx_cache_bytes": self.mx.get_cache_memory(),
                "mlx_peak_since_reset_bytes": peak,
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )

    def reset_wrapper(self, original):
        def reset():
            self.snapshot("before_peak_reset")
            return original()

        return reset

    def instrument(self, stack: ExitStack, cls, name: str, label: str, *, static=False, evaluate=False) -> None:
        original = getattr(cls, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            self.snapshot(label + "_start")
            started = time.perf_counter()
            result = original(*args, **kwargs)
            if evaluate:
                self.mx.eval(result)
            self.mx.synchronize()
            self.phases.append({"label": label, "duration_s": time.perf_counter() - started})
            self.snapshot(label + "_end")
            if label == "save":
                self.saved_images += 1
            return result

        stack.enter_context(patch.object(cls, name, staticmethod(wrapped) if static else wrapped))

    @staticmethod
    def command_output(command: list[str]) -> str | None:
        try:
            return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(
            description="Measure one FLUX 3 CLI process. Pass generation arguments after --."
        )
        parser.add_argument("--report", required=True, type=Path)
        parser.add_argument("cli_args", nargs=argparse.REMAINDER)
        args = parser.parse_args()
        cli_args = args.cli_args[1:] if args.cli_args[:1] == ["--"] else args.cli_args
        if not cli_args:
            parser.error("Pass FLUX 3 generation arguments after --")
        if platform.system() != "Darwin":
            parser.error("This benchmark reports macOS RSS in bytes and requires Apple Silicon")
        if args.report.exists():
            parser.error("Report already exists; choose a new path for each process")
        args.report.parent.mkdir(parents=True, exist_ok=True)

        import mlx.core as mx

        from mflux.callbacks.instances.memory_saver import MemorySaver
        from mflux.models.flux3.cli import flux3_generate
        from mflux.models.flux3.flux3_initializer import Flux3Initializer
        from mflux.models.flux3.variants.flux3_klein_core import _Flux3KleinCore
        from mflux.utils.generated_image import GeneratedImage

        benchmark = MemoryBenchmark(mx)
        source_path = Path(flux3_generate.__file__).resolve()
        result = {
            "schema_version": 1,
            "source_path": str(source_path),
            "source_sha": benchmark.command_output(["git", "-C", str(source_path.parent), "rev-parse", "HEAD"]),
            "benchmark_source_sha": benchmark.command_output(
                ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"]
            ),
            "platform": platform.platform(),
            "device": mx.device_info(),
            "versions": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-metal", "mflux")},
            "cli_args": cli_args,
            "swap_before": benchmark.command_output(["sysctl", "vm.swapusage"]),
            "pressure_before": benchmark.command_output(["sysctl", "kern.memorystatus_vm_pressure_level"]),
            "success": False,
            "notes": [
                "Python wall excludes interpreter and uv startup; measure external process wall separately.",
                "Synchronised phase boundaries perturb timing; use the same wrapper for all compared runs.",
                "Construction is lazy; SSD reads and weight materialisation also occur in later phases.",
                "MLX peak is the maximum across observed reset segments; RSS is a separate macOS metric.",
                "System swap and pressure include other processes; filesystem cache is uncontrolled.",
                "Reports contain local paths and prompts; review before sharing.",
            ],
        }
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(sys, "argv", ["mflux-generate-flux3", *cli_args]))
                stack.enter_context(
                    patch.object(
                        flux3_generate.CommandLineParser,
                        "parse_args",
                        benchmark.parse_wrapper(flux3_generate.CommandLineParser.parse_args),
                    )
                )
                stack.enter_context(
                    patch.object(mx, "reset_peak_memory", benchmark.reset_wrapper(mx.reset_peak_memory))
                )
                benchmark.instrument(stack, Flux3Initializer, "init", "construct", static=True)
                benchmark.instrument(stack, _Flux3KleinCore, "_encode_prompt", "encode", evaluate=True)
                benchmark.instrument(stack, _Flux3KleinCore, "_decode_latents", "decode", evaluate=True)
                benchmark.instrument(stack, MemorySaver, "call_before_loop", "before_loop")
                benchmark.instrument(stack, MemorySaver, "call_after_loop", "after_loop")
                benchmark.instrument(stack, GeneratedImage, "save", "save")
                benchmark.snapshot("ready")
                flux3_generate.main()
                benchmark.check_complete()
                result["success"] = True
        except BaseException as exc:
            result["error"] = repr(exc)
            raise
        finally:
            benchmark.snapshot("finished")
            result.update(
                python_wall_s=time.perf_counter() - START,
                phases=benchmark.phases,
                events=benchmark.events,
                saved_images=benchmark.saved_images,
                expected_images=benchmark.expected_images,
                mlx_lifetime_peak_bytes=max(benchmark.peaks),
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                swap_after=benchmark.command_output(["sysctl", "vm.swapusage"]),
                pressure_after=benchmark.command_output(["sysctl", "kern.memorystatus_vm_pressure_level"]),
            )
            args.report.write_text(json.dumps(result, indent=2) + "\n")
            print(f"Benchmark report: {args.report}")


if __name__ == "__main__":
    MemoryBenchmark.main()
