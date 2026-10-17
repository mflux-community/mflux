import importlib.util
import json
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.fast
path = Path(__file__).parents[2] / "scripts/flux3/benchmark_memory.py"
spec = importlib.util.spec_from_file_location("flux3_memory_benchmark", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MemoryBenchmark = module.MemoryBenchmark


class TestMemoryBenchmark:
    @staticmethod
    def backend():
        return Mock(
            get_peak_memory=Mock(return_value=120),
            get_active_memory=Mock(return_value=70),
            get_cache_memory=Mock(return_value=40),
        )

    def test_peak_before_reset_survives_lower_later_peak(self):
        mx = self.backend()
        benchmark = MemoryBenchmark(mx)
        reset = Mock(side_effect=lambda: setattr(mx.get_peak_memory, "return_value", 0))
        benchmark.reset_wrapper(reset)()
        mx.get_peak_memory.return_value = 80
        benchmark.snapshot("finished")
        assert max(benchmark.peaks) == 120
        assert benchmark.events[0]["label"] == "before_peak_reset"
        reset.assert_called_once_with()

    def test_wrapped_static_method_restored_after_exception(self):
        class Component:
            @staticmethod
            def load():
                raise ValueError("failed load")

        original = Component.__dict__["load"]
        benchmark = MemoryBenchmark(self.backend())
        with pytest.raises(ValueError, match="failed load"), ExitStack() as stack:
            benchmark.instrument(stack, Component, "load", "load", static=True)
            Component.load()
        assert Component.__dict__["load"] is original
        assert benchmark.phases == []

    def test_lazy_result_materialised_before_phase_end(self):
        order = []
        result = object()

        class Component:
            @staticmethod
            def encode():
                order.append("encode")
                return result

        mx = self.backend()
        mx.eval.side_effect = lambda value: order.append("eval")
        mx.synchronize.side_effect = lambda: order.append("sync")
        benchmark = MemoryBenchmark(mx)
        with ExitStack() as stack:
            benchmark.instrument(stack, Component, "encode", "encode", static=True, evaluate=True)
            assert Component.encode() is result
        assert order == ["sync", "encode", "eval", "sync", "sync"]
        mx.eval.assert_called_once_with(result)
        assert benchmark.phases[0]["label"] == "encode"

    def test_failed_save_not_counted_as_success(self):
        class Image:
            def save(self):
                raise OSError("disk full")

        benchmark = MemoryBenchmark(self.backend())
        with pytest.raises(OSError, match="disk full"), ExitStack() as stack:
            benchmark.instrument(stack, Image, "save", "save")
            Image().save()
        assert benchmark.saved_images == 0

    @pytest.mark.parametrize("saved,success", [(0, False), (1, False), (2, True)])
    @pytest.mark.parametrize("checkout", [False, True])
    def test_main_reports_completion_and_imported_source(self, monkeypatch, tmp_path, saved, success, checkout):
        report = tmp_path / "report.json"
        source = tmp_path / "installed" / "flux3_generate.py"
        source.parent.mkdir()
        source.touch()
        expected_sha = None
        if checkout:
            for arguments in (
                ["init", "--quiet"],
                [
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "--quiet",
                    "--allow-empty",
                    "-m",
                    "fixture",
                ],
            ):
                subprocess.run(["git", "-C", str(source.parent), *arguments], check=True)
            expected_sha = subprocess.check_output(
                ["git", "-C", str(source.parent), "rev-parse", "HEAD"], text=True
            ).strip()

        class Parser:
            def parse_args(self):
                return SimpleNamespace(seed=[42, 43])

        original_parse = Parser.parse_args
        image = SimpleNamespace(save=Mock())

        def generate():
            Parser().parse_args()
            for _ in range(saved):
                image.save()
            # The real CLI catches an interrupted generation and returns normally.

        cli = SimpleNamespace(__file__=str(source), CommandLineParser=Parser, main=generate)
        mx = self.backend()
        mx.device_info.return_value = {"device_name": "test"}
        mlx = ModuleType("mlx")
        mlx.core = mx
        replacements = {
            "mlx": mlx,
            "mlx.core": mx,
            "mflux.callbacks.instances.memory_saver": SimpleNamespace(
                MemorySaver=SimpleNamespace(call_before_loop=Mock(), call_after_loop=Mock())
            ),
            "mflux.models.flux3.cli": SimpleNamespace(flux3_generate=cli),
            "mflux.models.flux3.flux3_initializer": SimpleNamespace(Flux3Initializer=SimpleNamespace(init=Mock())),
            "mflux.models.flux3.variants.flux3_klein_core": SimpleNamespace(
                _Flux3KleinCore=SimpleNamespace(_encode_prompt=Mock(), _decode_latents=Mock())
            ),
            "mflux.utils.generated_image": SimpleNamespace(GeneratedImage=image),
        }
        for name, replacement in replacements.items():
            monkeypatch.setitem(sys.modules, name, replacement)
        monkeypatch.setattr(sys, "argv", ["benchmark", "--report", str(report), "--", "--auto-seeds", "2"])
        monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "test")
        if success:
            MemoryBenchmark.main()
        else:
            with pytest.raises(RuntimeError, match=f"saved {saved} of 2"):
                MemoryBenchmark.main()
        result = json.loads(report.read_text())
        assert result["success"] is success
        assert result["saved_images"] == saved
        assert result["expected_images"] == 2
        assert result["source_path"] == str(source)
        assert result["source_sha"] == expected_sha
        assert result["source_sha"] != result["benchmark_source_sha"]
        assert (
            result["benchmark_source_sha"]
            == subprocess.check_output(["git", "-C", str(path.parent), "rev-parse", "HEAD"], text=True).strip()
        )
        assert ("error" in result) is not success
        assert Parser.parse_args is original_parse
