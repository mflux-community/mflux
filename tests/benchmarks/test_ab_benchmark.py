import importlib.util
import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "ab_benchmark.py"
spec = importlib.util.spec_from_file_location("ab_benchmark", SCRIPT)
ab = importlib.util.module_from_spec(spec)
sys.modules["ab_benchmark"] = ab
spec.loader.exec_module(ab)

# A fake mflux command: prints a tqdm-style bar on stderr and writes the metadata sidecar next to the image.
FAKE = textwrap.dedent(
    """
    import json, sys, time
    image, delay, steps = sys.argv[1], float(sys.argv[2]), 4
    for n in range(steps + 1):
        sys.stderr.write(f"\\r{n * 100 // steps:3d}%|###| {n}/{steps} [00:00<00:00]")
        sys.stderr.flush()
        if n < steps:
            time.sleep(delay if n else delay * 3)
    open(image, "wb").write(b"png")
    open(image.rsplit(".", 1)[0] + ".metadata.json", "w").write(json.dumps({"generation_time_seconds": 1.5}))
    """
)


@pytest.fixture
def fake_cmd(tmp_path):
    script = tmp_path / "fake.py"
    script.write_text(FAKE)
    return script


class NoGuard(ab.Guard):
    def check(self):
        return None


@pytest.mark.fast
class TestParsing:
    def test_swap_units(self):
        assert ab.parse_swap_mb("total = 3072.00M  used = 60.50M  free = 1.0M") == 60.5
        assert ab.parse_swap_mb("total = 3.00G  used = 2.00G  free = 1.00G") == 2048.0
        assert ab.parse_swap_mb("used = 1024.00K") == 1.0

    def test_swap_without_a_figure_raises(self):
        with pytest.raises(ValueError):
            ab.parse_swap_mb("nothing here")

    def test_free_percent(self):
        assert ab.parse_free_percent("System-wide memory free percentage: 35%") == 35
        assert ab.parse_free_percent("no figure") is None

    def test_progress_segments_and_rest(self):
        steps, rest = ab.parse_progress(b"  0%|  | 0/8 [00:00<?]\r 12%|#| 1/8 [00:05<00:30]\r 25%|##| 2/")
        assert steps == [(0, 8), (1, 8)]
        assert rest == b" 25%|##| 2/"

    def test_progress_ignores_other_text(self):
        assert ab.parse_progress(b"Loading model\nsome warning\n")[0] == []


@pytest.mark.fast
class TestStepTimes:
    def test_first_and_steady(self):
        marks = {0: 0.0, 1: 9.0, 2: 14.0, 3: 19.5, 4: 24.5}
        first, steady = ab.step_times(marks)
        assert first == 9.0
        assert steady == 5.0

    def test_missing_marks_give_none(self):
        assert ab.step_times({1: 1.0}) == (None, None)
        assert ab.step_times({0: 0.0}) == (None, None)

    def test_one_step_has_no_steady_value(self):
        assert ab.step_times({0: 0.0, 1: 2.0}) == (2.0, None)


@pytest.mark.fast
class TestOrder:
    def test_two_arms_alternate_by_rep_and_case(self):
        arms = ["a", "b"]
        assert ab.run_order(arms, 0, 0) == ["a", "b"]
        assert ab.run_order(arms, 1, 0) == ["b", "a"]
        assert ab.run_order(arms, 0, 1) == ["b", "a"]
        assert ab.run_order(arms, 1, 1) == ["a", "b"]

    def test_every_arm_comes_first_equally_often(self):
        arms = ["a", "b", "c"]
        firsts = [ab.run_order(arms, rep, 0)[0] for rep in range(6)]
        assert {a: firsts.count(a) for a in arms} == {"a": 2, "b": 2, "c": 2}


@pytest.mark.fast
class TestSummary:
    def rows(self):
        def row(arm, wall, swap=0.0, ok=True):
            return ab.Row(arm, "c", 0, ok, wall, wall - 1, 5.0, 2.0 if arm == "fast" else 3.0, swap_grew_mb=swap)

        return [
            row("base", 30),
            row("base", 32),
            row("fast", 20),
            row("fast", 22),
            row("fast", 99, swap=500),
            row("base", 1, ok=False),
        ]

    def test_medians_ratio_and_exclusions(self):
        summary = ab.summarise(self.rows(), ["base", "fast"], discard_swap_mb=64)
        assert summary["runs_used"] == {"base": 2, "fast": 2}
        assert summary["medians"]["base"]["wall_s"] == 31
        assert summary["medians"]["fast"]["wall_s"] == 21
        assert summary["ratios_vs_first_arm"]["fast"]["wall_s"] == pytest.approx(31 / 21)
        assert summary["ratios_vs_first_arm"]["fast"]["steady_step_s"] == pytest.approx(1.5)

    def test_an_empty_arm_gives_none(self):
        summary = ab.summarise([ab.Row("base", "c", 0, True, 10.0)], ["base", "other"], 64)
        assert summary["ratios_vs_first_arm"]["other"]["wall_s"] is None

    def test_report_names_each_arm_and_left_out_runs(self):
        rows = self.rows()
        summary = ab.summarise(rows, ["base", "fast"], 64)
        text = ab.report(rows, ["base", "fast"], summary, {}, 64)
        assert "left out: swap grew" in text
        assert "| fast |" in text
        assert "above 1 means faster than 'base'" in text


@pytest.mark.fast
class TestLock:
    def test_lock_is_exclusive_and_released(self, tmp_path):
        path = tmp_path / "bench.lock"
        with ab.BenchLock(path):
            assert path.exists()
            with pytest.raises(TimeoutError):
                with ab.BenchLock(path, wait_s=0.2, poll_s=0.05):
                    pass
        assert not path.exists()

    def test_a_dead_holder_is_removed(self, tmp_path):
        path = tmp_path / "bench.lock"
        path.write_text(json.dumps({"pid": 2**22 + 12345, "at": 0}))
        with ab.BenchLock(path, wait_s=1, poll_s=0.05):
            assert json.loads(path.read_text())["pid"] == os.getpid()


@pytest.mark.fast
class TestRunOnce:
    def test_records_steps_and_returns_ok(self, tmp_path, fake_cmd):
        image = tmp_path / "x.png"
        row, _ = ab.run_once(
            [sys.executable, str(fake_cmd), str(image), "0.1"], image, NoGuard(3, 6000, 0.0, poll_s=0.05)
        )
        assert row.ok
        assert row.first_step_s == pytest.approx(0.3, abs=0.25)
        assert row.steady_step_s == pytest.approx(0.1, abs=0.1)
        assert ab.read_generation_time(image) == 1.5

    def test_the_guard_kills_the_process_group(self, tmp_path, fake_cmd):
        class Kill(ab.Guard):
            def check(self):
                return "test limit"

        image = tmp_path / "x.png"
        row, _ = ab.run_once([sys.executable, str(fake_cmd), str(image), "5"], image, Kill(3, 6000, 0.0, poll_s=0.05))
        assert not row.ok
        assert row.killed == "test limit"
        assert row.wall_s < 4

    def test_a_failing_command_is_not_ok(self, tmp_path):
        row, tail = ab.run_once(
            [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
            tmp_path / "x.png",
            NoGuard(3, 6000, 0.0, poll_s=0.05),
        )
        assert not row.ok
        assert "boom" in tail


@pytest.mark.fast
class TestMain:
    def test_two_arms_end_to_end(self, tmp_path, fake_cmd, monkeypatch, capsys):
        monkeypatch.setattr(ab, "swap_mb", lambda: 0.0)
        monkeypatch.setattr(ab, "free_percent", lambda: 90)
        out = tmp_path / "out"
        py = sys.executable
        code = ab.main(
            [
                "--arm", f"slow={py} {fake_cmd} {{image}} 0.2",
                "--arm", f"fast={py} {fake_cmd} {{image}} 0.05",
                "--case", "one=",
                "--reps", "2",
                "--out-dir", str(out),
                "--no-lock",
            ]
        )  # fmt: skip
        assert code == 0
        rows = json.loads((out / "rows.json").read_text())
        assert len(rows) == 4
        assert [r["arm"] for r in rows if r["rep"] == 0][0] != [r["arm"] for r in rows if r["rep"] == 1][0]
        assert "| fast |" in (out / "report.md").read_text()
        assert all(r["gen_s"] == 1.5 for r in rows)

    def test_an_arm_without_the_image_placeholder_is_refused(self, tmp_path):
        with pytest.raises(SystemExit):
            ab.main(["--arm", "a=echo hi", "--out-dir", str(tmp_path), "--no-lock"])

    def test_a_malformed_arm_is_refused(self, tmp_path):
        with pytest.raises(SystemExit):
            ab.main(["--arm", "nonsense", "--out-dir", str(tmp_path), "--no-lock"])
