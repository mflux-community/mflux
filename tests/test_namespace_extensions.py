import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.fast
class TestNamespaceExtensions:
    @pytest.mark.parametrize("parent_first", [True, False])
    @pytest.mark.parametrize("tokenizers_value", [None, "true"])
    def test_split_implicit_web_plugins(self, tmp_path: Path, parent_first: bool, tokenizers_value: str | None) -> None:
        source = Path(__file__).resolve().parents[1] / "src"
        web_root = tmp_path / "web_distribution"
        web = web_root / "mflux" / "web" / "other_ui"
        web.mkdir(parents=True)
        (web / "__init__.py").write_text("VALUE = 7\n", encoding="utf-8")
        plugin_root = tmp_path / "plugin_distribution"
        plugin = plugin_root / "mflux" / "web" / "example_ui"
        plugin.mkdir(parents=True)
        (plugin / "__init__.py").write_text("VALUE = 42\n", encoding="utf-8")
        paths = [source, web_root, plugin_root] if parent_first else [plugin_root, web_root, source]
        code = (
            f"import sys; sys.path[:0] = {[str(path) for path in paths]!r}\n"
            "import os, mflux, mflux.web, mflux.web.example_ui, mflux.web.other_ui\n"
            f"assert mflux.__file__ == {str(source / 'mflux' / '__init__.py')!r}\n"
            "assert mflux.web.example_ui.VALUE == 42\n"
            "assert mflux.web.other_ui.VALUE == 7\n"
            "assert mflux.web.__file__ is None\n"
            f"assert os.environ['TOKENIZERS_PARALLELISM'] == {tokenizers_value or 'false'!r}\n"
            "assert 'mflux.models' not in sys.modules\n"
        )
        env = os.environ.copy()
        env.pop("TOKENIZERS_PARALLELISM", None)
        if tokenizers_value is not None:
            env["TOKENIZERS_PARALLELISM"] = tokenizers_value
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", code],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == ""
