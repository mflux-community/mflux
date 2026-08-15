# Minimal escape-hatch stub: mlx.core is a compiled extension (core.*.so) that
# ships py.typed but no .pyi, so ty cannot resolve it. Everything is Any until
# mlx publishes real stubs; delete this directory when it does.
from typing import Any

def __getattr__(name: str) -> Any: ...
