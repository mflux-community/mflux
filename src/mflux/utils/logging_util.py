import logging
import sys


def _stderr_is_terminal() -> bool:
    try:
        return sys.stderr is not None and sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def configure_logging(verbose: bool = False) -> None:
    if logging.root.handlers:
        return
    handler = None
    if _stderr_is_terminal():
        try:
            from rich.console import Console
            from rich.logging import RichHandler

            handler = RichHandler(console=Console(stderr=True), rich_tracebacks=True, show_path=verbose)
        except ImportError:
            pass
    if not handler:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s" if verbose else "%(levelname)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        handlers=[handler],
    )
