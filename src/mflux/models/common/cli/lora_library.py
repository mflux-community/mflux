import argparse
import sys

from mflux.utils.lora_library_util import LoraLibraryUtil


def main():
    parser = argparse.ArgumentParser(
        description="MFLUX LoRA Library management tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=LoraLibraryUtil.epilog(),
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    list_parser = subparsers.add_parser("list", help="List all discovered LoRA files")
    list_parser.add_argument("--paths", nargs="+", help="Override LORA_LIBRARY_PATH with these directories (space-separated)")  # fmt: off
    list_parser.add_argument("--json", action="store_true", help="Print one JSON object for each file (name, path, root, relative_path, size_bytes, selected)")  # fmt: off
    args = parser.parse_args()

    if args.command == "list":
        return LoraLibraryUtil.list_loras(paths=args.paths, as_json=args.json)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
