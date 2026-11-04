import json
import os
import sys
from pathlib import Path

from mflux.models.common.resolution.lora_resolution import LibraryLoRA, LoraResolution


class LoraLibraryUtil:
    @staticmethod
    def epilog() -> str:
        return """
Environment Variables:
  LORA_LIBRARY_PATH    Colon-separated list of directories containing .safetensors files
                       Example: /path/to/loras:/another/path/to/loras

Examples:
  # List all discovered LoRA files using LORA_LIBRARY_PATH
  mflux-lora-library list

  # With environment variable set
  LORA_LIBRARY_PATH=/home/user/loras:/opt/shared/loras mflux-lora-library list

  # Override with specific paths
  mflux-lora-library list --paths /path/to/loras /another/path/to/loras

  # One JSON object for each file, for scripts and other apps
  mflux-lora-library list --json

When two files have the same name, --lora <name> takes the file in the first directory, and in one directory
the first file in sorted order. To use another one, pass its full path.
"""

    @staticmethod
    def list_loras(paths: list[str] | None = None, as_json: bool = False) -> int:
        if paths:
            # Use provided paths
            library_paths = [Path(p.strip()) for p in paths]
        else:
            # Use environment variable
            library_path_env = os.environ.get("LORA_LIBRARY_PATH")

            if not library_path_env:
                print("LORA_LIBRARY_PATH environment variable is not set.", file=sys.stderr)
                print(
                    "Set it to one or more colon-separated directories containing .safetensors files.", file=sys.stderr
                )
                print("Alternatively, use --paths to specify directories directly.", file=sys.stderr)
                return 1

            # Parse library paths from environment
            library_paths = [Path(p.strip()) for p in library_path_env.split(":") if p.strip()]

        # Check which paths exist
        valid_paths = []
        for path in library_paths:
            if path.exists() and path.is_dir():
                valid_paths.append(path)
            else:
                print(f"Warning: Path does not exist or is not a directory: {path}", file=sys.stderr)

        if not valid_paths:
            print("No valid directories found in LORA_LIBRARY_PATH.", file=sys.stderr)
            return 1

        files = LoraResolution.library_files(valid_paths)
        if as_json:
            print(json.dumps([LoraLibraryUtil._as_dict(lora) for lora in files], indent=2))
            return 0

        if not files:
            print("No .safetensors files found in the specified directories.")
            return 0

        selected = {lora.name: lora for lora in files if lora.selected}
        print("Discovered LoRA files:")
        for root in valid_paths:
            in_root = [lora for lora in files if lora.root == root]
            print("-" * 80)
            print(f"{root} ({len(in_root)} file{'' if len(in_root) == 1 else 's'})")
            for lora in in_root:
                line = f"  {lora.relative_path} -> {lora.name}"
                if not lora.selected:
                    winner = selected[lora.name]
                    line += f"  (not used: --lora {lora.name} takes {winner.path})"
                print(line)
        print("-" * 80)
        print(f"\nTotal LoRA files found: {len(files)}")
        if shadowed := sorted({lora.name for lora in files if not lora.selected}):
            print(
                f"\n⚠️  More than one file has the name {', '.join(shadowed)}. "
                "Rename a file, or pass its full path to --lora."
            )
        return 0

    @staticmethod
    def _as_dict(lora: LibraryLoRA) -> dict:
        return {
            "name": lora.name,
            "path": str(lora.path),
            "root": str(lora.root),
            "relative_path": str(lora.relative_path),
            "size_bytes": lora.path.stat().st_size if lora.path.exists() else None,
            "selected": lora.selected,
        }
