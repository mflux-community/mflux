import os
import shutil
import subprocess
import time
from pathlib import Path

import requests

from .git_operations import GitOperations


class PyPIPublisher:
    @staticmethod
    def build_and_verify_package() -> None:
        print("📦 Building and verifying package...")

        # Clean dist directory
        dist_dir = Path("dist")
        if dist_dir.exists():
            shutil.rmtree(dist_dir)
            print("🧹 Cleaned dist/ directory")

        # Build the package. Invoke the `uv` binary from PATH (guaranteed by
        # astral-sh/setup-uv in CI); `python -m uv` requires the uv PyPI package,
        # which is not installed in the release environment.
        GitOperations.run_command(["uv", "--version"], "Verify 'uv build' version")
        GitOperations.run_command(["uv", "build"], "Building package with 'uv build'")

        # Verify the package. twine is no longer a project dependency; run it in an
        # ephemeral environment via `uv tool run` for its metadata/README checks.
        print("🔍 Verifying package...")
        GitOperations.run_command(
            ["uv", "tool", "run", "twine", "check", "dist/*"],
            "Verifying distribution with Twine",
        )

    @staticmethod
    def version_exists_on_pypi(package_name: str, version: str) -> bool:
        print("🔍 Checking if version already exists on PyPI...")
        repo_url = "https://pypi.org/pypi"
        url = f"{repo_url}/{package_name}/json"

        for attempt in range(3):
            try:
                print(f"   Checking PyPI API (attempt {attempt + 1}/3): {url}")
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    releases = data.get("releases", {})
                    version_exists = version in releases
                    print(
                        f"   PyPI API response: {len(releases)} total versions, version {version} exists: {version_exists}"
                    )

                    if version_exists:
                        print(f"⚠️  Version {version} already exists on PyPI")

                    return version_exists
                elif response.status_code == 404:
                    print("   Package not found on PyPI (404) - this is normal for new packages")
                    return False
                else:
                    print(f"   PyPI API returned status {response.status_code}: {response.text[:200]}")
                    if attempt < 2:
                        time.sleep(2**attempt)
                        continue
                    else:
                        raise ValueError(f"PyPI API returned unexpected status {response.status_code}")

            except requests.RequestException as e:
                print(f"   Network error checking PyPI (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                else:
                    raise requests.RequestException(f"Failed to check PyPI after 3 attempts: {e}") from e

        raise RuntimeError("Unexpected error in version_exists_on_pypi")

    @staticmethod
    def publish_to_pypi(pypi_token: str, package_name: str, version: str) -> None:
        PyPIPublisher._upload_to_pypi(
            token=pypi_token,
            display_name="PyPI",
            package_name=package_name,
            version=version,
        )

    @staticmethod
    def _upload_to_pypi(
        token: str,
        display_name: str,
        package_name: str,
        version: str,
    ) -> None:
        print(f"📦 Publishing to {display_name}...")

        # Only select valid distribution files (.whl and .tar.gz)
        dist_dir = Path("dist")
        dist_files = [*dist_dir.glob("*.whl"), *dist_dir.glob("*.tar.gz")]
        if not dist_files:
            raise ValueError("No distribution files found in dist/")

        print(f"📦 Uploading {len(dist_files)} files to {display_name}...")
        for file_path in dist_files:
            print(f"   • {file_path.name}")

        # --check-url lets uv skip files already uploaded (equivalent to twine's skip_existing)
        cmd = [
            "uv",
            "publish",
            "--check-url",
            f"https://pypi.org/simple/{package_name}/",
            *[str(f) for f in dist_files],
        ]
        # Token passed via environment so it never appears in logged commands
        env = {**os.environ, "UV_PUBLISH_TOKEN": token}

        # Retry transient failures (e.g. 5xx replies / connection resets)
        for attempt in range(3):
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
            if result.returncode == 0:
                print(f"✅ {display_name} upload completed successfully (attempt {attempt + 1})")
                return

            error_msg = f"{result.stdout}\n{result.stderr}".lower()
            transient = any(x in error_msg for x in ["500", "502", "503", "504", "timeout"])
            if transient and attempt < 2:
                wait = 2**attempt
                print(f"⚠️  Transient {display_name} upload error (attempt {attempt + 1}/3). Retrying in {wait}s")
                time.sleep(wait)
                continue

            # Non-transient failure (or retries exhausted): report and continue the release,
            # matching the previous twine-based behavior of treating uploads as non-critical.
            print(f"⚠️  {display_name} upload failed (exit code {result.returncode})")
            print(f"   stdout: {result.stdout.strip()}")
            print(f"   stderr: {result.stderr.strip()}")
            if "already exists" in error_msg:
                print(f"⚠️  Version {version} already exists on {display_name}")
            elif "authentication" in error_msg or "403" in error_msg:
                print(f"   This appears to be an authentication issue - check your {display_name} token.")
            else:
                print("   This might be a partial upload where some files succeeded.")
            print(f"   Check {display_name} manually to verify which files were uploaded.")
            print(f"   {display_name} failures are non-critical, continuing with release process")
            return
