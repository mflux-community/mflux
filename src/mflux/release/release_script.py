import argparse
import os
import sys

import requests

from mflux.release.release_manager import ReleaseManager


def main():
    parser = argparse.ArgumentParser(description="MFLUX Release Management")
    parser.add_argument("--github-token", help="GitHub token")
    parser.add_argument("--pypi-token", help="PyPI API token")
    parser.add_argument(
        "--trusted-publishing",
        action="store_true",
        help="Build distributions but defer PyPI upload to a trusted publisher (e.g. GitHub Actions OIDC)",
    )
    args = parser.parse_args()

    trusted_publishing = args.trusted_publishing or os.getenv("MFLUX_TRUSTED_PUBLISHING", "").lower() in {
        "1",
        "true",
        "yes",
    }

    try:
        ReleaseManager.create_release(
            github_token=args.github_token or os.getenv("GITHUB_TOKEN"),
            pypi_token=args.pypi_token or os.getenv("PYPI_API_TOKEN"),
            trusted_publishing=trusted_publishing,
        )
    except (ValueError, FileNotFoundError, requests.RequestException) as e:
        print(f"❌ Release failed: {e}")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"❌ Release failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
