import os

import requests

from mflux.release.changelog_parser import ChangelogParser
from mflux.release.git_operations import GitOperations
from mflux.release.github_api import GitHubAPI
from mflux.release.pypi_publisher import PyPIPublisher
from mflux.release.release_validator import ReleaseValidator
from mflux.utils.version_util import VersionUtil


class ReleaseManager:
    @staticmethod
    def create_release(
        github_token: str,
        pypi_token: str | None,
        github_repo: str | None = None,
        package_name: str = "mflux",
        trusted_publishing: bool = False,
    ) -> None:
        github_repo = github_repo or os.getenv("GITHUB_REPOSITORY", "filipstrand/mflux")
        # 0. Load version from pyproject.toml
        version = VersionUtil.get_mflux_version()
        tag_name = f"v.{version}"
        print("🚀 Starting MFLUX release process...")
        print(f"📦 Releasing version: {version} (tag: {tag_name}) [from pyproject.toml]")

        # 1. Validate everything is ready for release
        ReleaseValidator.validate_release_ready(version)

        # 2. Check current release state
        git_tag_exists = GitOperations.check_tag_exists(tag_name)
        github_release_exists = GitHubAPI.check_github_release_exists(github_token, github_repo, tag_name)

        # 3. Print release status and exit early if already complete
        ReleaseManager._print_release_status(tag_name, git_tag_exists, github_release_exists)
        if ReleaseManager._is_release_complete(git_tag_exists, github_release_exists):
            return

        # 4. Handle PyPI publishing FIRST (before creating git artifacts)
        if ReleaseManager._should_publish_to_pypi(git_tag_exists, github_release_exists, package_name, version):
            PyPIPublisher.build_and_verify_package()
            if trusted_publishing:
                print("🔐 Trusted publishing enabled — deferring PyPI upload to the workflow publisher step")
            else:
                if not pypi_token:
                    raise ValueError(
                        "PyPI API token is required unless trusted publishing is enabled "
                        "(set MFLUX_TRUSTED_PUBLISHING=true or pass --trusted-publishing)"
                    )
                PyPIPublisher.publish_to_pypi(pypi_token, package_name, version)

        # 5. Create git tag if needed
        if not git_tag_exists:
            GitOperations.create_and_push_tag(tag_name, version)

        # 6. Create GitHub release if needed
        if not github_release_exists:
            release_notes = ChangelogParser.extract_release_notes_from_changelog(version)
            GitHubAPI.create_github_release(github_token, github_repo, tag_name, version, release_notes)

        print(f"🎉 Release process completed successfully for version {version}!")

    @staticmethod
    def _is_release_complete(git_tag_exists: bool, github_release_exists: bool) -> bool:
        return git_tag_exists and github_release_exists

    @staticmethod
    def _is_release_partial(git_tag_exists: bool, github_release_exists: bool) -> bool:
        return git_tag_exists or github_release_exists

    @staticmethod
    def _should_publish_to_pypi(
        git_tag_exists: bool,
        github_release_exists: bool,
        package_name: str,
        version: str,
    ) -> bool:
        # Only publish if this is a completely new release
        if ReleaseManager._is_release_partial(git_tag_exists, github_release_exists):
            print("⚠️  Skipping PyPI publishing since this appears to be a re-run")
            return False

        # Check if version already exists on PyPI
        try:
            if PyPIPublisher.version_exists_on_pypi(package_name, version):
                return False
        except (requests.RequestException, ValueError, OSError) as e:
            print(f"❌ Failed to check PyPI version: {e}")
            raise ValueError(f"PyPI version check failed: {e}") from e

        return True

    @staticmethod
    def _print_release_status(tag_name: str, git_tag_exists: bool, github_release_exists: bool) -> None:
        if ReleaseManager._is_release_complete(git_tag_exists, github_release_exists):
            print(f"✅ Release {tag_name} already exists completely")
            print("🔄 This appears to be a re-run of an existing release - nothing to do!")
        elif ReleaseManager._is_release_partial(git_tag_exists, github_release_exists):
            print("⚠️  Partial release state detected:")
            print(f"   Git tag exists: {git_tag_exists}")
            print(f"   GitHub release exists: {github_release_exists}")
            print("   Will complete the missing parts...")
