import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "release-note.yml"


def _check(body: str, tmp_path) -> tuple[int, str]:
    # Run the inline script from the workflow the way CI runs it, against one PR body.
    script = textwrap.dedent(WORKFLOW.read_text().split("python3 - <<'PY'\n", 1)[1].split("\n          PY", 1)[0])
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"body": body}}))
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "GITHUB_EVENT_PATH": str(event)},
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout


@pytest.mark.fast
def test_a_complete_block_passes(tmp_path):
    code, out = _check("## Release note\n\n```release-note\nSomething a user can read.\n```\n", tmp_path)

    assert code == 0
    assert "release-note block found" in out


@pytest.mark.fast
def test_a_fence_without_the_tag_says_what_the_author_typed(tmp_path):
    # The miss that cost a contributor a round trip: the note is there, on the fence line,
    # and the old message asked for "a release-note block" while one was on screen.
    code, out = _check("## Release note\n```improve the CI manifest\n```\n", tmp_path)

    assert code == 1
    assert "```improve the CI manifest" in out
    assert "opens with exactly ```release-note" in out


@pytest.mark.fast
def test_an_empty_block_says_it_is_empty(tmp_path):
    code, out = _check("## Release note\n\n```release-note\n```\n", tmp_path)

    assert code == 1
    assert "is empty" in out


@pytest.mark.fast
def test_no_block_says_none_was_found(tmp_path):
    code, out = _check("## What\n\nA change with no note.\n", tmp_path)

    assert code == 1
    assert "No ```release-note block found" in out


@pytest.mark.fast
def test_every_failure_shows_the_block_to_copy(tmp_path):
    for body in ("## Release note\n```oops\n```\n", "## Release note\n\n```release-note\n```\n", "nothing here"):
        code, out = _check(body, tmp_path)

        assert code == 1
        # %0A is how a GitHub annotation carries a newline; the example must survive in it.
        assert "```release-note%0AOne or two sentences a user can read.%0A```" in out


@pytest.mark.fast
def test_the_regex_still_matches_the_extractor():
    from mflux.release import release_notes

    match = re.search(r're\.search\(r"(.+?)", body, (.+?)\)', WORKFLOW.read_text())

    assert match is not None
    assert match.group(1) == release_notes._FENCE.pattern
