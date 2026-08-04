import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

WORKFLOW = Path(".github/workflows/predict.yml")


def test_prediction_workflow_serializes_shadow_ledger_writes_and_deploys_shadow_report():
    """Scheduled runs must not race ledger appends or publish an incomplete report."""
    workflow = WORKFLOW.read_text()

    assert "concurrency:" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "ref: ${{ github.ref }}" in workflow
    assert "fetch-depth: 0" in workflow

    assert "ledger_paths=(" in workflow
    assert "data/shadow/predictions.csv" in workflow
    assert "data/shadow/settlements.csv" in workflow
    assert 'git add "${ledger_paths[@]}"' in workflow
    assert "git diff --staged --quiet" in workflow
    assert 'git commit --only -m "chore: update shadow evaluation ledger"' in workflow
    assert 'git push origin "HEAD:${GITHUB_REF}"' in workflow

    assert "cp reports/shadow_evaluation.html docs/shadow.html" in workflow
    assert "reports/predictions_*.html" in workflow


def _shadow_persistence_script() -> str:
    workflow = WORKFLOW.read_text()
    match = re.search(
        r"- name: Commit shadow evaluation ledger\n        run: \|\n(?P<script>(?:          .*\n)+?)\n      - name:",
        workflow,
    )
    assert match, "The prediction workflow must include the shadow persistence step"
    return textwrap.dedent(match.group("script"))


@pytest.mark.parametrize(
    "ledger_name",
    ["predictions.csv", "settlements.csv"],
    ids=["prediction-only", "settlement-only"],
)
def test_shadow_persistence_bootstraps_when_only_one_ledger_exists(
    tmp_path, monkeypatch, ledger_name
):
    """A new ledger must commit successfully before its companion exists."""
    git_path = shutil.which("git")
    assert git_path is not None
    subprocess.run([git_path, "init"], cwd=tmp_path, check=True, capture_output=True)

    ledger = tmp_path / "data" / "shadow" / ledger_name
    ledger.parent.mkdir(parents=True)
    ledger.write_text("prediction_id\nexample\n")
    (tmp_path / "unrelated.txt").write_text("must not be staged\n")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_git = bin_dir / "git"
    fake_git.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "push" ]; then exit 0; fi\n'
        f'exec "{git_path}" "$@"\n'
    )
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    completed = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _shadow_persistence_script()],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    committed_paths = subprocess.run(
        [git_path, "show", "--format=", "--name-only", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert committed_paths == [f"data/shadow/{ledger_name}"]
    staged_paths = subprocess.run(
        [git_path, "diff", "--cached", "--name-only"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert staged_paths == ""
