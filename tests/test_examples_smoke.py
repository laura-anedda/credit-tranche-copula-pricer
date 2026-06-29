"""End-to-end smoke test for the worked example script.

Marked `slow` and excluded from the default test run (see
`pyproject.toml`): the example prices a 125-obligor tranche structure
and bootstraps base correlation across five pillars. Run explicitly
with `pytest -m slow`.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "examples" / "index_tranche_example.py"


@pytest.mark.slow
def test_example_script_runs_without_error(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")
    captured = capsys.readouterr()
    assert "Index notional" in captured.out
    assert "Base Correlation" in captured.out
