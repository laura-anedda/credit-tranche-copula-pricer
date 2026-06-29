"""End-to-end smoke test for the standalone HTML dashboard generator.

Marked `slow` and excluded from the default test run (see
`pyproject.toml`): building the full dashboard reprices the tranche
structure, runs the base correlation bootstrap, and evaluates three
risk-factor sensitivity sweeps, which together take on the order of a
minute. Run explicitly with `pytest -m slow`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
sys.path.insert(0, str(DASHBOARD_DIR))


@pytest.mark.slow
def test_dashboard_html_contains_all_sections() -> None:
    import generate_dashboard

    html = generate_dashboard.build_report_html()

    expected_section_ids = [
        "market-inputs",
        "model-parameters",
        "calibration",
        "pricing-results",
        "risk-measures",
        "diagnostics",
        "visualizations",
    ]
    for section_id in expected_section_ids:
        assert f'id="{section_id}"' in html

    assert "<!DOCTYPE html>" in html
    assert "Plotly.newPlot" in html or "Plotly.react" in html
