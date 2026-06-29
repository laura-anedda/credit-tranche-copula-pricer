"""
Generate the standalone HTML credit tranche pricer dashboard.

Builds a reference portfolio and pricing engine from the market inputs
defined below, runs pricing, calibration, sensitivity, and diagnostic
computations via `credit_copula.reporting` and `credit_copula.diagnostics`,
and assembles the results into a single self-contained HTML file (via
`dashboard.layout` and `dashboard.charts`) that opens directly in a
browser, with no application server.

Usage
-----
    python dashboard/generate_dashboard.py [--output PATH] [--open]

The default output path is `reports/credit_tranche_dashboard.html`,
relative to the repository root.

Notes
-----
The market inputs below are synthetic data constructed for reproducible
demonstration; replace `CDS_SPREADS`, `TRANCHE_PILLARS_PCT`, and the
other configuration constants with desk-sourced market data for live
use. The "market" tranche spreads used in the Calibration section are
a fixed-seed Gaussian perturbation of the model's own output, in the
absence of a live market data feed; see `NOTES.md` for the rationale
and `THEORY.md` for the calibration methodology this exercises.

`N_OBLIGORS`, `LOSS_UNIT`, `N_QUADRATURE_POINTS`, and
`N_INTEGRATION_STEPS_PER_YEAR` are set below the production-grade
defaults discussed in `README.md`'s Computational Notes, to keep report
generation time short given the repeated repricing performed by the
sensitivity sweeps and base correlation bootstrap; increase them for
higher-fidelity output at the cost of longer generation time.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import charts  # noqa: E402
import layout  # noqa: E402

from credit_copula import diagnostics  # noqa: E402
from credit_copula.base_correlation import bootstrap_base_correlation_from_standard_tranches  # noqa: E402
from credit_copula.market_data import DiscountCurve  # noqa: E402
from credit_copula.reporting import (  # noqa: E402
    base_tranche_expected_loss_curve,
    build_pricer,
    calibration_table,
    correlation_sensitivity,
    expected_tranche_loss_curves,
    hazard_rate_sensitivity,
    price_tranche_structure,
    recovery_rate_sensitivity,
)

# ---------------------------------------------------------------------------
# Synthetic market and model configuration, constructed for reproducible
# demonstration -- replace with desk-sourced market data for live use.
# ---------------------------------------------------------------------------
DISCOUNT_TENORS = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
DISCOUNT_RATES = np.full(DISCOUNT_TENORS.size, 0.03)
CDS_TENORS = np.array([1.0, 3.0, 5.0, 7.0, 10.0])
CDS_SPREADS = np.array([0.0040, 0.0065, 0.0090, 0.0105, 0.0120])
RECOVERY_RATE = 0.40
N_OBLIGORS = 60
CORRELATION = 0.25
LOSS_UNIT = 0.5
N_QUADRATURE_POINTS = 24
PAYMENT_FREQUENCY = 4
N_INTEGRATION_STEPS_PER_YEAR = 6
MATURITY = 5.0
TRANCHE_PILLARS_PCT = [(0.00, 0.03), (0.03, 0.07), (0.07, 0.10), (0.10, 0.15), (0.15, 0.30)]
MARKET_SPREAD_NOISE_BPS = 2.0
MARKET_SPREAD_NOISE_SEED = 11


def _tranche_label(attachment_pct: float, detachment_pct: float) -> str:
    return f"{attachment_pct:.0%}-{detachment_pct:.0%}"


def _fig_html(figure) -> str:
    figure.update_layout(autosize=True)
    return figure.to_html(
        full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False}
    )


def _chart_grid(*figures) -> str:
    cells = "".join(f'<div class="chart-cell">{_fig_html(fig)}</div>' for fig in figures)
    return f'<div class="chart-grid">{cells}</div>'


def build_report_html() -> str:
    """Run the full pricing/calibration/diagnostics pipeline and return the report HTML."""
    discount_curve = DiscountCurve(tenors=DISCOUNT_TENORS, zero_rates=DISCOUNT_RATES)
    pricer = build_pricer(
        cds_tenors=CDS_TENORS,
        cds_spreads=CDS_SPREADS,
        recovery_rate=RECOVERY_RATE,
        discount_curve=discount_curve,
        n_obligors=N_OBLIGORS,
        correlation=CORRELATION,
        loss_unit=LOSS_UNIT,
        n_quadrature_points=N_QUADRATURE_POINTS,
        payment_frequency=PAYMENT_FREQUENCY,
        n_integration_steps_per_year=N_INTEGRATION_STEPS_PER_YEAR,
    )
    credit_curve = pricer.portfolio.obligors[0].credit_curve
    index_notional = pricer.portfolio.total_notional
    tranche_labels = [_tranche_label(a, d) for a, d in TRANCHE_PILLARS_PCT]
    tranche_pillars_abs = [(a * index_notional, d * index_notional) for a, d in TRANCHE_PILLARS_PCT]
    max_curve_horizon = float(DISCOUNT_TENORS.max())

    print("Pricing tranche structure...")
    pricing_table = price_tranche_structure(pricer, TRANCHE_PILLARS_PCT, MATURITY)

    print("Evaluating expected tranche loss term structure...")
    etl_curves = expected_tranche_loss_curves(pricer, TRANCHE_PILLARS_PCT, MATURITY, n_points=30)

    print("Generating synthetic market quotes and bootstrapping base correlation...")
    model_spreads_bps = pricing_table["par_spread_bps"].to_numpy()
    rng = np.random.default_rng(MARKET_SPREAD_NOISE_SEED)
    market_spreads_bps = model_spreads_bps + rng.normal(0.0, MARKET_SPREAD_NOISE_BPS, size=model_spreads_bps.shape)
    market_spreads = market_spreads_bps * 1.0e-4

    base_curve, bc_diagnostics = bootstrap_base_correlation_from_standard_tranches(
        pricer, tranche_pillars_abs, market_spreads, MATURITY
    )

    cal_table, residuals = calibration_table(tranche_labels, market_spreads_bps, model_spreads_bps)
    rmse = diagnostics.root_mean_square_error(residuals.absolute_error_bps)
    max_error = diagnostics.maximum_absolute_error(residuals.absolute_error_bps)

    bc_table = pd.DataFrame(
        {
            "detachment_pct": [d for _, d in TRANCHE_PILLARS_PCT],
            "base_correlation": base_curve.correlations,
            "iterations": bc_diagnostics.iterations,
            "converged": bc_diagnostics.converged,
            "residual": bc_diagnostics.residual,
        }
    )

    print("Running calibration diagnostics...")
    worst_etl_violation = 0.0
    n_etl_violations = 0
    for label in tranche_labels:
        subset = etl_curves[etl_curves["tranche"] == label]
        check = diagnostics.check_expected_loss_time_monotonicity(
            subset["time"].to_numpy(), subset["expected_loss_pct"].to_numpy()
        )
        n_etl_violations += check.violation_indices.size
        worst_etl_violation = min(worst_etl_violation, check.max_violation)
    etl_check = diagnostics.ConsistencyCheck(
        passed=n_etl_violations == 0, max_violation=worst_etl_violation, violation_indices=np.array([], dtype=np.int64)
    )

    base_el_curve = base_tranche_expected_loss_curve(
        pricer, np.linspace(0.01, 1.0 - RECOVERY_RATE, 30), MATURITY
    )
    convexity_check = diagnostics.check_base_tranche_convexity(
        base_el_curve["detachment_pct"].to_numpy(), base_el_curve["expected_loss"].to_numpy()
    )
    monotonic_check = diagnostics.check_base_correlation_monotonicity(
        base_curve.detachment_points, base_curve.correlations
    )
    curvature = diagnostics.base_correlation_curvature(base_curve.detachment_points, base_curve.correlations)

    warnings = diagnostics.generate_calibration_warnings(
        residuals,
        expected_loss_time_check=etl_check,
        base_tranche_convexity_check=convexity_check,
        base_correlation_monotonicity_check=monotonic_check,
        curvature=curvature,
    )

    print("Computing risk sensitivities (correlation, recovery, hazard rate)...")
    corr_sensitivity_df = correlation_sensitivity(
        pricer, TRANCHE_PILLARS_PCT, MATURITY, np.linspace(0.05, 0.85, 6)
    )
    recovery_sensitivity_df = recovery_rate_sensitivity(
        cds_tenors=CDS_TENORS,
        cds_spreads=CDS_SPREADS,
        discount_curve=discount_curve,
        n_obligors=N_OBLIGORS,
        correlation=CORRELATION,
        loss_unit=LOSS_UNIT,
        tranche_pillars_pct=TRANCHE_PILLARS_PCT,
        maturity=MATURITY,
        recovery_grid=np.linspace(0.1, 0.6, 6),
        n_quadrature_points=N_QUADRATURE_POINTS,
        n_integration_steps_per_year=N_INTEGRATION_STEPS_PER_YEAR,
    )
    hazard_sensitivity_df = hazard_rate_sensitivity(
        pricer, TRANCHE_PILLARS_PCT, MATURITY, np.linspace(0.5, 1.5, 6)
    )

    print("Building model-transparency visualizations...")
    average_default_probability = float(pricer.portfolio.default_probabilities(MATURITY).mean())
    z_values = [-2.0, 0.0, 2.0]
    conditional_distributions = {
        f"Z = {z:+.1f}": pricer.conditional_loss_distribution(MATURITY, z) for z in z_values
    }
    unconditional_distribution = pricer.portfolio_loss_distribution(MATURITY)
    payoff_tranches = [
        (label, attachment, detachment) for label, (attachment, detachment) in zip(tranche_labels, tranche_pillars_abs)
    ]

    print("Assembling report sections...")
    nav: list[tuple[str, str]] = []
    sections: list[str] = []

    # ---- 1. Market Inputs ----
    market_tables = (
        '<div class="chart-grid">'
        f'<div class="chart-cell">{layout.render_table(pd.DataFrame({"Tenor (yrs)": DISCOUNT_TENORS, "Zero rate": DISCOUNT_RATES}), {"Zero rate": "{:.2%}"})}</div>'
        f'<div class="chart-cell">{layout.render_table(pd.DataFrame({"Tenor (yrs)": CDS_TENORS, "Par spread (bps)": CDS_SPREADS * 1e4}), {"Par spread (bps)": "{:.1f}"})}</div>'
        "</div>"
    )
    market_content = market_tables + _chart_grid(
        charts.discount_factor_figure(discount_curve, max_curve_horizon),
        charts.survival_probability_figure(credit_curve, max_curve_horizon),
    )
    sections.append(
        layout.render_section(
            "market-inputs", "Market Inputs", "Discount and single-name CDS curves used to calibrate the portfolio.", market_content
        )
    )
    nav.append(("market-inputs", "Market Inputs"))

    # ---- 2. Model Parameters ----
    params_df = pd.DataFrame(
        {
            "Parameter": [
                "Obligors", "Recovery rate", "Flat correlation", "Loss discretization unit",
                "Gauss-Hermite nodes", "Payment frequency (/yr)", "Protection leg steps (/yr)", "Maturity",
            ],
            "Value": [
                N_OBLIGORS, f"{RECOVERY_RATE:.0%}", f"{CORRELATION:.0%}", f"{LOSS_UNIT:.2f}",
                N_QUADRATURE_POINTS, PAYMENT_FREQUENCY, N_INTEGRATION_STEPS_PER_YEAR, f"{MATURITY:.1f}y",
            ],
        }
    )
    sections.append(
        layout.render_section(
            "model-parameters", "Model Parameters", "One-factor Gaussian copula engine configuration.", layout.render_table(params_df)
        )
    )
    nav.append(("model-parameters", "Model Parameters"))

    # ---- 3. Calibration ----
    calibration_metrics = layout.render_metric_row(
        [
            layout.metric_card("RMSE", f"{rmse:.2f} bps"),
            layout.metric_card("Max abs. error", f"{max_error:.2f} bps"),
            layout.metric_card("Base correlation curvature", f"{curvature:.3f}"),
            layout.metric_card("Pillars converged", f"{int(np.sum(bc_diagnostics.converged))}/{len(bc_diagnostics.converged)}"),
        ]
    )
    badges = " ".join(
        [
            layout.status_badge("ETL monotonicity", etl_check.passed),
            layout.status_badge("Base tranche convexity", convexity_check.passed),
            layout.status_badge("Skew monotonicity", monotonic_check.passed),
        ]
    )
    bc_chart_df = pd.DataFrame(
        {"detachment_pct": [d for _, d in TRANCHE_PILLARS_PCT], "base_correlation": base_curve.correlations}
    )
    calibration_content = (
        calibration_metrics
        + f"<p>{badges}</p>"
        + layout.render_warnings(warnings)
        + _chart_grid(charts.calibration_comparison_figure(cal_table), charts.base_correlation_figure(bc_chart_df))
        + '<p class="caption">Market quotes are synthetic: a fixed-seed Gaussian perturbation of the model\'s own spreads, in the absence of a live market data feed.</p>'
        + layout.render_table(
            cal_table,
            {
                "market_spread_bps": "{:.1f}", "model_spread_bps": "{:.1f}",
                "absolute_error_bps": "{:+.2f}", "relative_error_pct": "{:+.2f}",
            },
        )
        + layout.render_table(
            bc_table,
            {"detachment_pct": "{:.0%}", "base_correlation": "{:.4f}", "residual": "{:.2e}"},
        )
    )
    sections.append(
        layout.render_section(
            "calibration", "Calibration", "Market-vs-model spread comparison and base correlation bootstrap diagnostics.", calibration_content
        )
    )
    nav.append(("calibration", "Calibration"))

    # ---- 4. Pricing Results ----
    pricing_content = _chart_grid(
        charts.par_spread_bar_figure(pricing_table), charts.tranche_payoff_profile_figure(pricer.loss_grid, payoff_tranches)
    ) + layout.render_table(
        pricing_table.drop(columns=["attachment_pct", "detachment_pct"]),
        {"par_spread_bps": "{:.1f}", "risky_pv01": "{:.3f}", "protection_leg_pv": "{:.3f}", "premium_leg_annuity": "{:.3f}"},
    )
    sections.append(
        layout.render_section(
            "pricing-results", "Pricing Results", "Par spreads and leg present values for the standard tranche structure.", pricing_content
        )
    )
    nav.append(("pricing-results", "Pricing Results"))

    # ---- 5. Risk Measures ----
    risk_content = _chart_grid(
        charts.tranche_outstanding_notional_figure(etl_curves),
        charts.attachment_detachment_loss_profile_figure(base_el_curve, [d for _, d in TRANCHE_PILLARS_PCT]),
    ) + _chart_grid(
        charts.sensitivity_figure(corr_sensitivity_df, "correlation", "Correlation", "Spread sensitivity to correlation"),
        charts.sensitivity_figure(recovery_sensitivity_df, "recovery_rate", "Recovery rate", "Spread sensitivity to recovery rate"),
    ) + _chart_grid(
        charts.sensitivity_figure(
            hazard_sensitivity_df, "hazard_rate_multiplier", "Hazard rate multiplier (x baseline)", "Spread sensitivity to hazard rate"
        )
    )
    sections.append(
        layout.render_section(
            "risk-measures", "Risk Measures", "Outstanding notional, loss profile, and par spread sensitivities.", risk_content
        )
    )
    nav.append(("risk-measures", "Risk Measures"))

    # ---- 6. Diagnostics ----
    diagnostics_content = _chart_grid(
        charts.monotonicity_diagnostic_figure(
            base_el_curve["detachment_pct"].to_numpy(), base_el_curve["expected_loss"].to_numpy(),
            convexity_check.violation_indices, "Detachment K", "E[min(L,K)]", "Base tranche convexity check",
            x_tickformat=".0%",
        ),
        charts.monotonicity_diagnostic_figure(
            base_curve.detachment_points / index_notional, base_curve.correlations,
            monotonic_check.violation_indices, "Detachment", "Base correlation", "Base correlation monotonicity check",
            x_tickformat=".0%",
        ),
    ) + layout.render_table(
        bc_table, {"detachment_pct": "{:.0%}", "base_correlation": "{:.4f}", "residual": "{:.2e}"}
    )
    sections.append(
        layout.render_section(
            "diagnostics", "Diagnostics", "No-arbitrage and stability checks on the calibrated base correlation curve.", diagnostics_content
        )
    )
    nav.append(("diagnostics", "Diagnostics"))

    # ---- 7. Visualizations ----
    visualizations_content = _chart_grid(
        charts.copula_dependence_figure(), charts.conditional_default_probability_figure(average_default_probability, [0.1, 0.3, 0.5, 0.7, 0.9])
    ) + _chart_grid(
        charts.conditional_loss_distribution_figure(pricer.loss_grid, conditional_distributions),
        charts.unconditional_loss_distribution_figure(pricer.loss_grid, unconditional_distribution),
    ) + _chart_grid(charts.expected_tranche_loss_figure(etl_curves))
    sections.append(
        layout.render_section(
            "visualizations", "Visualizations",
            "Copula dependence structure and conditional/unconditional portfolio loss distributions.",
            visualizations_content,
        )
    )
    nav.append(("visualizations", "Visualizations"))

    return layout.render_page(
        title="Credit Index Tranche Pricer &mdash; Analytics Report",
        subtitle=f"One-factor Gaussian Copula | {N_OBLIGORS} obligors | &rho;={CORRELATION:.0%} | {MATURITY:.0f}Y maturity",
        sections_html=sections,
        nav_items=nav,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the standalone credit tranche pricer HTML dashboard.")
    default_output = (
        Path(__file__).resolve().parent.parent / "reports" / "credit_tranche_dashboard.html"
    )
    parser.add_argument(
        "--output", type=Path, default=default_output,
        help="Output HTML file path.",
    )
    parser.add_argument("--open", action="store_true", help="Open the generated report in the default browser.")
    args = parser.parse_args()

    html = build_report_html()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Report written to {args.output}")

    if args.open:
        webbrowser.open(args.output.resolve().as_uri())


if __name__ == "__main__":
    main()
