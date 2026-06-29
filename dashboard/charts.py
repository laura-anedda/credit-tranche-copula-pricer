"""
Plotly figure builders for the credit tranche pricer dashboard.

Every function in this module is a pure transformation from a
`pandas.DataFrame` (produced by :mod:`credit_copula.reporting`) or a
`credit_copula` curve object into a `plotly.graph_objects.Figure`. No
function reads from or writes to any external or global state, so each
is independently testable and reusable outside the dashboard (e.g. in a
notebook or batch report).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from credit_copula import copula as copula_model
from credit_copula.market_data import CreditCurve, DiscountCurve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from illustrations import simulate_latent_pair  # noqa: E402

__all__ = [
    "survival_probability_figure",
    "discount_factor_figure",
    "par_spread_bar_figure",
    "expected_tranche_loss_figure",
    "base_correlation_figure",
    "copula_dependence_figure",
    "conditional_default_probability_figure",
    "conditional_loss_distribution_figure",
    "unconditional_loss_distribution_figure",
    "tranche_payoff_profile_figure",
    "tranche_outstanding_notional_figure",
    "attachment_detachment_loss_profile_figure",
    "sensitivity_figure",
    "calibration_comparison_figure",
    "monotonicity_diagnostic_figure",
]

_PLOTLY_LAYOUT_DEFAULTS = {
    "template": "plotly_white",
    "margin": {"l": 48, "r": 20, "t": 36, "b": 40},
    "height": 340,
    "font": {"size": 11.5},
    "title": {"font": {"size": 13}},
}


def survival_probability_figure(credit_curve: CreditCurve, max_years: float, n_points: int = 200) -> go.Figure:
    """
    Plot the bootstrapped survival probability curve :math:`Q(t)`.

    Parameters
    ----------
    credit_curve : CreditCurve
        Bootstrapped credit curve.
    max_years : float
        Right-hand edge of the time axis, in years.
    n_points : int, default=200
        Number of evaluation points.

    Returns
    -------
    go.Figure
        Line chart of survival probability against time.
    """
    times = np.linspace(0.0, max_years, n_points)
    survival = credit_curve.survival_probability(times)
    figure = px.line(
        x=times,
        y=survival,
        labels={"x": "Time (years)", "y": "Survival probability Q(t)"},
        title="Bootstrapped survival probability curve",
    )
    figure.update_yaxes(range=[0.0, 1.0])
    figure.update_layout(**_PLOTLY_LAYOUT_DEFAULTS)
    return figure


def discount_factor_figure(discount_curve: DiscountCurve, max_years: float, n_points: int = 200) -> go.Figure:
    """
    Plot the discount factor curve :math:`DF(t)`.

    Parameters
    ----------
    discount_curve : DiscountCurve
        Risk-free discount curve.
    max_years : float
        Right-hand edge of the time axis, in years.
    n_points : int, default=200
        Number of evaluation points.

    Returns
    -------
    go.Figure
        Line chart of discount factor against time.
    """
    times = np.linspace(0.0, max_years, n_points)
    discount_factors = discount_curve.discount_factor(times)
    figure = px.line(
        x=times,
        y=discount_factors,
        labels={"x": "Time (years)", "y": "Discount factor DF(t)"},
        title="Risk-free discount curve",
    )
    figure.update_layout(**_PLOTLY_LAYOUT_DEFAULTS)
    return figure


def par_spread_bar_figure(tranche_pricing_table: pd.DataFrame) -> go.Figure:
    """
    Plot a bar chart of par spreads (basis points) across the tranche structure.

    Parameters
    ----------
    tranche_pricing_table : pd.DataFrame
        Output of :func:`credit_copula.reporting.price_tranche_structure`,
        with columns ``tranche`` and ``par_spread_bps``.

    Returns
    -------
    go.Figure
        Bar chart, one bar per tranche, ordered as supplied.
    """
    figure = px.bar(
        tranche_pricing_table,
        x="tranche",
        y="par_spread_bps",
        labels={"tranche": "Tranche", "par_spread_bps": "Par spread (bps)"},
        title="Tranche par spreads",
        text_auto=".1f",
    )
    figure.update_layout(**_PLOTLY_LAYOUT_DEFAULTS)
    return figure


def expected_tranche_loss_figure(etl_curves: pd.DataFrame) -> go.Figure:
    """
    Plot expected tranche loss (as a fraction of tranche notional) over time.

    Parameters
    ----------
    etl_curves : pd.DataFrame
        Long-format output of
        :func:`credit_copula.reporting.expected_tranche_loss_curves`, with
        columns ``time``, ``tranche``, ``expected_loss_pct``.

    Returns
    -------
    go.Figure
        Line chart, one line per tranche.
    """
    figure = px.line(
        etl_curves,
        x="time",
        y="expected_loss_pct",
        color="tranche",
        labels={
            "time": "Time (years)",
            "expected_loss_pct": "Expected tranche loss (% of tranche notional)",
            "tranche": "Tranche",
        },
        title="Expected tranche loss term structure",
    )
    figure.update_yaxes(tickformat=".0%")
    figure.update_layout(**_PLOTLY_LAYOUT_DEFAULTS)
    return figure


def base_correlation_figure(base_correlation_curve: pd.DataFrame) -> go.Figure:
    """
    Plot the bootstrapped base correlation skew.

    Parameters
    ----------
    base_correlation_curve : pd.DataFrame
        Output of :func:`credit_copula.reporting.base_correlation_table`,
        with columns ``detachment_pct`` and ``base_correlation``.

    Returns
    -------
    go.Figure
        Line-and-marker chart of base correlation against detachment
        point.
    """
    figure = px.line(
        base_correlation_curve,
        x="detachment_pct",
        y="base_correlation",
        markers=True,
        labels={"detachment_pct": "Detachment point", "base_correlation": "Base correlation"},
        title="Base correlation skew",
    )
    figure.update_xaxes(tickformat=".0%")
    figure.update_yaxes(range=[0.0, 1.0])
    figure.update_layout(**_PLOTLY_LAYOUT_DEFAULTS)
    return figure


def copula_dependence_figure(
    correlation_grid: list[float] | None = None, n_samples: int = 2500
) -> go.Figure:
    """
    Visualize the one-factor Gaussian copula dependence structure via simulated latent pairs.

    Renders a scatter plot of paired latent asset values
    :math:`(X_i, X_j)` (see
    :func:`dashboard.illustrations.simulate_latent_pair`), with a slider
    scrubbing through a grid of correlation values to show how the
    dependence structure tightens around the diagonal as :math:`\\rho`
    increases.

    Parameters
    ----------
    correlation_grid : list[float], optional
        Correlation values to expose via the slider. Defaults to
        ``[0.05, 0.15, ..., 0.95]``.
    n_samples : int, default=2500
        Number of simulated latent pairs per frame.

    Returns
    -------
    go.Figure
        Scatter plot with a correlation slider.
    """
    if correlation_grid is None:
        correlation_grid = [round(0.05 + 0.10 * i, 2) for i in range(10)]

    frames = []
    for rho in correlation_grid:
        x, y = simulate_latent_pair(rho, n_samples=n_samples)
        frames.append(go.Frame(data=[go.Scatter(x=x, y=y)], name=f"{rho:.2f}"))

    initial_x, initial_y = simulate_latent_pair(correlation_grid[0], n_samples=n_samples)
    figure = go.Figure(
        data=[
            go.Scatter(
                x=initial_x,
                y=initial_y,
                mode="markers",
                marker={"size": 3, "opacity": 0.45, "color": "#0b3d63"},
            )
        ],
        frames=frames,
    )
    figure.update_layout(
        title="Latent asset value dependence: X_i vs X_j",
        xaxis_title="X_i",
        yaxis_title="X_j",
        sliders=[
            {
                "currentvalue": {"prefix": "correlation rho = "},
                "steps": [
                    {"args": [[f"{rho:.2f}"], {"frame": {"duration": 0}, "mode": "immediate"}],
                     "label": f"{rho:.2f}", "method": "animate"}
                    for rho in correlation_grid
                ],
            }
        ],
    )
    figure.update_xaxes(range=[-4.0, 4.0])
    figure.update_yaxes(range=[-4.0, 4.0])
    figure.update_layout(**_PLOTLY_LAYOUT_DEFAULTS)
    return figure


def conditional_default_probability_figure(
    default_probability: float, correlation_values: list[float], z_range: tuple[float, float] = (-4.0, 4.0)
) -> go.Figure:
    """
    Plot conditional default probability as a function of the systemic factor.

    .. math::
        p(z) = \\Phi\\!\\left(\\frac{\\Phi^{-1}(p) - \\sqrt{\\rho}\\, z}{\\sqrt{1-\\rho}}\\right)

    drawn for several correlation values, holding the marginal default
    probability `p` fixed, so the effect of correlation on factor
    sensitivity is directly comparable.

    Parameters
    ----------
    default_probability : float
        Marginal (unconditional) default probability `p`.
    correlation_values : list[float]
        Correlation values to plot as separate curves.
    z_range : tuple[float, float], default=(-4.0, 4.0)
        Range of the systemic factor axis.

    Returns
    -------
    go.Figure
        One line per correlation value.
    """
    z = np.linspace(z_range[0], z_range[1], 200)
    figure = go.Figure()
    for rho in correlation_values:
        p_z = copula_model.conditional_default_probability(default_probability, rho, z)
        figure.add_trace(go.Scatter(x=z, y=p_z, mode="lines", name=f"rho = {rho:.2f}"))
    figure.update_layout(
        title="Conditional default probability p(Z)",
        xaxis_title="Systemic factor Z",
        yaxis_title="Conditional default probability",
    )
    figure.update_yaxes(tickformat=".0%")
    figure.update_layout(**_PLOTLY_LAYOUT_DEFAULTS)
    return figure


def conditional_loss_distribution_figure(
    loss_grid: np.ndarray, conditional_distributions: dict[str, np.ndarray]
) -> go.Figure:
    """
    Overlay the portfolio loss distribution conditional on representative systemic factor realizations.

    Parameters
    ----------
    loss_grid : np.ndarray
        Portfolio loss values (currency units), shared across all
        distributions.
    conditional_distributions : dict[str, np.ndarray]
        Mapping of legend label (e.g. ``"Z = -2.0"``) to the
        corresponding conditional loss probability mass function (see
        :func:`credit_copula.portfolio.conditional_loss_distribution`).

    Returns
    -------
    go.Figure
        Overlaid line traces, one per systemic factor realization.
    """
    figure = go.Figure()
    for label, distribution in conditional_distributions.items():
        figure.add_trace(go.Scatter(x=loss_grid, y=distribution, mode="lines", name=label))
    figure.update_layout(
        title="Conditional portfolio loss distribution P(L | Z)",
        xaxis_title="Portfolio loss (currency units)",
        yaxis_title="Probability",
    )
    figure.update_layout(**_PLOTLY_LAYOUT_DEFAULTS)
    return figure


def unconditional_loss_distribution_figure(loss_grid: np.ndarray, probabilities: np.ndarray) -> go.Figure:
    """
    Plot the unconditional portfolio loss distribution.

    Parameters
    ----------
    loss_grid : np.ndarray
        Portfolio loss values (currency units).
    probabilities : np.ndarray
        Probability mass at each loss grid point (see
        :func:`credit_copula.portfolio.loss_distribution`).

    Returns
    -------
    go.Figure
        Bar chart of the loss distribution.
    """
    figure = px.bar(
        x=loss_grid,
        y=probabilities,
        labels={"x": "Portfolio loss (currency units)", "y": "Probability"},
        title="Unconditional portfolio loss distribution P(L)",
    )
    figure.update_layout(**_PLOTLY_LAYOUT_DEFAULTS)
    return figure


def tranche_payoff_profile_figure(
    loss_grid: np.ndarray, tranches: list[tuple[str, float, float]]
) -> go.Figure:
    """
    Plot tranche loss payoff as a function of portfolio loss.

    .. math::
        \\text{TrancheLoss}(L) = \\min(\\max(L - A, 0), D - A)

    Parameters
    ----------
    loss_grid : np.ndarray
        Portfolio loss values (currency units) at which to evaluate the
        payoff.
    tranches : list[tuple[str, float, float]]
        ``(label, attachment, detachment)`` triples, in currency units.

    Returns
    -------
    go.Figure
        One payoff line per tranche.
    """
    figure = go.Figure()
    for label, attachment, detachment in tranches:
        payoff = np.clip(loss_grid - attachment, 0.0, detachment - attachment)
        figure.add_trace(go.Scatter(x=loss_grid, y=payoff, mode="lines", name=label))
    figure.update_layout(
        title="Tranche loss payoff profile",
        xaxis_title="Portfolio loss (currency units)",
        yaxis_title="Tranche loss (currency units)",
    )
    figure.update_layout(**_PLOTLY_LAYOUT_DEFAULTS)
    return figure


def tranche_outstanding_notional_figure(etl_curves: pd.DataFrame) -> go.Figure:
    """
    Plot expected outstanding tranche notional through time.

    Derived from expected tranche loss as
    :math:`1 - \\text{ETL}(t) / N`, i.e. the surviving fraction of
    tranche notional.

    Parameters
    ----------
    etl_curves : pd.DataFrame
        Long-format output of
        :func:`credit_copula.reporting.expected_tranche_loss_curves`.

    Returns
    -------
    go.Figure
        One outstanding-notional line per tranche.
    """
    df = etl_curves.copy()
    df["outstanding_pct"] = 1.0 - df["expected_loss_pct"]
    figure = px.line(
        df,
        x="time",
        y="outstanding_pct",
        color="tranche",
        labels={"time": "Time (years)", "outstanding_pct": "Outstanding notional (% of tranche)", "tranche": "Tranche"},
        title="Expected outstanding tranche notional",
    )
    figure.update_yaxes(tickformat=".0%", range=[0.0, 1.02])
    figure.update_layout(**_PLOTLY_LAYOUT_DEFAULTS)
    return figure


def attachment_detachment_loss_profile_figure(
    base_el_curve: pd.DataFrame, tranche_pillars_pct: list[float] | None = None
) -> go.Figure:
    """
    Plot the base tranche expected-loss curve E[min(L, K)] against detachment point.

    Parameters
    ----------
    base_el_curve : pd.DataFrame
        Output of
        :func:`credit_copula.reporting.base_tranche_expected_loss_curve`,
        with columns ``detachment_pct`` and ``expected_loss``.
    tranche_pillars_pct : list[float], optional
        Standard tranche detachment pillars to mark with vertical
        reference lines.

    Returns
    -------
    go.Figure
        Line chart of the attachment/detachment loss profile.
    """
    figure = px.line(
        base_el_curve,
        x="detachment_pct",
        y="expected_loss",
        markers=True,
        labels={"detachment_pct": "Detachment point K", "expected_loss": "E[min(L,K)] (currency units)"},
        title="Attachment/detachment loss profile",
    )
    figure.update_xaxes(tickformat=".0%")
    if tranche_pillars_pct:
        for pillar in tranche_pillars_pct:
            figure.add_vline(x=pillar, line_dash="dot", line_color="#9aa5b1")
    figure.update_layout(**_PLOTLY_LAYOUT_DEFAULTS)
    return figure


def sensitivity_figure(
    sensitivity_df: pd.DataFrame, x_column: str, x_label: str, title: str
) -> go.Figure:
    """
    Plot tranche par spread sensitivity to a single risk factor.

    Parameters
    ----------
    sensitivity_df : pd.DataFrame
        Long-format output of
        :func:`credit_copula.reporting.correlation_sensitivity`,
        :func:`credit_copula.reporting.recovery_rate_sensitivity`, or
        :func:`credit_copula.reporting.hazard_rate_sensitivity`, with
        columns ``[x_column, "tranche", "par_spread_bps"]``.
    x_column : str
        Name of the swept parameter column.
    x_label : str
        Axis label for the swept parameter.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        One line per tranche; tranches can be toggled via the legend.
    """
    figure = px.line(
        sensitivity_df,
        x=x_column,
        y="par_spread_bps",
        color="tranche",
        markers=True,
        labels={x_column: x_label, "par_spread_bps": "Par spread (bps)", "tranche": "Tranche"},
        title=title,
    )
    figure.update_layout(**_PLOTLY_LAYOUT_DEFAULTS)
    return figure


def calibration_comparison_figure(calibration_df: pd.DataFrame) -> go.Figure:
    """
    Plot market-versus-model par spreads with absolute pricing error.

    Parameters
    ----------
    calibration_df : pd.DataFrame
        Output of :func:`credit_copula.reporting.calibration_table`,
        with columns ``tranche``, ``market_spread_bps``,
        ``model_spread_bps``, ``absolute_error_bps``.

    Returns
    -------
    go.Figure
        Grouped bar chart (market vs. model) with the absolute error
        annotated above each tranche pair.
    """
    figure = go.Figure()
    figure.add_trace(go.Bar(x=calibration_df["tranche"], y=calibration_df["market_spread_bps"], name="Market"))
    figure.add_trace(go.Bar(x=calibration_df["tranche"], y=calibration_df["model_spread_bps"], name="Model"))
    for _, row in calibration_df.iterrows():
        figure.add_annotation(
            x=row["tranche"],
            y=max(row["market_spread_bps"], row["model_spread_bps"]),
            text=f"{row['absolute_error_bps']:+.1f} bps",
            showarrow=False,
            yshift=12,
            font={"size": 10},
        )
    figure.update_layout(
        barmode="group",
        title="Market vs. model par spread",
        xaxis_title="Tranche",
        yaxis_title="Par spread (bps)",
    )
    figure.update_layout(**_PLOTLY_LAYOUT_DEFAULTS)
    return figure


def monotonicity_diagnostic_figure(
    x: np.ndarray,
    y: np.ndarray,
    violation_indices: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    x_tickformat: str | None = None,
) -> go.Figure:
    """
    Plot a diagnostic curve with violation points highlighted.

    Parameters
    ----------
    x, y : np.ndarray
        Curve coordinates.
    violation_indices : np.ndarray
        Indices into `x`/`y` at which a monotonicity/convexity check
        (see :mod:`credit_copula.diagnostics`) was violated.
    x_label, y_label : str
        Axis labels.
    title : str
        Chart title.
    x_tickformat : str, optional
        Plotly d3-format string for the x-axis ticks (e.g. ``".0%"`` for
        fractional detachment points expressed as percentages).

    Returns
    -------
    go.Figure
        Line chart with violated points marked in red.
    """
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="curve", marker={"color": "#0b3d63"}))
    if violation_indices.size > 0:
        figure.add_trace(
            go.Scatter(
                x=np.asarray(x)[violation_indices],
                y=np.asarray(y)[violation_indices],
                mode="markers",
                name="violation",
                marker={"color": "#a3261f", "size": 10, "symbol": "x"},
            )
        )
    figure.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, showlegend=violation_indices.size > 0)
    if x_tickformat:
        figure.update_xaxes(tickformat=x_tickformat)
    figure.update_layout(**_PLOTLY_LAYOUT_DEFAULTS)
    return figure
