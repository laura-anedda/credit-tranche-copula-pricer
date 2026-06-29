"""
Tabular reporting layer for tranche pricing results.

This module exposes pure, side-effect-free functions that translate the
core pricing primitives in :mod:`credit_copula.pricer` and
:mod:`credit_copula.base_correlation` into tabular (`pandas.DataFrame`)
form suitable for presentation layers -- dashboards, notebooks, or
batch reports -- without introducing any presentation-layer dependency
into the core pricing library. Plotting and UI frameworks are
deliberately kept out of this module and out of the `credit_copula`
package as a whole; see `dashboard/` for the standalone HTML/Plotly
presentation layer built on top of these functions.

All functions accept and return only primitive types, NumPy arrays, and
`pandas` objects, so they remain independently unit-testable without a
running UI process.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from credit_copula.base_correlation import bootstrap_base_correlation
from credit_copula.diagnostics import PricingResiduals, compute_pricing_residuals
from credit_copula.market_data import CreditCurve, DiscountCurve, bootstrap_hazard_rates
from credit_copula.portfolio import CreditPortfolio, Obligor
from credit_copula.pricer import CDOPricer
from credit_copula.tranche import Tranche

__all__ = [
    "build_pricer",
    "price_tranche_structure",
    "expected_tranche_loss_curves",
    "base_correlation_table",
    "calibration_table",
    "base_tranche_expected_loss_curve",
    "correlation_sensitivity",
    "recovery_rate_sensitivity",
    "hazard_rate_sensitivity",
]


def _tranche_label(attachment_pct: float, detachment_pct: float) -> str:
    """Format a tranche pillar pair as a market-style label, e.g. ``"3%-7%"``."""
    return f"{attachment_pct:.0%}-{detachment_pct:.0%}"


def price_tranche_structure(
    pricer: CDOPricer,
    tranche_pillars_pct: Sequence[tuple[float, float]],
    maturity: float,
) -> pd.DataFrame:
    """
    Price a structure of standard tranches and tabulate the results.

    Parameters
    ----------
    pricer : CDOPricer
        Configured pricing engine.
    tranche_pillars_pct : Sequence[tuple[float, float]]
        Tranche attachment/detachment points expressed as fractions of
        the portfolio's total notional (e.g. ``(0.03, 0.07)`` for a
        3%-7% tranche). Each pair must satisfy
        ``0 <= attachment < detachment <= 1``.
    maturity : float
        Common maturity, in years, for all tranches in the structure.

    Returns
    -------
    pd.DataFrame
        One row per tranche, with columns:

        - ``tranche``: market-style label (e.g. ``"3%-7%"``)
        - ``attachment_pct``, ``detachment_pct``: pillar points as fractions
        - ``par_spread_bps``: break-even running spread in basis points
        - ``risky_pv01``: risky PV01 per unit notional
        - ``protection_leg_pv``, ``premium_leg_annuity``: leg present values

    Raises
    ------
    ValueError
        If any pillar pair violates ``0 <= attachment < detachment <= 1``.
    """
    index_notional = pricer.portfolio.total_notional
    rows: list[dict[str, float | str]] = []

    for attachment_pct, detachment_pct in tranche_pillars_pct:
        if not (0.0 <= attachment_pct < detachment_pct <= 1.0):
            raise ValueError(
                "each tranche pillar pair must satisfy 0 <= attachment < detachment <= 1, "
                f"got ({attachment_pct}, {detachment_pct})"
            )
        tranche = Tranche(
            attachment=attachment_pct * index_notional,
            detachment=detachment_pct * index_notional,
        )
        result = pricer.price_tranche(tranche, maturity)
        rows.append(
            {
                "tranche": _tranche_label(attachment_pct, detachment_pct),
                "attachment_pct": attachment_pct,
                "detachment_pct": detachment_pct,
                "par_spread_bps": result.par_spread * 1.0e4,
                "risky_pv01": result.risky_pv01,
                "protection_leg_pv": result.protection_leg_pv,
                "premium_leg_annuity": result.risky_annuity,
            }
        )

    return pd.DataFrame.from_records(rows)


def expected_tranche_loss_curves(
    pricer: CDOPricer,
    tranche_pillars_pct: Sequence[tuple[float, float]],
    maturity: float,
    n_points: int = 50,
) -> pd.DataFrame:
    """
    Evaluate the expected tranche loss (ETL) term structure for a set of tranches.

    Parameters
    ----------
    pricer : CDOPricer
        Configured pricing engine.
    tranche_pillars_pct : Sequence[tuple[float, float]]
        Tranche attachment/detachment points as fractions of total
        notional, as in :func:`price_tranche_structure`.
    maturity : float
        Maximum time horizon, in years, over which to evaluate the ETL
        curve.
    n_points : int, default=50
        Number of evenly-spaced time points (excluding ``t=0``) at
        which to evaluate expected tranche loss.

    Returns
    -------
    pd.DataFrame
        Long-format ("tidy") table with columns ``time``, ``tranche``,
        and ``expected_loss_pct`` (expected tranche loss as a fraction
        of the tranche's own notional, in ``[0, 1]``), suitable for
        direct use with grouped line-chart plotting libraries.
    """
    index_notional = pricer.portfolio.total_notional
    times = np.linspace(0.0, maturity, n_points + 1)
    records: list[dict[str, float | str]] = []

    for attachment_pct, detachment_pct in tranche_pillars_pct:
        tranche = Tranche(
            attachment=attachment_pct * index_notional,
            detachment=detachment_pct * index_notional,
        )
        label = _tranche_label(attachment_pct, detachment_pct)
        for t in times:
            etl = pricer.expected_tranche_loss(float(t), tranche)
            records.append(
                {
                    "time": float(t),
                    "tranche": label,
                    "expected_loss_pct": etl / tranche.notional,
                }
            )

    return pd.DataFrame.from_records(records)


def base_correlation_table(
    pricer: CDOPricer,
    detachment_points_pct: Sequence[float],
    maturity: float,
) -> pd.DataFrame:
    """
    Bootstrap and tabulate the base correlation skew.

    The base tranche par spreads used as bootstrap targets are computed
    directly from `pricer`'s own (per-obligor) correlation assumptions,
    i.e. this reports the base correlation skew implied by treating the
    portfolio's own correlation structure as the "market". To bootstrap
    against genuine market quotes, use
    :func:`credit_copula.base_correlation.bootstrap_base_correlation`
    directly with externally-sourced par spreads.

    Parameters
    ----------
    pricer : CDOPricer
        Configured pricing engine.
    detachment_points_pct : Sequence[float]
        Base tranche detachment points as fractions of total notional,
        strictly increasing, each in :math:`(0, 1]`.
    maturity : float
        Common maturity, in years.

    Returns
    -------
    pd.DataFrame
        One row per detachment point, with columns ``detachment_pct``
        and ``base_correlation``.
    """
    index_notional = pricer.portfolio.total_notional
    detachment_points = np.asarray(detachment_points_pct, dtype=np.float64) * index_notional

    base_tranche_spreads = np.array(
        [
            pricer.price_tranche(Tranche(0.0, float(detachment)), maturity).par_spread
            for detachment in detachment_points
        ]
    )

    curve = bootstrap_base_correlation(pricer, detachment_points, base_tranche_spreads, maturity)

    return pd.DataFrame(
        {
            "detachment_pct": np.asarray(detachment_points_pct, dtype=np.float64),
            "base_correlation": curve.correlations,
        }
    )


def build_pricer(
    cds_tenors: np.ndarray,
    cds_spreads: np.ndarray,
    recovery_rate: float,
    discount_curve: DiscountCurve,
    n_obligors: int,
    correlation: float,
    loss_unit: float,
    n_quadrature_points: int = 32,
    payment_frequency: int = 4,
    n_integration_steps_per_year: int = 12,
    notional_per_obligor: float = 1.0,
) -> CDOPricer:
    """
    Construct a `CDOPricer` for a homogeneous reference portfolio from primitive market inputs.

    This factors out the curve bootstrap and portfolio construction
    logic shared by the dashboard, example scripts, and sensitivity
    sweeps in this module, so that a consistent portfolio can be rebuilt
    cheaply under perturbed inputs (e.g. a different recovery rate or
    hazard rate level; see :func:`recovery_rate_sensitivity` and
    :func:`hazard_rate_sensitivity`).

    Parameters
    ----------
    cds_tenors : np.ndarray
        Single-name CDS curve maturities, in years.
    cds_spreads : np.ndarray
        Single-name CDS par spreads, in decimal form, aligned with
        `cds_tenors`.
    recovery_rate : float
        Constant recovery rate applied both to the CDS bootstrap and to
        every obligor in the portfolio.
    discount_curve : DiscountCurve
        Risk-free discount curve.
    n_obligors : int
        Number of homogeneous obligors in the reference portfolio.
    correlation : float
        Common one-factor copula correlation assigned to every obligor.
    loss_unit : float
        Loss discretization bucket size passed to `CDOPricer`.
    n_quadrature_points : int, default=32
        Gauss-Hermite quadrature nodes passed to `CDOPricer`.
    payment_frequency : int, default=4
        Premium payments per annum passed to `CDOPricer`.
    n_integration_steps_per_year : int, default=12
        Protection leg integration steps per annum passed to `CDOPricer`.
    notional_per_obligor : float, default=1.0
        Reference notional assigned to each obligor.

    Returns
    -------
    CDOPricer
        Configured pricing engine over a homogeneous reference
        portfolio.
    """
    credit_curve = bootstrap_hazard_rates(
        cds_tenors=np.asarray(cds_tenors, dtype=np.float64),
        cds_spreads=np.asarray(cds_spreads, dtype=np.float64),
        recovery_rate=recovery_rate,
        discount_curve=discount_curve,
    )
    obligors = tuple(
        Obligor(
            name=f"Obligor_{i:03d}",
            notional=notional_per_obligor,
            recovery_rate=recovery_rate,
            correlation=correlation,
            credit_curve=credit_curve,
        )
        for i in range(n_obligors)
    )
    portfolio = CreditPortfolio(obligors=obligors)
    return CDOPricer(
        portfolio=portfolio,
        discount_curve=discount_curve,
        loss_unit=loss_unit,
        n_quadrature_points=n_quadrature_points,
        payment_frequency=payment_frequency,
        n_integration_steps_per_year=n_integration_steps_per_year,
    )


def calibration_table(
    tranche_labels: Sequence[str],
    market_spreads_bps: np.ndarray,
    model_spreads_bps: np.ndarray,
) -> tuple[pd.DataFrame, PricingResiduals]:
    """
    Tabulate a market-versus-model calibration comparison.

    Parameters
    ----------
    tranche_labels : Sequence[str]
        Tranche labels.
    market_spreads_bps : np.ndarray
        Quoted market par spreads, in basis points.
    model_spreads_bps : np.ndarray
        Model-implied par spreads, in basis points.

    Returns
    -------
    pd.DataFrame
        One row per tranche, with columns ``tranche``,
        ``market_spread_bps``, ``model_spread_bps``,
        ``absolute_error_bps``, ``relative_error_pct``.
    PricingResiduals
        The underlying residuals object (see
        :mod:`credit_copula.diagnostics`), for direct use with the
        monotonicity/RMSE diagnostics and warning generator.
    """
    residuals = compute_pricing_residuals(tranche_labels, market_spreads_bps, model_spreads_bps)
    table = pd.DataFrame(
        {
            "tranche": residuals.labels,
            "market_spread_bps": residuals.market_spread_bps,
            "model_spread_bps": residuals.model_spread_bps,
            "absolute_error_bps": residuals.absolute_error_bps,
            "relative_error_pct": residuals.relative_error_pct,
        }
    )
    return table, residuals


def base_tranche_expected_loss_curve(
    pricer: CDOPricer,
    detachment_points_pct: Sequence[float],
    maturity: float,
    correlations: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Evaluate the base tranche expected-loss curve :math:`\\text{EL}(K) = E[\\min(L, K)]`.

    This is the expected loss absorbed by a base tranche :math:`[0, K]`
    -- i.e. `CDOPricer.expected_tranche_loss` evaluated on `Tranche(0, K)`
    -- which is non-decreasing and concave in `K`. The curve is the
    basis of the no-arbitrage concavity check in
    :func:`credit_copula.diagnostics.check_base_tranche_convexity` and
    is also informative on its own as an "attachment/detachment loss
    profile" visualization.

    Parameters
    ----------
    pricer : CDOPricer
        Configured pricing engine.
    detachment_points_pct : Sequence[float]
        Detachment points as fractions of total portfolio notional,
        strictly increasing.
    maturity : float
        Evaluation horizon, in years.
    correlations : np.ndarray, optional
        Per-obligor correlation override; if `None`, each obligor's own
        correlation is used.

    Returns
    -------
    pd.DataFrame
        Columns ``detachment_pct`` and ``expected_loss`` (currency
        units).
    """
    index_notional = pricer.portfolio.total_notional
    expected_losses = np.array(
        [
            pricer.expected_tranche_loss(
                maturity, Tranche(0.0, float(detachment_pct) * index_notional), correlations
            )
            for detachment_pct in detachment_points_pct
        ]
    )
    return pd.DataFrame(
        {
            "detachment_pct": np.asarray(detachment_points_pct, dtype=np.float64),
            "expected_loss": expected_losses,
        }
    )


def correlation_sensitivity(
    pricer: CDOPricer,
    tranche_pillars_pct: Sequence[tuple[float, float]],
    maturity: float,
    correlation_grid: np.ndarray,
) -> pd.DataFrame:
    """
    Sweep tranche par spread across a grid of flat one-factor correlations.

    Parameters
    ----------
    pricer : CDOPricer
        Configured pricing engine.
    tranche_pillars_pct : Sequence[tuple[float, float]]
        Tranche attachment/detachment points as fractions of total
        notional.
    maturity : float
        Common maturity, in years.
    correlation_grid : np.ndarray
        Correlation values to evaluate, each in :math:`(0, 1)`.

    Returns
    -------
    pd.DataFrame
        Long-format table with columns ``correlation``, ``tranche``,
        ``par_spread_bps``.
    """
    index_notional = pricer.portfolio.total_notional
    n_obligors = pricer.portfolio.n_obligors
    records: list[dict[str, float | str]] = []

    for attachment_pct, detachment_pct in tranche_pillars_pct:
        tranche = Tranche(attachment_pct * index_notional, detachment_pct * index_notional)
        label = _tranche_label(attachment_pct, detachment_pct)
        for rho in correlation_grid:
            result = pricer.price_tranche(
                tranche, maturity, correlations=np.full(n_obligors, float(rho))
            )
            records.append({"correlation": float(rho), "tranche": label, "par_spread_bps": result.par_spread * 1.0e4})

    return pd.DataFrame.from_records(records)


def recovery_rate_sensitivity(
    cds_tenors: np.ndarray,
    cds_spreads: np.ndarray,
    discount_curve: DiscountCurve,
    n_obligors: int,
    correlation: float,
    loss_unit: float,
    tranche_pillars_pct: Sequence[tuple[float, float]],
    maturity: float,
    recovery_grid: np.ndarray,
    n_quadrature_points: int = 32,
    payment_frequency: int = 4,
    n_integration_steps_per_year: int = 12,
) -> pd.DataFrame:
    """
    Sweep tranche par spread across a grid of recovery rate assumptions.

    For each trial recovery rate, the single-name credit curve is
    re-bootstrapped from the same CDS par spread quotes (since recovery
    enters the CDS protection leg, the hazard rate consistent with a
    given spread quote depends on the assumed recovery), and the
    portfolio is rebuilt with that recovery applied uniformly.

    Notes
    -----
    Because the hazard rate is re-derived from the same market spreads
    at each trial recovery, this sweep conflates two offsetting effects:
    a higher recovery directly reduces loss given default, but it also
    forces the bootstrap to imply a *higher* hazard rate to match the
    unchanged CDS quote (since the CDS protection leg scales with
    :math:`(1-R) \\cdot \\lambda`). The resulting tranche spread is
    therefore not guaranteed to move monotonically with recovery; this
    sweep isolates the net effect actually observed in the market,
    rather than the loss-given-default effect in isolation.

    Parameters
    ----------
    cds_tenors, cds_spreads : np.ndarray
        Single-name CDS curve used to re-bootstrap hazard rates at each
        trial recovery rate.
    discount_curve : DiscountCurve
        Risk-free discount curve.
    n_obligors : int
        Number of homogeneous obligors.
    correlation : float
        Common one-factor copula correlation.
    loss_unit : float
        Loss discretization bucket size.
    tranche_pillars_pct : Sequence[tuple[float, float]]
        Tranche attachment/detachment points as fractions of total
        notional.
    maturity : float
        Common maturity, in years.
    recovery_grid : np.ndarray
        Recovery rate values to evaluate, each in :math:`[0, 1)`.
    n_quadrature_points, payment_frequency, n_integration_steps_per_year
        Numerical settings passed through to `CDOPricer`.

    Returns
    -------
    pd.DataFrame
        Long-format table with columns ``recovery_rate``, ``tranche``,
        ``par_spread_bps``.
    """
    records: list[dict[str, float | str]] = []
    for recovery_rate in recovery_grid:
        pricer = build_pricer(
            cds_tenors=cds_tenors,
            cds_spreads=cds_spreads,
            recovery_rate=float(recovery_rate),
            discount_curve=discount_curve,
            n_obligors=n_obligors,
            correlation=correlation,
            loss_unit=loss_unit,
            n_quadrature_points=n_quadrature_points,
            payment_frequency=payment_frequency,
            n_integration_steps_per_year=n_integration_steps_per_year,
        )
        index_notional = pricer.portfolio.total_notional
        for attachment_pct, detachment_pct in tranche_pillars_pct:
            tranche = Tranche(attachment_pct * index_notional, detachment_pct * index_notional)
            label = _tranche_label(attachment_pct, detachment_pct)
            result = pricer.price_tranche(tranche, maturity)
            records.append(
                {
                    "recovery_rate": float(recovery_rate),
                    "tranche": label,
                    "par_spread_bps": result.par_spread * 1.0e4,
                }
            )
    return pd.DataFrame.from_records(records)


def hazard_rate_sensitivity(
    pricer: CDOPricer,
    tranche_pillars_pct: Sequence[tuple[float, float]],
    maturity: float,
    multiplier_grid: np.ndarray,
) -> pd.DataFrame:
    """
    Sweep tranche par spread across a grid of parallel hazard rate shifts.

    Each trial scales the calibrated hazard rate curve by a multiplier
    (holding the curve's term structure shape, recovery rate, and
    correlation fixed), rebuilds the portfolio, and reprices the tranche
    structure. A multiplier of 1.0 reproduces `pricer`'s own pricing.

    Parameters
    ----------
    pricer : CDOPricer
        Configured pricing engine; its first obligor's credit curve,
        recovery rate, and correlation are used as the basis for the
        perturbed portfolios (the homogeneous-portfolio case relevant
        to this sweep).
    tranche_pillars_pct : Sequence[tuple[float, float]]
        Tranche attachment/detachment points as fractions of total
        notional.
    maturity : float
        Common maturity, in years.
    multiplier_grid : np.ndarray
        Multiplicative factors applied to the hazard rate curve, each
        strictly positive.

    Returns
    -------
    pd.DataFrame
        Long-format table with columns ``hazard_rate_multiplier``,
        ``tranche``, ``par_spread_bps``.
    """
    reference_obligor = pricer.portfolio.obligors[0]
    base_curve = reference_obligor.credit_curve
    n_obligors = pricer.portfolio.n_obligors

    records: list[dict[str, float | str]] = []
    for multiplier in multiplier_grid:
        scaled_curve = CreditCurve(
            tenors=base_curve.tenors,
            hazard_rates=base_curve.hazard_rates * float(multiplier),
            recovery_rate=base_curve.recovery_rate,
        )
        obligors = tuple(
            Obligor(
                name=f"Obligor_{i:03d}",
                notional=reference_obligor.notional,
                recovery_rate=reference_obligor.recovery_rate,
                correlation=reference_obligor.correlation,
                credit_curve=scaled_curve,
            )
            for i in range(n_obligors)
        )
        scaled_pricer = CDOPricer(
            portfolio=CreditPortfolio(obligors=obligors),
            discount_curve=pricer.discount_curve,
            loss_unit=pricer.loss_unit,
            n_quadrature_points=pricer.n_quadrature_points,
            payment_frequency=pricer.payment_frequency,
            n_integration_steps_per_year=pricer.n_integration_steps_per_year,
        )
        index_notional = scaled_pricer.portfolio.total_notional
        for attachment_pct, detachment_pct in tranche_pillars_pct:
            tranche = Tranche(attachment_pct * index_notional, detachment_pct * index_notional)
            label = _tranche_label(attachment_pct, detachment_pct)
            result = scaled_pricer.price_tranche(tranche, maturity)
            records.append(
                {
                    "hazard_rate_multiplier": float(multiplier),
                    "tranche": label,
                    "par_spread_bps": result.par_spread * 1.0e4,
                }
            )
    return pd.DataFrame.from_records(records)
