"""
Calibration and model-consistency diagnostics.

This module implements the quantitative checks practitioners apply to
validate a calibrated one-factor Gaussian copula model before relying
on it for risk management: pricing accuracy against market quotes,
no-arbitrage shape constraints on the base correlation skew, and
monotonicity properties that any valid loss distribution must satisfy.
None of these checks alter pricing behaviour; they are purely
diagnostic and are designed to make a poorly-calibrated or numerically
unstable result visible rather than silently accepted.

All functions are pure and operate on plain NumPy arrays, so they can
be unit-tested independently of any particular pricer configuration
and reused directly by the dashboard's calibration section.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from credit_copula.portfolio import discretize_loss_given_default

__all__ = [
    "PricingResiduals",
    "ConsistencyCheck",
    "CalibrationDiagnostics",
    "DiscretizationErrorEstimate",
    "QuadratureConvergenceResult",
    "compute_pricing_residuals",
    "root_mean_square_error",
    "maximum_absolute_error",
    "check_expected_loss_time_monotonicity",
    "check_base_tranche_convexity",
    "check_base_correlation_monotonicity",
    "base_correlation_curvature",
    "check_probability_mass_conservation",
    "check_probabilities_non_negative",
    "check_expected_tranche_loss_bounds",
    "check_correlation_bounds",
    "estimate_loss_discretization_error",
    "assess_quadrature_convergence",
    "generate_calibration_warnings",
    "summarize_calibration",
]


@dataclass(frozen=True)
class PricingResiduals:
    """
    Market-versus-model pricing comparison for a set of tranches.

    Attributes
    ----------
    labels : np.ndarray
        Tranche labels (object array of strings).
    market_spread_bps : np.ndarray
        Quoted market par spreads, in basis points.
    model_spread_bps : np.ndarray
        Model-implied par spreads, in basis points.
    absolute_error_bps : np.ndarray
        ``model_spread_bps - market_spread_bps``.
    relative_error_pct : np.ndarray
        Absolute error as a percentage of the market spread:
        ``100 * absolute_error_bps / market_spread_bps``.
    """

    labels: np.ndarray
    market_spread_bps: np.ndarray
    model_spread_bps: np.ndarray
    absolute_error_bps: np.ndarray
    relative_error_pct: np.ndarray


@dataclass(frozen=True)
class ConsistencyCheck:
    """
    Outcome of a single monotonicity or convexity consistency check.

    Attributes
    ----------
    passed : bool
        Whether the check was satisfied within `tolerance`.
    max_violation : float
        Largest signed violation magnitude observed (zero or negative
        if the check passed everywhere).
    violation_indices : np.ndarray
        Indices into the input arrays at which a violation occurred.
    """

    passed: bool
    max_violation: float
    violation_indices: np.ndarray


def compute_pricing_residuals(
    labels: list[str] | np.ndarray,
    market_spreads_bps: np.ndarray,
    model_spreads_bps: np.ndarray,
) -> PricingResiduals:
    """
    Compare market and model-implied tranche par spreads.

    Parameters
    ----------
    labels : list[str] or np.ndarray
        Tranche labels, aligned with the spread arrays.
    market_spreads_bps : np.ndarray
        Quoted market par spreads, in basis points.
    model_spreads_bps : np.ndarray
        Model-implied par spreads, in basis points.

    Returns
    -------
    PricingResiduals
        Absolute and relative pricing errors per tranche.

    Raises
    ------
    ValueError
        If the input array lengths are inconsistent.
    """
    labels_arr = np.asarray(labels, dtype=object)
    market = np.asarray(market_spreads_bps, dtype=np.float64)
    model = np.asarray(model_spreads_bps, dtype=np.float64)
    if not (labels_arr.shape == market.shape == model.shape):
        raise ValueError("labels, market_spreads_bps and model_spreads_bps must have equal length")

    absolute_error = model - market
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_error = np.where(market != 0.0, 100.0 * absolute_error / market, np.nan)

    return PricingResiduals(
        labels=labels_arr,
        market_spread_bps=market,
        model_spread_bps=model,
        absolute_error_bps=absolute_error,
        relative_error_pct=relative_error,
    )


def root_mean_square_error(residuals: np.ndarray) -> float:
    """
    Root-mean-square error of a residual array.

    .. math::
        \\text{RMSE} = \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} r_i^2}

    Parameters
    ----------
    residuals : np.ndarray
        Pricing residuals (e.g. `PricingResiduals.absolute_error_bps`).

    Returns
    -------
    float
        Root-mean-square error, in the same units as `residuals`.
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    return float(np.sqrt(np.mean(residuals**2)))


def maximum_absolute_error(residuals: np.ndarray) -> float:
    """
    Largest absolute pricing residual.

    Parameters
    ----------
    residuals : np.ndarray
        Pricing residuals.

    Returns
    -------
    float
        ``max(abs(residuals))``.
    """
    return float(np.max(np.abs(np.asarray(residuals, dtype=np.float64))))


def check_expected_loss_time_monotonicity(
    times: np.ndarray, expected_tranche_loss: np.ndarray, tolerance: float = 1.0e-8
) -> ConsistencyCheck:
    """
    Verify that expected tranche loss is non-decreasing in time.

    Under risk-neutral valuation, expected tranche loss is a
    cumulative expectation of realized losses and therefore must be
    non-decreasing in `t`:

    .. math::
        \\text{ETL}(t_{i+1}) \\ge \\text{ETL}(t_i) \\quad \\forall i

    A violation indicates a numerical inconsistency (e.g. insufficient
    loss discretization granularity or quadrature resolution) rather
    than a modelling choice, since no economically valid loss process
    can lose expected loss over time.

    Parameters
    ----------
    times : np.ndarray
        Strictly increasing evaluation times.
    expected_tranche_loss : np.ndarray
        Expected tranche loss evaluated at `times`, aligned in shape.
    tolerance : float, default=1e-8
        Negative increments smaller in magnitude than `tolerance` are
        treated as numerical noise rather than violations.

    Returns
    -------
    ConsistencyCheck
        Result of the monotonicity check.
    """
    times_arr = np.asarray(times, dtype=np.float64)
    if np.any(np.diff(times_arr) <= 0.0):
        raise ValueError("times must be strictly increasing")
    etl = np.asarray(expected_tranche_loss, dtype=np.float64)
    increments = np.diff(etl)
    violations = np.where(increments < -tolerance)[0]
    max_violation = float(np.min(increments)) if increments.size > 0 else 0.0
    return ConsistencyCheck(
        passed=violations.size == 0,
        max_violation=min(max_violation, 0.0),
        violation_indices=violations,
    )


def check_base_tranche_convexity(
    detachment_points: np.ndarray, base_expected_losses: np.ndarray, tolerance: float = 1.0e-6
) -> ConsistencyCheck:
    """
    Verify the no-arbitrage shape of the base tranche expected-loss curve.

    The expected loss absorbed by a base tranche :math:`[0, K]` is
    :math:`\\text{EL}(K) = E[\\min(L, K)]`, where `L` is total portfolio
    loss (this is exactly the quantity returned by
    `CDOPricer.expected_tranche_loss` for a tranche attached at zero;
    see :func:`credit_copula.reporting.base_tranche_expected_loss_curve`).
    Writing :math:`F` for the CDF of `L`, :math:`\\text{EL}'(K) = P(L > K)
    = 1 - F(K) \\in [0, 1]` is non-increasing in `K`, so `EL` is both
    non-decreasing and concave in the detachment point:

    .. math::
        \\frac{\\partial \\text{EL}}{\\partial K} \\ge 0,
        \\qquad
        \\frac{\\partial^2 \\text{EL}}{\\partial K^2} \\le 0

    A violation of either condition implies a negative implied density
    for the portfolio loss distribution between the corresponding
    pillars, i.e. an internally arbitrageable set of base correlation
    quotes (Andersen, Sidenius & Basu, 2003, Sec. 5; O'Kane, 2008, Ch.
    15). This check operates on discretized finite differences and is
    therefore a necessary, not sufficient, no-arbitrage condition.

    Parameters
    ----------
    detachment_points : np.ndarray
        Strictly increasing base tranche detachment points.
    base_expected_losses : np.ndarray
        :math:`E[\\min(L, K)]` at each detachment point `K`, aligned in
        shape with `detachment_points`.
    tolerance : float, default=1e-6
        Magnitude below which a violation is treated as numerical
        noise rather than a genuine inconsistency.

    Returns
    -------
    ConsistencyCheck
        Combined result of the monotonicity (non-decreasing) and
        concavity checks; `violation_indices` indexes into the interior
        points of `detachment_points` (the concavity check has no value
        at the endpoints).
    """
    detachments = np.asarray(detachment_points, dtype=np.float64)
    losses = np.asarray(base_expected_losses, dtype=np.float64)

    first_differences = np.diff(losses) / np.diff(detachments)
    monotonicity_violations = np.where(first_differences < -tolerance)[0]

    second_differences = np.diff(first_differences)
    concavity_violations = np.where(second_differences > tolerance)[0] + 1

    all_violations = np.union1d(monotonicity_violations, concavity_violations)
    worst_monotonicity = float(-np.min(first_differences)) if first_differences.size > 0 else 0.0
    worst_concavity = float(np.max(second_differences)) if second_differences.size > 0 else 0.0
    max_violation = max(worst_monotonicity, worst_concavity, 0.0)

    return ConsistencyCheck(
        passed=all_violations.size == 0,
        max_violation=max_violation,
        violation_indices=all_violations.astype(np.int64),
    )


def check_base_correlation_monotonicity(
    detachment_points: np.ndarray, base_correlations: np.ndarray, tolerance: float = 1.0e-6
) -> ConsistencyCheck:
    """
    Check whether the base correlation skew follows the conventional increasing shape.

    Empirically, base correlation curves observed in the CDX/iTraxx
    market are increasing in detachment point. This is a market
    convention/stylized fact, not a no-arbitrage requirement -- unlike
    :func:`check_base_tranche_convexity`, a non-monotonic base
    correlation curve does not by itself imply an arbitrage. It is,
    however, a useful early indicator of calibration instability or
    poor-quality input quotes, and is reported as a separate, explicitly
    labelled diagnostic rather than conflated with the arbitrage checks.

    Parameters
    ----------
    detachment_points : np.ndarray
        Strictly increasing detachment points.
    base_correlations : np.ndarray
        Bootstrapped base correlations, aligned in shape.
    tolerance : float, default=1e-6
        Decrease smaller in magnitude than `tolerance` is treated as
        numerical noise.

    Returns
    -------
    ConsistencyCheck
        Result of the monotonicity check.
    """
    del detachment_points
    correlations = np.asarray(base_correlations, dtype=np.float64)
    increments = np.diff(correlations)
    violations = np.where(increments < -tolerance)[0]
    max_violation = float(np.min(increments)) if increments.size > 0 else 0.0
    return ConsistencyCheck(
        passed=violations.size == 0,
        max_violation=min(max_violation, 0.0),
        violation_indices=violations,
    )


def base_correlation_curvature(detachment_points: np.ndarray, base_correlations: np.ndarray) -> float:
    """
    Smoothness/stability metric for a base correlation curve.

    Defined as the root-mean-square of the discrete second derivative
    of the base correlation curve with respect to detachment point:

    .. math::
        \\text{Curvature} = \\sqrt{\\frac{1}{n-2} \\sum_{i=2}^{n-1}
            \\left(\\frac{\\rho_{i+1} - 2\\rho_i + \\rho_{i-1}}
                       {(\\Delta K)^2}\\right)^2}

    assuming roughly evenly spaced detachment points. Large values
    indicate an erratic, unstable skew -- often a symptom of
    inconsistent or noisy market quotes rather than a genuine model
    or market feature -- and warrant closer inspection before the curve
    is used for interpolation-based pricing of bespoke tranches.

    Parameters
    ----------
    detachment_points : np.ndarray
        Strictly increasing detachment points, at least 3 pillars.
    base_correlations : np.ndarray
        Bootstrapped base correlations, aligned in shape.

    Returns
    -------
    float
        Root-mean-square curvature. Returns 0.0 if fewer than 3 pillars
        are supplied (curvature is undefined).
    """
    detachments = np.asarray(detachment_points, dtype=np.float64)
    correlations = np.asarray(base_correlations, dtype=np.float64)
    if detachments.size < 3:
        return 0.0

    step = np.diff(detachments)
    average_step = float(np.mean(step))
    second_derivative = np.diff(correlations, n=2) / (average_step**2)
    return float(np.sqrt(np.mean(second_derivative**2)))


def generate_calibration_warnings(
    residuals: PricingResiduals,
    expected_loss_time_check: ConsistencyCheck | None = None,
    base_tranche_convexity_check: ConsistencyCheck | None = None,
    base_correlation_monotonicity_check: ConsistencyCheck | None = None,
    curvature: float | None = None,
    rmse_threshold_bps: float = 5.0,
    max_error_threshold_bps: float = 15.0,
    curvature_threshold: float = 0.5,
) -> list[str]:
    """
    Aggregate calibration diagnostics into human-readable warning messages.

    Parameters
    ----------
    residuals : PricingResiduals
        Market-versus-model pricing comparison.
    expected_loss_time_check : ConsistencyCheck, optional
        Result of :func:`check_expected_loss_time_monotonicity`.
    base_tranche_convexity_check : ConsistencyCheck, optional
        Result of :func:`check_base_tranche_convexity`.
    base_correlation_monotonicity_check : ConsistencyCheck, optional
        Result of :func:`check_base_correlation_monotonicity`.
    curvature : float, optional
        Result of :func:`base_correlation_curvature`.
    rmse_threshold_bps : float, default=5.0
        RMSE above this level triggers a warning.
    max_error_threshold_bps : float, default=15.0
        Any single absolute error above this level triggers a warning.
    curvature_threshold : float, default=0.5
        Curvature above this level triggers an instability warning.

    Returns
    -------
    list[str]
        Zero or more warning messages, ordered by severity of the
        underlying check (pricing accuracy first, then no-arbitrage
        violations, then stability/shape observations).
    """
    warnings: list[str] = []

    rmse = root_mean_square_error(residuals.absolute_error_bps)
    max_error = maximum_absolute_error(residuals.absolute_error_bps)
    if rmse > rmse_threshold_bps:
        warnings.append(
            f"Calibration RMSE of {rmse:.2f} bps exceeds the {rmse_threshold_bps:.1f} bps "
            "threshold: model-implied spreads do not closely track market quotes."
        )
    if max_error > max_error_threshold_bps:
        worst_idx = int(np.argmax(np.abs(residuals.absolute_error_bps)))
        worst_label = residuals.labels[worst_idx]
        warnings.append(
            f"Maximum absolute pricing error of {max_error:.2f} bps (tranche {worst_label}) "
            f"exceeds the {max_error_threshold_bps:.1f} bps threshold."
        )

    if expected_loss_time_check is not None and not expected_loss_time_check.passed:
        warnings.append(
            "Expected tranche loss is not monotonically non-decreasing in time "
            f"(largest violation: {expected_loss_time_check.max_violation:.3e}); "
            "increase loss discretization or quadrature resolution."
        )

    if base_tranche_convexity_check is not None and not base_tranche_convexity_check.passed:
        warnings.append(
            "Base tranche expected-loss curve violates the no-arbitrage monotonicity/convexity "
            f"condition (largest violation: {base_tranche_convexity_check.max_violation:.3e}); "
            "the implied base correlation skew may not be arbitrage-free."
        )

    if base_correlation_monotonicity_check is not None and not base_correlation_monotonicity_check.passed:
        warnings.append(
            "Base correlation curve is not monotonically increasing in detachment point "
            f"(largest violation: {base_correlation_monotonicity_check.max_violation:.4f}); "
            "this departs from the conventional market skew shape."
        )

    if curvature is not None and curvature > curvature_threshold:
        warnings.append(
            f"Base correlation curve curvature ({curvature:.3f}) exceeds the stability "
            f"threshold ({curvature_threshold:.3f}); the skew may be unreliable for "
            "interpolation-based bespoke tranche pricing."
        )

    return warnings


@dataclass(frozen=True)
class CalibrationDiagnostics:
    """
    Aggregate calibration diagnostics for a base correlation bootstrap.

    Bundles pricing-accuracy diagnostics (residuals, RMSE, maximum
    error), per-pillar root-finder convergence diagnostics, and
    no-arbitrage/stability checks into a single structured object, so
    that calibration quality can be inspected and reported without
    re-deriving it from the underlying arrays at each call site (see
    :func:`summarize_calibration`).

    Attributes
    ----------
    residuals : PricingResiduals
        Market-versus-model pricing comparison.
    rmse_bps : float
        Root-mean-square pricing error, in basis points.
    max_absolute_error_bps : float
        Largest single-tranche absolute pricing error, in basis points.
    correlation_bounds : tuple[float, float]
        Root-finder search bracket used for every pillar.
    root_tolerance : float
        Root-finder absolute tolerance (`xtol`) used for every pillar.
    pillar_iterations : np.ndarray
        Brent iteration count consumed at each base correlation pillar.
    pillar_converged : np.ndarray
        Boolean convergence flag reported at each pillar.
    pillar_residuals : np.ndarray
        Root-finder objective function value at the solved correlation,
        at each pillar; should be close to zero at genuine convergence.
    expected_loss_time_check : ConsistencyCheck or None
        Result of :func:`check_expected_loss_time_monotonicity`, if
        evaluated.
    base_tranche_convexity_check : ConsistencyCheck or None
        Result of :func:`check_base_tranche_convexity`, if evaluated.
    base_correlation_monotonicity_check : ConsistencyCheck or None
        Result of :func:`check_base_correlation_monotonicity`, if
        evaluated.
    base_correlation_curvature : float or None
        Result of :func:`base_correlation_curvature`, if evaluated.
    warnings : list[str]
        Aggregated plain-language warnings (see
        :func:`generate_calibration_warnings`), plus any convergence
        warning raised by `summarize_calibration` itself.
    """

    residuals: PricingResiduals
    rmse_bps: float
    max_absolute_error_bps: float
    correlation_bounds: tuple[float, float]
    root_tolerance: float
    pillar_iterations: np.ndarray
    pillar_converged: np.ndarray
    pillar_residuals: np.ndarray
    expected_loss_time_check: ConsistencyCheck | None
    base_tranche_convexity_check: ConsistencyCheck | None
    base_correlation_monotonicity_check: ConsistencyCheck | None
    base_correlation_curvature: float | None
    warnings: list[str]

    @property
    def all_pillars_converged(self) -> bool:
        """Whether every base correlation pillar's root-finder reported convergence."""
        return bool(np.all(self.pillar_converged))

    @property
    def is_reliable(self) -> bool:
        """Whether calibration converged everywhere and triggered no diagnostic warnings."""
        return self.all_pillars_converged and len(self.warnings) == 0


def summarize_calibration(
    tranche_labels: list[str],
    market_spreads_bps: np.ndarray,
    model_spreads_bps: np.ndarray,
    pillar_iterations: np.ndarray,
    pillar_converged: np.ndarray,
    pillar_residuals: np.ndarray,
    correlation_bounds: tuple[float, float],
    root_tolerance: float,
    expected_loss_time_check: ConsistencyCheck | None = None,
    base_tranche_convexity_check: ConsistencyCheck | None = None,
    base_correlation_monotonicity_check: ConsistencyCheck | None = None,
    curvature: float | None = None,
) -> CalibrationDiagnostics:
    """
    Assemble a complete :class:`CalibrationDiagnostics` record from its constituent checks.

    Parameters
    ----------
    tranche_labels : list[str]
        Tranche labels, aligned with the spread arrays.
    market_spreads_bps, model_spreads_bps : np.ndarray
        Market and model-implied par spreads, in basis points.
    pillar_iterations, pillar_converged, pillar_residuals : np.ndarray
        Per-pillar root-finder diagnostics from the base correlation
        bootstrap (see
        :class:`credit_copula.base_correlation.BaseCorrelationDiagnostics`).
    correlation_bounds : tuple[float, float]
        Root-finder search bracket used for every pillar.
    root_tolerance : float
        Root-finder absolute tolerance (`xtol`) used for every pillar.
    expected_loss_time_check, base_tranche_convexity_check,
    base_correlation_monotonicity_check, curvature
        Pre-computed diagnostic check results; see
        :func:`generate_calibration_warnings` for how each contributes
        to the aggregated warning list.

    Returns
    -------
    CalibrationDiagnostics
        Complete calibration diagnostics record.
    """
    residuals = compute_pricing_residuals(tranche_labels, market_spreads_bps, model_spreads_bps)
    rmse = root_mean_square_error(residuals.absolute_error_bps)
    max_error = maximum_absolute_error(residuals.absolute_error_bps)

    warnings = generate_calibration_warnings(
        residuals,
        expected_loss_time_check=expected_loss_time_check,
        base_tranche_convexity_check=base_tranche_convexity_check,
        base_correlation_monotonicity_check=base_correlation_monotonicity_check,
        curvature=curvature,
    )
    if not np.all(pillar_converged):
        n_failed = int(np.sum(~np.asarray(pillar_converged, dtype=bool)))
        warnings.append(
            f"{n_failed} of {len(pillar_converged)} base correlation root-finder calls did not "
            "report convergence within the configured iteration budget."
        )

    return CalibrationDiagnostics(
        residuals=residuals,
        rmse_bps=rmse,
        max_absolute_error_bps=max_error,
        correlation_bounds=correlation_bounds,
        root_tolerance=root_tolerance,
        pillar_iterations=np.asarray(pillar_iterations),
        pillar_converged=np.asarray(pillar_converged),
        pillar_residuals=np.asarray(pillar_residuals),
        expected_loss_time_check=expected_loss_time_check,
        base_tranche_convexity_check=base_tranche_convexity_check,
        base_correlation_monotonicity_check=base_correlation_monotonicity_check,
        base_correlation_curvature=curvature,
        warnings=warnings,
    )


def check_probability_mass_conservation(
    probabilities: np.ndarray, tolerance: float = 1.0e-6
) -> ConsistencyCheck:
    """
    Verify that a discrete probability distribution sums to one.

    Parameters
    ----------
    probabilities : np.ndarray
        Probability mass function (e.g. the output of
        :func:`credit_copula.portfolio.loss_distribution`).
    tolerance : float, default=1e-6
        Maximum allowed deviation of the sum from 1.0. The default is
        looser than typical floating-point round-off, since loss
        discretization and quadrature truncation also contribute to
        the total.

    Returns
    -------
    ConsistencyCheck
        `passed` is `True` iff ``abs(sum(probabilities) - 1) <= tolerance``;
        `max_violation` holds the signed deviation from 1.0;
        `violation_indices` is always empty (this is a single, scalar
        check, not a per-element one).
    """
    total = float(np.sum(np.asarray(probabilities, dtype=np.float64)))
    deviation = total - 1.0
    return ConsistencyCheck(
        passed=abs(deviation) <= tolerance,
        max_violation=deviation,
        violation_indices=np.array([], dtype=np.int64),
    )


def check_probabilities_non_negative(
    probabilities: np.ndarray, tolerance: float = 1.0e-10
) -> ConsistencyCheck:
    """
    Verify that no probability mass point is negative beyond numerical tolerance.

    The Andersen-Sidenius-Basu recursion (see
    :func:`credit_copula.portfolio.conditional_loss_distribution`) is
    analytically guaranteed to produce non-negative probabilities;
    a violation here indicates floating-point cancellation, typically
    arising from an extreme correlation (close to 0 or 1) combined with
    coarse loss discretization, rather than a modelling error.

    Parameters
    ----------
    probabilities : np.ndarray
        Probability mass function.
    tolerance : float, default=1e-10
        Magnitude below which a negative value is treated as numerical
        noise.

    Returns
    -------
    ConsistencyCheck
        `max_violation` is the magnitude of the most negative value
        found (0.0 if none).
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    violations = np.where(probs < -tolerance)[0]
    max_violation = float(-np.min(probs)) if probs.size > 0 else 0.0
    return ConsistencyCheck(
        passed=violations.size == 0,
        max_violation=max(max_violation, 0.0),
        violation_indices=violations,
    )


def check_expected_tranche_loss_bounds(
    expected_tranche_loss: np.ndarray, tranche_notional: float, tolerance: float = 1.0e-8
) -> ConsistencyCheck:
    """
    Verify that expected tranche loss lies within :math:`[0, N]`.

    Expected tranche loss is the expectation of a payoff bounded
    between zero and the tranche notional `N` (see
    :func:`credit_copula.tranche.tranche_loss_payoff`); the expectation
    of a bounded random variable inherits the same bounds.

    Parameters
    ----------
    expected_tranche_loss : np.ndarray or float
        Expected tranche loss value(s) to check.
    tranche_notional : float
        Tranche notional `N = D - A`.
    tolerance : float, default=1e-8
        Tolerance for boundary violations.

    Returns
    -------
    ConsistencyCheck
        `max_violation` is the largest out-of-bounds excursion found
        (0.0 if none).
    """
    etl = np.atleast_1d(np.asarray(expected_tranche_loss, dtype=np.float64))
    below = np.where(etl < -tolerance)[0]
    above = np.where(etl > tranche_notional + tolerance)[0]
    violations = np.union1d(below, above)
    excursion_below = float(-np.min(etl)) if etl.size > 0 else 0.0
    excursion_above = float(np.max(etl) - tranche_notional) if etl.size > 0 else 0.0
    max_violation = max(excursion_below, excursion_above, 0.0)
    return ConsistencyCheck(
        passed=violations.size == 0,
        max_violation=max_violation,
        violation_indices=violations.astype(np.int64),
    )


def check_correlation_bounds(correlations: np.ndarray, tolerance: float = 1.0e-9) -> ConsistencyCheck:
    """
    Verify that one-factor copula correlations lie within :math:`[0, 1)`.

    Parameters
    ----------
    correlations : np.ndarray
        Correlation value(s) to check (e.g. a calibrated base
        correlation curve).
    tolerance : float, default=1e-9
        Tolerance for boundary violations.

    Returns
    -------
    ConsistencyCheck
        `max_violation` is the largest out-of-bounds excursion found
        (0.0 if none).
    """
    corr = np.atleast_1d(np.asarray(correlations, dtype=np.float64))
    violations = np.where((corr < -tolerance) | (corr > 1.0 + tolerance))[0]
    excursion_below = float(-np.min(corr)) if corr.size > 0 else 0.0
    excursion_above = float(np.max(corr) - 1.0) if corr.size > 0 else 0.0
    max_violation = max(excursion_below, excursion_above, 0.0)
    return ConsistencyCheck(
        passed=violations.size == 0,
        max_violation=max_violation,
        violation_indices=violations,
    )


@dataclass(frozen=True)
class DiscretizationErrorEstimate:
    """
    Bias introduced by rounding obligor loss-given-default onto the loss-unit grid.

    Attributes
    ----------
    exact_expected_loss : float
        Exact expected portfolio loss, :math:`\\sum_i Q_i^D \\cdot \\ell_i`,
        computed from continuous (undiscretized) loss-given-default
        amounts. Expectation is linear and therefore independent of the
        dependence structure (correlation), so this reference value
        does not require specifying a copula.
    discretized_expected_loss : float
        The same quantity computed from loss-given-default amounts
        rounded to the nearest `loss_unit` multiple (see
        :func:`credit_copula.portfolio.discretize_loss_given_default`),
        isolating the bias introduced by that rounding step from any
        bias introduced by the recursive aggregation or quadrature
        integration steps.
    absolute_error : float
        ``discretized_expected_loss - exact_expected_loss``.
    relative_error_pct : float
        Absolute error as a percentage of `exact_expected_loss`.
    """

    exact_expected_loss: float
    discretized_expected_loss: float
    absolute_error: float
    relative_error_pct: float


def estimate_loss_discretization_error(
    default_probabilities: np.ndarray, loss_given_defaults: np.ndarray, loss_unit: float
) -> DiscretizationErrorEstimate:
    """
    Estimate the bias in expected portfolio loss introduced by loss-unit discretization.

    Parameters
    ----------
    default_probabilities : np.ndarray
        Marginal cumulative default probabilities, shape ``(N,)``.
    loss_given_defaults : np.ndarray
        Continuous per-obligor loss-given-default amounts (currency
        units), shape ``(N,)``.
    loss_unit : float
        Discretization bucket size passed to the pricing engine.

    Returns
    -------
    DiscretizationErrorEstimate
        Exact versus discretized expected portfolio loss and the
        resulting bias.

    Notes
    -----
    Because :math:`E[L] = \\sum_i Q_i^D \\ell_i` is correlation-
    independent, this isolates the discretization step's contribution
    to pricing bias from quadrature error (see
    :func:`assess_quadrature_convergence`) and from the recursive
    aggregation algorithm itself (which is exact given its discretized
    inputs). A large relative error indicates `loss_unit` is too coarse
    relative to typical single-name loss-given-default amounts.
    """
    default_probabilities = np.asarray(default_probabilities, dtype=np.float64)
    loss_given_defaults = np.asarray(loss_given_defaults, dtype=np.float64)

    exact = float(np.sum(default_probabilities * loss_given_defaults))
    discretized_lgd = discretize_loss_given_default(loss_given_defaults, loss_unit) * loss_unit
    discretized = float(np.sum(default_probabilities * discretized_lgd))
    absolute_error = discretized - exact
    relative_error_pct = 100.0 * absolute_error / exact if exact != 0.0 else 0.0

    return DiscretizationErrorEstimate(
        exact_expected_loss=exact,
        discretized_expected_loss=discretized,
        absolute_error=absolute_error,
        relative_error_pct=relative_error_pct,
    )


@dataclass(frozen=True)
class QuadratureConvergenceResult:
    """
    Sensitivity of a model output to Gauss-Hermite quadrature order.

    Attributes
    ----------
    node_counts : np.ndarray
        Quadrature node counts evaluated, in increasing order.
    values : np.ndarray
        The model output (e.g. a tranche's expected loss) evaluated at
        each node count.
    successive_differences : np.ndarray
        ``np.diff(values)``: the change in output as node count
        increases at each step.
    converged : bool
        Whether the final successive difference's magnitude is within
        the requested tolerance.
    """

    node_counts: np.ndarray
    values: np.ndarray
    successive_differences: np.ndarray
    converged: bool


def assess_quadrature_convergence(
    values_by_node_count: dict[int, float], tolerance: float = 1.0e-6
) -> QuadratureConvergenceResult:
    """
    Assess convergence of a model output with respect to Gauss-Hermite quadrature order.

    Parameters
    ----------
    values_by_node_count : dict[int, float]
        Mapping from quadrature node count to the corresponding model
        output (e.g. ``{16: 0.1234, 32: 0.1231, 64: 0.1231}``),
        computed by repricing with
        :class:`credit_copula.pricer.CDOPricer` configured at each
        node count.
    tolerance : float, default=1e-6
        Convergence threshold on the magnitude of the last successive
        difference.

    Returns
    -------
    QuadratureConvergenceResult
        Convergence diagnostic across the supplied node counts.

    Raises
    ------
    ValueError
        If fewer than two node counts are supplied.
    """
    if len(values_by_node_count) < 2:
        raise ValueError("at least two node counts are required to assess convergence")
    node_counts = np.array(sorted(values_by_node_count))
    values = np.array([values_by_node_count[n] for n in node_counts])
    successive_differences = np.diff(values)
    converged = bool(abs(successive_differences[-1]) <= tolerance)
    return QuadratureConvergenceResult(
        node_counts=node_counts,
        values=values,
        successive_differences=successive_differences,
        converged=converged,
    )
