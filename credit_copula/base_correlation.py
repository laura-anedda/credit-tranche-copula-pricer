"""
Base correlation bootstrapping from market tranche quotes.

Base correlation is the market-standard methodology (McGinty, Beinstein,
Ahluwalia & Watts, 2004, JPMorgan) for reconciling the one-factor
Gaussian copula model with the observed pricing of multiple tranches on
the same underlying index. Rather than searching for a single
"compound" correlation that simultaneously reprices every tranche
(which generally does not exist, since the market exhibits a
correlation skew), base correlation reframes every traded tranche as a
*difference of two base tranches* attached at 0:

.. math::
    \\text{Tranche}[A, D] = \\text{BaseTranche}[0, D] - \\text{BaseTranche}[0, A]

A single flat correlation :math:`\\rho_K` is then calibrated
independently for each base tranche detachment point `K`, such that the
model-implied par spread (or upfront, for the equity tranche) of the
base tranche :math:`[0, K]`, priced with *every* obligor sharing the
same correlation :math:`\\rho_K`, matches the market quote for that
detachment point. The resulting set of pairs :math:`\\{(K_i, \\rho_{K_i})\\}`
is the base correlation curve (skew), from which any other tranche
:math:`[A, D]` with attachment/detachment points not necessarily
matching the quoted pillars can be priced by interpolating the base
correlation curve and applying the differencing identity above.

This module bootstraps the base correlation curve only; pricing an
arbitrary tranche via base correlation interpolation is a downstream
application left to the caller (interpolation methodology -- linear,
or with extrapolation rules -- is itself a modelling choice that varies
across market participants and is intentionally not hard-coded here).

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from credit_copula import numerics
from credit_copula.pricer import CDOPricer
from credit_copula.tranche import Tranche

__all__ = [
    "BaseCorrelationCurve",
    "BaseCorrelationDiagnostics",
    "bootstrap_base_correlation",
    "bootstrap_base_correlation_from_standard_tranches",
]


@dataclass(frozen=True)
class BaseCorrelationDiagnostics:
    """
    Per-pillar root-finder convergence diagnostics for a base correlation bootstrap.

    Attributes
    ----------
    detachment_points : np.ndarray
        Detachment points, aligned with the other attributes.
    iterations : np.ndarray
        Number of Brent iterations consumed at each pillar.
    converged : np.ndarray
        Boolean convergence flag reported by the root-finder at each
        pillar.
    residual : np.ndarray
        Objective function value at the solved correlation; should be
        close to zero at convergence; values markedly larger than the
        solver tolerance indicate the calibration did not fully
        converge.
    function_calls : np.ndarray
        Number of objective function evaluations consumed at each
        pillar.
    """

    detachment_points: np.ndarray
    iterations: np.ndarray
    converged: np.ndarray
    residual: np.ndarray
    function_calls: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return the diagnostics as a column-oriented dict, suitable for `pandas.DataFrame`."""
        return {
            "detachment_points": self.detachment_points,
            "iterations": self.iterations,
            "converged": self.converged,
            "residual": self.residual,
            "function_calls": self.function_calls,
        }


@dataclass(frozen=True)
class BaseCorrelationCurve:
    """
    Bootstrapped base correlation skew.

    Attributes
    ----------
    detachment_points : np.ndarray
        Base tranche detachment points `K_i` (currency units),
        strictly increasing.
    correlations : np.ndarray
        Bootstrapped flat correlations :math:`\\rho_{K_i} \\in (0, 1)`,
        aligned with `detachment_points`.
    """

    detachment_points: np.ndarray
    correlations: np.ndarray

    def interpolate(self, detachment: float) -> float:
        """
        Linearly interpolate the base correlation at an arbitrary detachment point.

        Parameters
        ----------
        detachment : float
            Detachment point at which to interpolate (currency units).

        Returns
        -------
        float
            Interpolated base correlation. Flat extrapolation is
            applied outside the bootstrapped range, consistent with
            the convention used for curve interpolation elsewhere in
            this package (see
            :meth:`credit_copula.market_data.DiscountCurve.zero_rate`).

        Notes
        -----
        Linear interpolation of base correlation is a market
        convention, not a theoretical requirement; some practitioners
        instead interpolate the implied compound correlation or use
        spline methods. Linear interpolation is adopted here as the
        simplest, most transparent default.
        """
        return float(np.interp(detachment, self.detachment_points, self.correlations))


def bootstrap_base_correlation(
    pricer: CDOPricer,
    detachment_points: np.ndarray,
    market_par_spreads: np.ndarray,
    maturity: float,
    correlation_bounds: tuple[float, float] = (1.0e-4, 0.999),
) -> BaseCorrelationCurve:
    """
    Bootstrap a base correlation curve from quoted base tranche par spreads.

    For each detachment point :math:`K_i`, solves for the flat
    correlation :math:`\\rho_{K_i}` such that the model par spread of
    the base tranche :math:`[0, K_i]`, computed with every obligor's
    one-factor copula correlation set to :math:`\\rho_{K_i}`, equals the
    quoted market par spread:

    .. math::
        s^{\\text{model}}_{[0, K_i]}(\\rho_{K_i}) - s^{\\text{market}}_{K_i} = 0

    Each equation is solved independently via Brent's method (see
    :func:`credit_copula.numerics.solve_root_brent`).

    Parameters
    ----------
    pricer : CDOPricer
        Pricing engine configured with the reference portfolio,
        discount curve, and numerical settings. The per-obligor
        correlations stored on the engine's portfolio are not used in
        this routine -- a flat correlation override is substituted for
        every detachment point -- so the portfolio's own `correlation`
        attributes are immaterial to the result and may be left at any
        placeholder value.
    detachment_points : np.ndarray
        Base tranche detachment points (currency units), strictly
        increasing.
    market_par_spreads : np.ndarray
        Quoted market par spreads for each base tranche
        :math:`[0, K_i]`, aligned with `detachment_points`, in decimal
        form.
    maturity : float
        Common maturity (years) of the quoted base tranches.
    correlation_bounds : tuple[float, float], default=(1e-4, 0.999)
        Search bracket for the root-finder. The lower bound is kept
        strictly above zero and the upper bound strictly below one to
        avoid the degenerate limits of the copula's default barrier
        construction.

    Returns
    -------
    BaseCorrelationCurve
        Bootstrapped base correlation curve.

    Raises
    ------
    ValueError
        If `detachment_points` is not strictly increasing, if the
        input array shapes are inconsistent, or if no root is found
        within `correlation_bounds` for some detachment point (which
        signals a market quote inconsistent with the one-factor
        Gaussian copula model at any feasible correlation).

    Notes
    -----
    The par spread of a base tranche :math:`[0, K]` is, for fixed
    market and portfolio inputs, generically monotonically decreasing
    in the flat correlation :math:`\\rho`: increasing correlation
    redistributes probability mass from the central region of the loss
    distribution toward both tails, increasing the probability of
    very small losses (which leaves the most subordinated, equity-like
    base tranche unaffected) and of very large losses (which the base
    tranche, attached at zero, fully absorbs up to its detachment
    point); the net effect for a tranche anchored at zero subordination
    is a reduction in expected tranche loss and hence in par spread.
    This monotonicity is what guarantees Brent's method finds a unique
    root within the search bracket for well-behaved market quotes.
    """
    detachment_points = np.asarray(detachment_points, dtype=np.float64)
    market_par_spreads = np.asarray(market_par_spreads, dtype=np.float64)
    if detachment_points.shape != market_par_spreads.shape:
        raise ValueError("detachment_points and market_par_spreads must have the same shape")
    if np.any(np.diff(detachment_points) <= 0.0):
        raise ValueError("detachment_points must be strictly increasing")

    n_obligors = pricer.portfolio.n_obligors
    bootstrapped_correlations = np.zeros_like(detachment_points)

    for i, (detachment, market_spread) in enumerate(
        zip(detachment_points, market_par_spreads)
    ):
        base_tranche = Tranche(attachment=0.0, detachment=float(detachment))

        def _spread_error(rho: float) -> float:
            flat_correlations = np.full(n_obligors, rho)
            result = pricer.price_tranche(base_tranche, maturity, correlations=flat_correlations)
            return result.par_spread - market_spread

        bootstrapped_correlations[i] = numerics.solve_root_brent(
            _spread_error, correlation_bounds[0], correlation_bounds[1]
        )

    return BaseCorrelationCurve(
        detachment_points=detachment_points, correlations=bootstrapped_correlations
    )


def bootstrap_base_correlation_from_standard_tranches(
    pricer: CDOPricer,
    tranche_pillars: Sequence[tuple[float, float]],
    market_par_spreads: Sequence[float],
    maturity: float,
    correlation_bounds: tuple[float, float] = (1.0e-4, 0.999),
) -> tuple[BaseCorrelationCurve, BaseCorrelationDiagnostics]:
    """
    Bootstrap base correlation from quoted *standard* (non-base) tranche spreads.

    This implements the sequential base-tranche stripping algorithm used
    in practice (McGinty et al., 2004; O'Kane, 2008, Ch. 15) to convert a
    contiguous structure of standard tranche quotes
    :math:`[A_1, D_1], [A_2, D_2], \\ldots` with :math:`A_1 = 0` and
    :math:`A_i = D_{i-1}` into a base correlation curve. Unlike
    :func:`bootstrap_base_correlation`, which requires market quotes for
    base tranches :math:`[0, K]` directly, this function accepts the
    market-standard quoting convention in which only the most
    subordinated tranche is itself a base tranche; every other tranche's
    quote must be converted into an *incremental* base tranche
    contribution before a correlation can be implied.

    The base tranche differencing identity

    .. math::
        \\text{Tranche}[A_i, D_i] = \\text{BaseTranche}[0, D_i]
            - \\text{BaseTranche}[0, A_i]

    applies leg-by-leg, so for pillar `i` the protection and premium legs
    satisfy

    .. math::
        V_{\\text{protection}}^{[A_i, D_i]}
            = V_{\\text{protection}}^{[0, D_i]}(\\rho_{D_i})
              - V_{\\text{protection}}^{[0, A_i]}(\\rho_{A_i})

    .. math::
        \\text{Annuity}^{[A_i, D_i]}
            = \\text{Annuity}^{[0, D_i]}(\\rho_{D_i})
              - \\text{Annuity}^{[0, A_i]}(\\rho_{A_i})

    Since :math:`\\rho_{A_i}` was already solved at the previous pillar
    (or is vacuously absent for the first, equity-attached pillar where
    :math:`A_1 = 0`), the only unknown is :math:`\\rho_{D_i}`, found by
    solving

    .. math::
        \\big[V_{\\text{protection}}^{[0, D_i]}(\\rho_{D_i}) - V_{\\text{protection}}^{[0, A_i]}\\big]
        - s_i^{\\text{market}} \\big[\\text{Annuity}^{[0, D_i]}(\\rho_{D_i}) - \\text{Annuity}^{[0, A_i]}\\big]
        = 0

    for :math:`\\rho_{D_i}`, where :math:`s_i^{\\text{market}}` is the
    quoted par spread of tranche `i`. Solving pillars in increasing order
    of detachment point makes each equation single-unknown.

    Parameters
    ----------
    pricer : CDOPricer
        Pricing engine; per-obligor correlations on the portfolio are
        overridden pillar-by-pillar and are otherwise immaterial.
    tranche_pillars : Sequence[tuple[float, float]]
        Contiguous standard tranche attachment/detachment pairs
        (currency units), sorted by increasing detachment, with the
        first attachment equal to zero and each subsequent attachment
        equal to the previous detachment.
    market_par_spreads : Sequence[float]
        Quoted market par spreads for each standard tranche, aligned
        with `tranche_pillars`, in decimal form.
    maturity : float
        Common maturity (years) of the quoted tranches.
    correlation_bounds : tuple[float, float], default=(1e-4, 0.999)
        Search bracket for the root-finder at each pillar.

    Returns
    -------
    BaseCorrelationCurve
        Bootstrapped base correlation curve, one point per detachment
        pillar.
    BaseCorrelationDiagnostics
        Per-pillar convergence diagnostics (iteration count, convergence
        flag, residual, function evaluation count).

    Raises
    ------
    ValueError
        If `tranche_pillars` is not contiguous and increasing starting
        at zero attachment, if input shapes are inconsistent, or if no
        root is found within `correlation_bounds` at some pillar.

    Notes
    -----
    Each pillar requires one extra `CDOPricer.price_tranche` evaluation
    per root-finder iteration relative to
    :func:`bootstrap_base_correlation`, since the trial base tranche
    :math:`[0, D_i]` must be repriced at every trial correlation while
    the prior pillar's leg present values are held fixed; computational
    cost therefore scales identically with portfolio size and loss
    discretization granularity (see :class:`credit_copula.pricer.CDOPricer`)
    but linearly with the number of pillars processed sequentially.
    """
    tranche_pillars = list(tranche_pillars)
    market_par_spreads = np.asarray(market_par_spreads, dtype=np.float64)
    if len(tranche_pillars) != market_par_spreads.size:
        raise ValueError("tranche_pillars and market_par_spreads must have the same length")
    if tranche_pillars[0][0] != 0.0:
        raise ValueError("the first tranche pillar must be attached at zero (the equity tranche)")
    for previous, current in zip(tranche_pillars[:-1], tranche_pillars[1:]):
        if current[0] != previous[1]:
            raise ValueError(
                "tranche_pillars must be contiguous: each attachment point must equal the "
                "previous detachment point"
            )

    n_obligors = pricer.portfolio.n_obligors
    n_pillars = len(tranche_pillars)
    detachment_points = np.array([detachment for _, detachment in tranche_pillars], dtype=np.float64)
    correlations = np.zeros(n_pillars, dtype=np.float64)
    iterations = np.zeros(n_pillars, dtype=np.int64)
    converged = np.zeros(n_pillars, dtype=bool)
    residuals = np.zeros(n_pillars, dtype=np.float64)
    function_calls = np.zeros(n_pillars, dtype=np.int64)

    prior_protection_pv = 0.0
    prior_annuity = 0.0

    for i, ((_, detachment), market_spread) in enumerate(zip(tranche_pillars, market_par_spreads)):
        base_tranche = Tranche(attachment=0.0, detachment=float(detachment))

        def _incremental_spread_error(rho: float) -> float:
            flat_correlations = np.full(n_obligors, rho)
            result = pricer.price_tranche(base_tranche, maturity, correlations=flat_correlations)
            incremental_protection = result.protection_leg_pv - prior_protection_pv
            incremental_annuity = result.risky_annuity - prior_annuity
            return incremental_protection - market_spread * incremental_annuity

        diagnostics = numerics.solve_root_brent(
            _incremental_spread_error,
            correlation_bounds[0],
            correlation_bounds[1],
            full_output=True,
        )
        correlations[i] = diagnostics.root
        iterations[i] = diagnostics.iterations
        converged[i] = diagnostics.converged
        residuals[i] = diagnostics.residual
        function_calls[i] = diagnostics.function_calls

        final_result = pricer.price_tranche(
            base_tranche, maturity, correlations=np.full(n_obligors, diagnostics.root)
        )
        prior_protection_pv = final_result.protection_leg_pv
        prior_annuity = final_result.risky_annuity

    curve = BaseCorrelationCurve(detachment_points=detachment_points, correlations=correlations)
    pillar_diagnostics = BaseCorrelationDiagnostics(
        detachment_points=detachment_points,
        iterations=iterations,
        converged=converged,
        residual=residuals,
        function_calls=function_calls,
    )
    return curve, pillar_diagnostics
