"""
Tranche specification, expected tranche loss, and leg valuation.

This module implements the cash-flow valuation of a single synthetic
CDO / credit index tranche given a model-implied expected tranche loss
(ETL) term structure, following standard risk-neutral valuation:

.. math::
    V_{\\text{protection}} = \\mathbb{E}^{\\mathbb{Q}}\\left[
        \\int_0^T DF(t)\\, d\\,\\text{ETL}(t)
    \\right]
    \\qquad
    V_{\\text{premium}} = s \\cdot \\mathbb{E}^{\\mathbb{Q}}\\left[
        \\sum_i DF(t_i)\\, \\Delta_i\\, \\big(N - \\text{ETL}(t_i)\\big)
    \\right]

where :math:`N` is the tranche notional, `s` the contractual spread,
:math:`\\Delta_i` the accrual factor for period `i`, and `ETL(t)` the
expected loss absorbed by the tranche by time `t` under the
risk-neutral measure :math:`\\mathbb{Q}`.

A tranche's payoff is the standard call-spread structure on portfolio
loss:

.. math::
    \\text{TrancheLoss}(L, A, D) = \\min\\big(\\max(L - A, 0),\\, D - A\\big)
        = (L - A)^+ - (L - D)^+

where `A` and `D` are the tranche attachment and detachment points
(currency units) and `L` is cumulative portfolio loss.

Notes
-----
Accrued premium on default is included in the premium leg via the
standard market midpoint approximation: see
:func:`premium_leg_annuity` for the formula and its derivation.

"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from credit_copula.conventions import (
    DayCountConvention,
    add_months,
    build_cds_premium_schedule,
    year_fraction,
)
from credit_copula.market_data import DiscountCurve

__all__ = [
    "Tranche",
    "tranche_loss_payoff",
    "expected_tranche_loss",
    "generate_payment_schedule",
    "protection_leg_pv",
    "premium_leg_annuity",
    "par_spread",
    "mark_to_market",
    "fair_upfront",
]


@dataclass(frozen=True)
class Tranche:
    """
    A synthetic CDO / credit index tranche specification.

    Parameters
    ----------
    attachment : float
        Attachment point, in currency units of portfolio loss
        (the subordination level below which the tranche is fully
        protected from loss). Must satisfy
        ``0 <= attachment < detachment``.
    detachment : float
        Detachment point, in currency units of portfolio loss (the
        point beyond which the tranche notional is fully exhausted).

    Raises
    ------
    ValueError
        If ``attachment < 0`` or ``attachment >= detachment``.
    """

    attachment: float
    detachment: float

    def __post_init__(self) -> None:
        if self.attachment < 0.0:
            raise ValueError("attachment must be non-negative")
        if self.attachment >= self.detachment:
            raise ValueError("attachment must be strictly less than detachment")

    @property
    def notional(self) -> float:
        """Tranche notional, i.e. the maximum loss it can absorb."""
        return self.detachment - self.attachment


def tranche_loss_payoff(
    portfolio_loss: np.ndarray | float, attachment: float, detachment: float
) -> np.ndarray:
    """
    Evaluate the tranche loss payoff for given portfolio loss realization(s).

    .. math::
        \\text{TrancheLoss}(L) = \\min\\big(\\max(L - A, 0),\\, D - A\\big)

    Parameters
    ----------
    portfolio_loss : array_like or float
        Cumulative portfolio loss realization(s) (currency units).
    attachment : float
        Tranche attachment point `A`.
    detachment : float
        Tranche detachment point `D`.

    Returns
    -------
    np.ndarray
        Tranche loss, bounded in :math:`[0, D - A]`, same shape as
        `portfolio_loss`.
    """
    loss = np.asarray(portfolio_loss, dtype=np.float64)
    return np.clip(loss - attachment, 0.0, detachment - attachment)


def expected_tranche_loss(
    loss_grid: np.ndarray, probabilities: np.ndarray, attachment: float, detachment: float
) -> float:
    """
    Expected tranche loss from a discretized portfolio loss distribution.

    .. math::
        \\text{ETL} = \\sum_k P(L = \\ell_k)\\, \\text{TrancheLoss}(\\ell_k, A, D)

    Parameters
    ----------
    loss_grid : np.ndarray
        Portfolio loss values (currency units) corresponding to each
        probability mass point, shape ``(n_buckets,)``.
    probabilities : np.ndarray
        Probability mass function over `loss_grid`, shape
        ``(n_buckets,)``, should sum to approximately 1.0.
    attachment : float
        Tranche attachment point.
    detachment : float
        Tranche detachment point.

    Returns
    -------
    float
        Expected tranche loss in currency units, bounded in
        :math:`[0, D - A]`.
    """
    payoff = tranche_loss_payoff(loss_grid, attachment, detachment)
    return float(np.sum(probabilities * payoff))


def generate_payment_schedule(
    maturity: float,
    frequency: int = 4,
    valuation_date: date | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a premium payment schedule.

    Parameters
    ----------
    maturity : float
        Tranche maturity in years from `valuation_date` (or from an
        unspecified valuation date, if `valuation_date` is `None`).
        Must be strictly positive.
    frequency : int, default=4
        Number of payments per annum (4 = quarterly, the standard
        market convention for index tranches). Used directly as the
        schedule's roll frequency in months (``12 / frequency``) when
        `valuation_date` is supplied.
    valuation_date : date, optional
        If supplied, the schedule is built from real calendar dates via
        :func:`credit_copula.conventions.build_cds_premium_schedule`
        (ACT/360 accrual, Modified Following business-day-adjusted
        payment dates, standard quarterly roll pattern), with the
        maturity date computed as `maturity` whole years after
        `valuation_date`. Returned times are year fractions from
        `valuation_date` under ACT/365F, the convention used throughout
        this package for converting calendar dates into the time
        variable consumed by :class:`credit_copula.market_data.DiscountCurve`
        and :class:`credit_copula.market_data.CreditCurve`.

        If `None` (the default), an evenly-spaced schedule with constant
        accrual fraction ``1 / frequency`` is returned instead -- a
        calendar-free approximation suited to fast sensitivity analysis
        and scenarios where no specific valuation date is meaningful
        (e.g. parameter sweeps), at the cost of not reflecting actual
        day-count and business-day effects; see `NOTES.md` for this
        module's scope decision on the two schedule construction paths.

    Returns
    -------
    payment_times : np.ndarray
        Payment times in years from the valuation date.
    accrual_fractions : np.ndarray
        Year fraction accrued for each payment period.

    Raises
    ------
    ValueError
        If `maturity` or `frequency` is non-positive.
    """
    if maturity <= 0.0:
        raise ValueError("maturity must be strictly positive")
    if frequency <= 0:
        raise ValueError("frequency must be a positive integer")

    if valuation_date is None:
        n_periods = max(int(round(maturity * frequency)), 1)
        payment_times = np.linspace(maturity / n_periods, maturity, n_periods)
        accrual_fractions = np.full(n_periods, 1.0 / frequency)
        return payment_times, accrual_fractions

    maturity_date = add_months(valuation_date, round(maturity * 12))
    schedule = build_cds_premium_schedule(
        effective_date=valuation_date,
        maturity_date=maturity_date,
        frequency_months=round(12 / frequency),
    )
    payment_times = np.array(
        [year_fraction(valuation_date, period.payment_date, DayCountConvention.ACT_365F) for period in schedule]
    )
    accrual_fractions = np.array([period.year_fraction for period in schedule])
    return payment_times, accrual_fractions


def protection_leg_pv(
    expected_tranche_losses: np.ndarray,
    integration_times: np.ndarray,
    discount_curve: DiscountCurve,
) -> float:
    """
    Present value of the protection leg.

    The protection leg pays the increase in expected tranche loss as it
    is realized:

    .. math::
        V_{\\text{protection}} = \\sum_{j=1}^{M} DF(t_j)\\,
            \\big[\\text{ETL}(t_j) - \\text{ETL}(t_{j-1})\\big]

    where the time grid :math:`\\{t_0=0, t_1, \\ldots, t_M\\}` should be
    fine enough to accurately capture the curvature of the ETL term
    structure (a monthly or finer grid is standard for index tranches
    with maturities up to 10 years).

    Parameters
    ----------
    expected_tranche_losses : np.ndarray
        Expected tranche loss evaluated at `integration_times`, shape
        ``(M+1,)``, with ``expected_tranche_losses[0]`` corresponding to
        ``t=0`` (and therefore equal to 0.0).
    integration_times : np.ndarray
        Strictly increasing time grid in years, shape ``(M+1,)``, with
        ``integration_times[0] == 0.0``.
    discount_curve : DiscountCurve
        Risk-free discount curve.

    Returns
    -------
    float
        Protection leg present value, in currency units.

    Raises
    ------
    ValueError
        If the input array shapes are inconsistent or
        `integration_times` does not start at zero.
    """
    if expected_tranche_losses.shape != integration_times.shape:
        raise ValueError("expected_tranche_losses and integration_times must align")
    if integration_times[0] != 0.0:
        raise ValueError("integration_times must start at t=0")

    discount_factors = discount_curve.discount_factor(integration_times[1:])
    loss_increments = np.diff(expected_tranche_losses)
    return float(np.sum(discount_factors * loss_increments))


def premium_leg_annuity(
    tranche_notional: float,
    expected_tranche_losses_at_payments: np.ndarray,
    payment_times: np.ndarray,
    accrual_fractions: np.ndarray,
    discount_curve: DiscountCurve,
    expected_tranche_losses_at_period_start: np.ndarray | None = None,
) -> float:
    """
    Risky annuity (premium leg present value per unit of contractual spread).

    .. math::
        \\text{Annuity} = \\underbrace{\\sum_{i=1}^{n} DF(t_i)\\, \\Delta_i\\,
            \\big(N - \\text{ETL}(t_i)\\big)}_{\\text{coupon on surviving notional}}
            \\; + \\;
            \\underbrace{\\sum_{i=1}^{n} DF(t_i)\\, \\tfrac{1}{2}\\,\\Delta_i\\,
            \\big(\\text{ETL}(t_i) - \\text{ETL}(t_{i-1})\\big)}_{\\text{accrued premium on default}}

    where :math:`N - \\text{ETL}(t_i)` is the expected surviving tranche
    notional at the i-th payment date, and the second term is the
    expected accrued premium owed on the portion of tranche notional
    that defaults during period `i`.

    **Accrued premium on default.** When a default reduces tranche
    notional partway through an accrual period, the protection buyer
    owes accrued premium from the period's start up to the default
    date on the notional that has just defaulted. This is approximated
    by assuming, conditional on a notional-reducing event occurring
    within period `i`, that it occurs on average at the period
    midpoint -- contributing half the period's accrual fraction,
    weighted by the period's expected tranche loss increment
    :math:`\\text{ETL}(t_i) - \\text{ETL}(t_{i-1})` (the expected
    fraction of tranche notional that defaults during the period).
    This midpoint approximation is the standard treatment of accrued
    interest on default in the CDS market (the same approximation
    underlies the ISDA CDS Standard Model's accrued-on-default term for
    single-name contracts); it is exact only if defaults occur exactly
    at period midpoints in expectation, and introduces a bias that
    shrinks as accrual periods shorten. Omitting `expected_tranche_losses_at_period_start`
    (passing `None`) excludes this term entirely, reproducing the
    coupon-only annuity.

    Parameters
    ----------
    tranche_notional : float
        Tranche notional `N = D - A`.
    expected_tranche_losses_at_payments : np.ndarray
        Expected tranche loss evaluated at each payment (period end)
        date, shape ``(n,)``.
    payment_times : np.ndarray
        Payment times in years, shape ``(n,)``.
    accrual_fractions : np.ndarray
        Accrual year fractions for each payment period, shape ``(n,)``.
    discount_curve : DiscountCurve
        Risk-free discount curve.
    expected_tranche_losses_at_period_start : np.ndarray, optional
        Expected tranche loss evaluated at each accrual period's start
        date, shape ``(n,)``, with
        ``expected_tranche_losses_at_period_start[0]`` corresponding to
        the first period's start (0.0 if the schedule begins at
        valuation). When supplied, the accrued-premium-on-default term
        above is included; when `None`, it is omitted.

    Returns
    -------
    float
        Risky annuity in currency units (premium leg PV per unit of
        spread), including accrued premium on default when
        `expected_tranche_losses_at_period_start` is supplied.
    """
    discount_factors = discount_curve.discount_factor(payment_times)
    surviving_notional = tranche_notional - expected_tranche_losses_at_payments
    coupon_pv = float(np.sum(discount_factors * accrual_fractions * surviving_notional))

    if expected_tranche_losses_at_period_start is None:
        return coupon_pv

    loss_increments = expected_tranche_losses_at_payments - expected_tranche_losses_at_period_start
    accrued_on_default_pv = float(
        np.sum(discount_factors * 0.5 * accrual_fractions * loss_increments)
    )
    return coupon_pv + accrued_on_default_pv


def par_spread(protection_pv: float, annuity: float) -> float:
    """
    Par (break-even) spread that equates protection and premium leg PVs.

    .. math::
        s^{\\*} = \\frac{V_{\\text{protection}}}{\\text{Annuity}}

    Parameters
    ----------
    protection_pv : float
        Protection leg present value.
    annuity : float
        Risky annuity (premium leg PV per unit spread).

    Returns
    -------
    float
        Par spread, in decimal form (e.g. 0.01 = 100 bps).

    Raises
    ------
    ValueError
        If `annuity` is not strictly positive.
    """
    if annuity <= 0.0:
        raise ValueError("annuity must be strictly positive to imply a par spread")
    return protection_pv / annuity


def mark_to_market(
    contractual_spread: float,
    protection_pv: float,
    annuity: float,
    upfront_payment: float = 0.0,
) -> float:
    """
    Mark-to-market value of a protection buyer's position.

    .. math::
        V = V_{\\text{protection}} - s \\cdot \\text{Annuity} - U

    where `U` is any upfront payment made by the protection buyer at
    inception (relevant for equity tranches such as the 0-3% tranche,
    which are conventionally quoted with an upfront fee plus a fixed
    running spread, typically 500 bps).

    Parameters
    ----------
    contractual_spread : float
        Fixed running spread paid by the protection buyer, in decimal
        form.
    protection_pv : float
        Protection leg present value.
    annuity : float
        Risky annuity (premium leg PV per unit spread).
    upfront_payment : float, default=0.0
        Upfront amount paid by the protection buyer at trade inception,
        in currency units.

    Returns
    -------
    float
        Mark-to-market value to the protection buyer, in currency
        units. Positive values indicate the position has positive
        value to the protection buyer.
    """
    return protection_pv - contractual_spread * annuity - upfront_payment


def fair_upfront(
    protection_pv: float, annuity: float, contractual_spread: float, tranche_notional: float
) -> float:
    """
    Fair upfront payment, as a fraction of tranche notional, given a fixed contractual spread.

    Index tranche equity pieces (e.g. the CDX.NA.IG 0-3% tranche) trade
    with a fixed, standardized running spread set well below the level
    that would be required to clear the trade on a running-spread-only
    basis; the residual present value is exchanged as an upfront
    payment at inception. Setting :func:`mark_to_market` to zero and
    solving for the upfront payment gives

    .. math::
        U^{*} = \\frac{V_{\\text{protection}} - s_{\\text{fixed}} \\cdot \\text{Annuity}}{N}

    where `N` is the tranche notional; this is the upfront fraction
    quoted in the market for upfront-convention tranches.

    Parameters
    ----------
    protection_pv : float
        Protection leg present value.
    annuity : float
        Risky annuity (premium leg PV per unit spread).
    contractual_spread : float
        Fixed running spread paid by the protection buyer, in decimal
        form (e.g. 0.05 for a standardized 500 bps coupon).
    tranche_notional : float
        Tranche notional, in currency units.

    Returns
    -------
    float
        Fair upfront payment as a fraction of tranche notional.
        Positive values denote a payment from protection buyer to
        protection seller at inception, the market convention for
        index tranche equity pieces.

    Raises
    ------
    ValueError
        If `tranche_notional` is not strictly positive.
    """
    if tranche_notional <= 0.0:
        raise ValueError("tranche_notional must be strictly positive")
    return (protection_pv - contractual_spread * annuity) / tranche_notional
