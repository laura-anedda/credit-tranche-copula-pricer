"""
Market data structures: discount curves and single-name credit curves.

This module provides immutable, vectorized representations of the term
structures required to price credit derivatives under risk-neutral
valuation: the risk-free discount curve and the single-name survival
(credit) curve. It also implements the standard piecewise-constant
hazard rate bootstrap used to translate market CDS par spread quotes
into a survival probability curve.

Notes
-----
All probabilities of default referred to throughout this package are
risk-neutral default probabilities implied by traded CDS/CDX spreads,
not real-world (physical) default probabilities. Discounting is
performed under the risk-neutral measure using the risk-free curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

__all__ = ["DiscountCurve", "CreditCurve", "bootstrap_hazard_rates"]


@dataclass(frozen=True)
class DiscountCurve:
    """
    Continuously-compounded zero-coupon discount curve.

    The discount factor is defined as

    .. math::
        DF(t) = \\exp\\left(-\\int_0^t f(s)\\, ds\\right)

    where the instantaneous forward rate :math:`f(s)` is held piecewise
    constant between consecutive pillar dates. Equivalently,
    :math:`\\ln DF(t)` is linearly interpolated between pillars (and
    between the origin and the first pillar, using the first pillar's
    own zero rate as the flat short-end forward). This **flat-forward
    interpolation** is the standard convention for production discount
    curves: it guarantees strictly positive, monotonically decreasing
    discount factors under non-negative rates and avoids the
    oscillating, occasionally negative implied forward rates that
    naive linear interpolation of zero rates can produce between
    widely-spaced pillars.

    Parameters
    ----------
    tenors : np.ndarray
        Strictly increasing pillar times in years, with ``tenors[0] > 0``.
    zero_rates : np.ndarray
        Continuously-compounded zero rates corresponding to ``tenors``,
        expressed as decimals (e.g. 0.03 for 3%).

    Raises
    ------
    ValueError
        If ``tenors`` is not strictly increasing, is empty, or the two
        input arrays differ in length.
    """

    tenors: np.ndarray
    zero_rates: np.ndarray
    _log_discount_factors: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        tenors = np.asarray(self.tenors, dtype=np.float64)
        zero_rates = np.asarray(self.zero_rates, dtype=np.float64)
        if tenors.shape != zero_rates.shape:
            raise ValueError("tenors and zero_rates must have the same shape")
        if tenors.size == 0:
            raise ValueError("tenors must be non-empty")
        if np.any(np.diff(tenors) <= 0.0) or tenors[0] <= 0.0:
            raise ValueError("tenors must be strictly increasing and positive")
        object.__setattr__(self, "tenors", tenors)
        object.__setattr__(self, "zero_rates", zero_rates)
        object.__setattr__(self, "_log_discount_factors", -zero_rates * tenors)

    def discount_factor(self, t: np.ndarray | float) -> np.ndarray:
        """
        Compute the discount factor under flat-forward interpolation of the curve.

        Parameters
        ----------
        t : array_like or float
            Time(s) in years from the valuation date. Negative values
            are not supported.

        Returns
        -------
        np.ndarray
            Discount factor(s), same shape as ``t``.

        Raises
        ------
        ValueError
            If any element of `t` is negative.

        Notes
        -----
        For ``t`` beyond the last pillar, the discount factor is
        extrapolated by holding the terminal pillar's zero rate flat as
        the forward rate beyond that point (flat-forward
        extrapolation), consistent with the interpolation scheme used
        between pillars.
        """
        t_arr = np.asarray(t, dtype=np.float64)
        if np.any(t_arr < 0.0):
            raise ValueError("time to discount must be non-negative")

        pillar_times = np.concatenate(([0.0], self.tenors))
        pillar_log_df = np.concatenate(([0.0], self._log_discount_factors))
        log_df = np.interp(t_arr, pillar_times, pillar_log_df)

        beyond_last_pillar = t_arr > self.tenors[-1]
        if np.any(beyond_last_pillar):
            terminal_forward_rate = self.zero_rates[-1]
            extrapolated_log_df = self._log_discount_factors[-1] - terminal_forward_rate * (
                t_arr - self.tenors[-1]
            )
            log_df = np.where(beyond_last_pillar, extrapolated_log_df, log_df)

        return np.exp(log_df)

    def zero_rate(self, t: np.ndarray | float) -> np.ndarray:
        """
        Continuously-compounded zero rate implied by the flat-forward discount curve.

        .. math::
            r(t) = -\\frac{\\ln DF(t)}{t}

        Parameters
        ----------
        t : array_like or float
            Time(s) in years, measured from the valuation date.

        Returns
        -------
        np.ndarray
            Implied zero rate(s), same shape as ``t``. The limit as
            ``t \\to 0`` (returned exactly at ``t=0``) equals the first
            pillar's own zero rate, consistent with the flat short-end
            forward used by :meth:`discount_factor`.
        """
        t_arr = np.asarray(t, dtype=np.float64)
        # The t=0 division is singular (0/0); substitute a placeholder
        # denominator there and override the result with the t -> 0 limit
        # afterward, rather than evaluating the indeterminate form.
        safe_t = np.where(t_arr == 0.0, 1.0, t_arr)
        implied_zero_rate = -np.log(self.discount_factor(safe_t)) / safe_t
        return np.where(t_arr == 0.0, self.zero_rates[0], implied_zero_rate)


@dataclass(frozen=True)
class CreditCurve:
    """
    Single-name (or index-average) survival probability term structure.

    The curve is parametrized by a piecewise-constant forward hazard
    (default intensity) function :math:`\\lambda(t)`, consistent with
    the standard CDS pricing convention. The survival probability is

    .. math::
        Q(t) = \\exp\\left(-\\int_0^t \\lambda(s)\\, ds\\right)
             = \\exp\\left(-\\sum_{i: t_i \\le t} \\lambda_i \\Delta_i
                           - \\lambda_k (t - t_{k-1})\\right)

    where :math:`\\lambda_i` is the hazard rate applicable on the
    interval :math:`(t_{i-1}, t_i]` and `k` is the index of the
    interval containing `t`.

    Parameters
    ----------
    tenors : np.ndarray
        Strictly increasing pillar times in years, with ``tenors[0] > 0``.
    hazard_rates : np.ndarray
        Piecewise-constant forward hazard rates applicable on
        ``(tenors[i-1], tenors[i]]`` (with ``tenors[-1] = 0``),
        expressed as decimals per annum.
    recovery_rate : float
        Constant fractional recovery rate assumed upon default,
        in :math:`[0, 1)`. Loss given default (LGD) is
        ``1 - recovery_rate``.

    Raises
    ------
    ValueError
        If array shapes are inconsistent, hazard rates are negative,
        or `recovery_rate` lies outside :math:`[0, 1)`.
    """

    tenors: np.ndarray
    hazard_rates: np.ndarray
    recovery_rate: float
    _cum_hazard: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        tenors = np.asarray(self.tenors, dtype=np.float64)
        hazard_rates = np.asarray(self.hazard_rates, dtype=np.float64)
        if tenors.shape != hazard_rates.shape:
            raise ValueError("tenors and hazard_rates must have the same shape")
        if tenors.size == 0:
            raise ValueError("tenors must be non-empty")
        if np.any(np.diff(tenors) <= 0.0) or tenors[0] <= 0.0:
            raise ValueError("tenors must be strictly increasing and positive")
        if np.any(hazard_rates < 0.0):
            raise ValueError("hazard rates must be non-negative")
        if not (0.0 <= self.recovery_rate < 1.0):
            raise ValueError("recovery_rate must lie in [0, 1)")

        object.__setattr__(self, "tenors", tenors)
        object.__setattr__(self, "hazard_rates", hazard_rates)

        # Pre-compute cumulative hazard at each pillar for O(log n) lookup
        # of survival probability via interpolation of the cumulative
        # hazard function (piecewise-linear, by construction continuous
        # and monotonically non-decreasing).
        widths = np.diff(np.concatenate(([0.0], tenors)))
        cum_hazard = np.cumsum(hazard_rates * widths)
        object.__setattr__(self, "_cum_hazard", cum_hazard)

    def cumulative_hazard(self, t: np.ndarray | float) -> np.ndarray:
        """
        Evaluate the cumulative hazard function :math:`\\Lambda(t) = \\int_0^t \\lambda(s)\\,ds`.

        Parameters
        ----------
        t : array_like or float
            Time(s) in years, non-negative.

        Returns
        -------
        np.ndarray
            Cumulative hazard, same shape as `t`. Flat extrapolation
            (constant final hazard rate) is applied beyond the last
            pillar.
        """
        t_arr = np.asarray(t, dtype=np.float64)
        if np.any(t_arr < 0.0):
            raise ValueError("time must be non-negative")
        pillars = np.concatenate(([0.0], self.tenors))
        cum = np.concatenate(([0.0], self._cum_hazard))
        result = np.interp(t_arr, pillars, cum)
        # Extrapolate beyond the last pillar using the terminal hazard rate.
        beyond = t_arr > self.tenors[-1]
        if np.any(beyond):
            result = np.where(
                beyond,
                self._cum_hazard[-1] + self.hazard_rates[-1] * (t_arr - self.tenors[-1]),
                result,
            )
        return result

    def survival_probability(self, t: np.ndarray | float) -> np.ndarray:
        """
        Compute the risk-neutral survival probability :math:`Q(t) = e^{-\\Lambda(t)}`.

        Parameters
        ----------
        t : array_like or float
            Time(s) in years, non-negative.

        Returns
        -------
        np.ndarray
            Survival probability in :math:`(0, 1]`, same shape as `t`.
        """
        return np.exp(-self.cumulative_hazard(t))

    def default_probability(self, t: np.ndarray | float) -> np.ndarray:
        """
        Compute the cumulative risk-neutral default probability :math:`1 - Q(t)`.

        Parameters
        ----------
        t : array_like or float
            Time(s) in years, non-negative.

        Returns
        -------
        np.ndarray
            Default probability in :math:`[0, 1)`, same shape as `t`.
        """
        return 1.0 - self.survival_probability(t)


def bootstrap_hazard_rates(
    cds_tenors: np.ndarray,
    cds_spreads: np.ndarray,
    recovery_rate: float,
    discount_curve: DiscountCurve,
    payment_frequency: int = 4,
) -> CreditCurve:
    """
    Bootstrap a piecewise-constant hazard rate curve from CDS par spreads.

    The bootstrap solves, sequentially for each maturity pillar, the
    hazard rate :math:`\\lambda_i` on the interval :math:`(t_{i-1}, t_i]`
    that equates the CDS premium leg and protection leg present values,
    i.e. sets the par CDS spread for that maturity to its quoted market
    value. The CDS valuation equations used are the standard semi-annual
    (or quarterly) risky-annuity approximation under flat hazard-rate
    interpolation:

    .. math::
        \\text{Protection Leg PV} = (1 - R) \\int_0^{T} DF(t)\\, dQ(t)
        \\;\\approx\\; (1-R) \\sum_{j} DF(t_j)\\,[Q(t_{j-1}) - Q(t_j)]

    .. math::
        \\text{Premium Leg PV} = s \\sum_{j} DF(t_j)\\, \\Delta_j\\, Q(t_j)
        \\; + \\; s \\sum_{j} DF(t_j)\\, \\tfrac{1}{2}\\,\\Delta_j\\,
              \\big[Q(t_{j-1}) - Q(t_j)\\big]

    where the protection leg integral is discretized on a fine
    sub-grid (the accrual schedule subdivided by `payment_frequency`),
    and the second premium leg term is the accrued premium owed by the
    protection buyer on default, under the standard market midpoint
    approximation (default within period `j` assumed to occur, on
    average, at the period midpoint, contributing half of `Δ_j`); see
    :func:`credit_copula.tranche.premium_leg_annuity` for the
    tranche-level analogue of this same approximation.

    Parameters
    ----------
    cds_tenors : np.ndarray
        Strictly increasing CDS maturities in years (bootstrap pillars).
    cds_spreads : np.ndarray
        Market-quoted CDS par spreads (decimals, e.g. 0.01 for 100 bps),
        aligned with `cds_tenors`.
    recovery_rate : float
        Constant assumed recovery rate in :math:`[0, 1)`.
    discount_curve : DiscountCurve
        Risk-free discount curve used to compute present values.
    payment_frequency : int, default=4
        Number of premium payments per annum (4 = quarterly, the
        standard CDS market convention).

    Returns
    -------
    CreditCurve
        Bootstrapped survival curve with one piecewise-constant hazard
        rate per maturity pillar.

    Raises
    ------
    ValueError
        If input array shapes are inconsistent or `cds_tenors` is not
        strictly increasing.

    Notes
    -----
    Each pillar hazard rate is solved independently via Brent's method
    (`scipy.optimize.brentq`), holding all hazard rates on prior
    intervals fixed at their already-bootstrapped values. This is the
    standard sequential ("forward substitution") bootstrap algorithm
    used throughout CDS curve construction. Convergence is guaranteed
    within machine precision because the premium leg PV is strictly
    monotonically decreasing in the hazard rate (greater intensity
    reduces survival probability and hence risky annuity), while the
    protection leg PV is strictly increasing; the root is therefore
    unique and bracketed by economically sensible hazard rate bounds.

    References
    ----------
    .. [OKane2008] O'Kane, D. (2008). "Modelling Single-name and
       Multi-name Credit Derivatives." Wiley Finance, Ch. 4.
    """
    cds_tenors = np.asarray(cds_tenors, dtype=np.float64)
    cds_spreads = np.asarray(cds_spreads, dtype=np.float64)
    if cds_tenors.shape != cds_spreads.shape:
        raise ValueError("cds_tenors and cds_spreads must have the same shape")
    if np.any(np.diff(cds_tenors) <= 0.0) or cds_tenors[0] <= 0.0:
        raise ValueError("cds_tenors must be strictly increasing and positive")

    n_pillars = cds_tenors.size
    hazard_rates = np.zeros(n_pillars, dtype=np.float64)
    dt_accrual = 1.0 / payment_frequency

    def _survival_with_trial_rate(
        trial_rate: float, pillar_idx: int, eval_times: np.ndarray
    ) -> np.ndarray:
        """Survival probability at `eval_times` using already-bootstrapped
        hazard rates for earlier intervals and `trial_rate` for the
        current (pillar_idx) interval.

        Delegates to `CreditCurve.survival_probability`, which correctly
        handles evaluation times that fall before the most recently
        bootstrapped pillar via piecewise-linear interpolation of the
        cumulative hazard function; `eval_times` is restricted to
        `[0, cds_tenors[pillar_idx]]` by construction in `_pricing_error`,
        so no extrapolation beyond the trial pillar is required.
        """
        trial_pillars = np.concatenate((cds_tenors[:pillar_idx], [cds_tenors[pillar_idx]]))
        trial_rates = np.concatenate((hazard_rates[:pillar_idx], [trial_rate]))
        trial_curve = CreditCurve(
            tenors=trial_pillars, hazard_rates=trial_rates, recovery_rate=recovery_rate
        )
        return trial_curve.survival_probability(eval_times)

    def _pricing_error(trial_rate: float, pillar_idx: int) -> float:
        maturity = cds_tenors[pillar_idx]
        spread = cds_spreads[pillar_idx]

        n_periods = max(int(round(maturity * payment_frequency)), 1)
        payment_times = np.linspace(dt_accrual, maturity, n_periods)
        period_start_times = np.concatenate(([0.0], payment_times[:-1]))
        survival_at_payments = _survival_with_trial_rate(trial_rate, pillar_idx, payment_times)
        survival_at_period_start = _survival_with_trial_rate(trial_rate, pillar_idx, period_start_times)
        discount_at_payments = discount_curve.discount_factor(payment_times)

        coupon_pv = spread * np.sum(discount_at_payments * dt_accrual * survival_at_payments)
        default_probability_increment = survival_at_period_start - survival_at_payments
        accrued_on_default_pv = spread * np.sum(
            discount_at_payments * 0.5 * dt_accrual * default_probability_increment
        )
        premium_leg_pv = coupon_pv + accrued_on_default_pv

        # Protection leg: integrate over a fine sub-grid to approximate
        # the continuous-time default time integral.
        n_sub_steps = max(n_periods * 4, 4)
        grid = np.linspace(0.0, maturity, n_sub_steps + 1)
        survival_grid = _survival_with_trial_rate(trial_rate, pillar_idx, grid)
        discount_grid = discount_curve.discount_factor(grid)
        default_increment = survival_grid[:-1] - survival_grid[1:]
        protection_leg_pv = (1.0 - recovery_rate) * np.sum(
            discount_grid[1:] * default_increment
        )

        return protection_leg_pv - premium_leg_pv

    for i in range(n_pillars):
        # Hazard rate bounds chosen to comfortably bracket investment-
        # grade through distressed credit spread levels (0 to 500%
        # annualized intensity), with the root-finder failing loudly
        # if the bracket is insufficient for an unusual input.
        hazard_rates[i] = brentq(_pricing_error, 1.0e-8, 5.0, args=(i,), xtol=1.0e-12)

    return CreditCurve(
        tenors=cds_tenors.copy(),
        hazard_rates=hazard_rates,
        recovery_rate=recovery_rate,
    )
