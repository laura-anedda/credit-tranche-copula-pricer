"""
High-level orchestration of the one-factor Gaussian copula tranche pricer.

This module ties together market data (:mod:`credit_copula.market_data`),
the copula dependence model (:mod:`credit_copula.copula`), the portfolio
loss distribution machinery (:mod:`credit_copula.portfolio`), and tranche
cash-flow valuation (:mod:`credit_copula.tranche`) into a single,
stateless pricing engine, `CDOPricer`.

The engine evaluates expected tranche loss on a time grid spanning the
tranche maturity, then values the protection and premium legs from that
curve under risk-neutral valuation. All computation is deterministic
given the engine configuration and is free of side effects, making the
engine straightforward to unit test and to parallelize across tranches
or scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from credit_copula import tranche as tranche_mod
from credit_copula.market_data import DiscountCurve
from credit_copula.portfolio import (
    CreditPortfolio,
    conditional_loss_distribution,
    discretize_loss_given_default,
    loss_distribution,
)
from credit_copula.tranche import Tranche

__all__ = ["TranchePricingResult", "CDOPricer"]


@dataclass(frozen=True)
class TranchePricingResult:
    """
    Outcome of pricing a single tranche over its full term.

    Attributes
    ----------
    par_spread : float
        Break-even running spread, in decimal form.
    protection_leg_pv : float
        Present value of the protection leg, in currency units.
    risky_annuity : float
        Premium leg present value per unit of running spread (the
        risky PV01 multiplied by tranche notional).
    tranche_notional : float
        Tranche notional `D - A`, in currency units.
    expected_tranche_loss_curve : np.ndarray
        Expected tranche loss evaluated at `integration_times`.
    integration_times : np.ndarray
        Time grid (years from valuation date) underlying
        `expected_tranche_loss_curve`.
    """

    par_spread: float
    protection_leg_pv: float
    risky_annuity: float
    tranche_notional: float
    expected_tranche_loss_curve: np.ndarray
    integration_times: np.ndarray

    @property
    def risky_pv01(self) -> float:
        """Risky PV01 per unit notional (annuity normalized by tranche notional)."""
        return self.risky_annuity / self.tranche_notional


@dataclass(frozen=True)
class CDOPricer:
    """
    One-factor Gaussian copula pricing engine for synthetic CDO tranches.

    Parameters
    ----------
    portfolio : CreditPortfolio
        Reference portfolio of obligors.
    discount_curve : DiscountCurve
        Risk-free discount curve used for all present value
        calculations.
    loss_unit : float
        Discretization bucket size (currency units) for the
        Andersen-Sidenius-Basu recursive loss distribution. Smaller
        values reduce discretization bias at the cost of increased
        computation (the number of loss buckets scales as
        ``total portfolio LGD / loss_unit``).
    n_quadrature_points : int, default=32
        Number of Gauss-Hermite quadrature nodes used to integrate the
        conditional loss distribution over the systemic factor.
    payment_frequency : int, default=4
        Premium payments per annum (4 = quarterly).
    n_integration_steps_per_year : int, default=12
        Number of time steps per annum used to discretize the
        protection leg loss integral (12 = monthly, a standard
        resolution for capturing ETL curvature without excessive
        computational cost).

    Notes
    -----
    All loss distribution and tranche valuation calculations performed
    by this engine are pure functions of their inputs (see
    :mod:`credit_copula.portfolio` and :mod:`credit_copula.tranche`);
    `CDOPricer` itself holds no mutable state and produces identical
    output for identical input across repeated calls.
    """

    portfolio: CreditPortfolio
    discount_curve: DiscountCurve
    loss_unit: float
    n_quadrature_points: int = 32
    payment_frequency: int = 4
    n_integration_steps_per_year: int = 12
    _loss_grid: np.ndarray = field(init=False, repr=False)
    _n_buckets: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.loss_unit <= 0.0:
            raise ValueError("loss_unit must be strictly positive")
        total_lgd = float(np.sum(self.portfolio.loss_given_defaults()))
        n_buckets = int(np.ceil(total_lgd / self.loss_unit)) + 1
        loss_grid = np.arange(n_buckets, dtype=np.float64) * self.loss_unit
        object.__setattr__(self, "_n_buckets", n_buckets)
        object.__setattr__(self, "_loss_grid", loss_grid)

    @property
    def loss_grid(self) -> np.ndarray:
        """Discretized portfolio loss support, in currency units (``loss_unit`` multiples)."""
        return self._loss_grid

    @property
    def n_loss_buckets(self) -> int:
        """Number of loss buckets spanning the discretized portfolio loss support."""
        return self._n_buckets

    def conditional_loss_distribution(
        self, t: float, systemic_factor: float, correlations: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Portfolio loss distribution conditional on a systemic factor realization.

        Parameters
        ----------
        t : float
            Time horizon in years.
        systemic_factor : float
            Realization `z` of the systemic factor `Z`.
        correlations : np.ndarray, optional
            Per-obligor correlation override, see
            :meth:`portfolio_loss_distribution`.

        Returns
        -------
        np.ndarray
            Conditional probability mass function over `self.loss_grid`,
            via the Andersen-Sidenius-Basu recursion (see
            :func:`credit_copula.portfolio.conditional_loss_distribution`).
            Exposed primarily for model-transparency visualizations (the
            effect of the systemic factor on the loss distribution
            shape); pricing itself integrates over `Z` via
            :meth:`portfolio_loss_distribution`.
        """
        default_probabilities = self.portfolio.default_probabilities(t)
        corr = (
            self.portfolio.correlations() if correlations is None else np.asarray(correlations)
        )
        lgd_buckets = discretize_loss_given_default(self.portfolio.loss_given_defaults(), self.loss_unit)
        return conditional_loss_distribution(
            default_probabilities, corr, systemic_factor, lgd_buckets, self._n_buckets
        )

    def portfolio_loss_distribution(
        self, t: float, correlations: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Unconditional portfolio loss distribution at time `t`.

        Parameters
        ----------
        t : float
            Time horizon in years.
        correlations : np.ndarray, optional
            Override for per-obligor one-factor correlations, shape
            ``(n_obligors,)``. If `None`, each obligor's own
            `correlation` attribute is used. A flat override (identical
            correlation for all obligors) is the standard device used
            in base correlation bootstrapping (see
            :mod:`credit_copula.base_correlation`).

        Returns
        -------
        np.ndarray
            Probability mass function over `self._loss_grid`.
        """
        default_probabilities = self.portfolio.default_probabilities(t)
        corr = (
            self.portfolio.correlations() if correlations is None else np.asarray(correlations)
        )
        return loss_distribution(
            default_probabilities,
            corr,
            self.portfolio.loss_given_defaults(),
            self.loss_unit,
            self._n_buckets,
            self.n_quadrature_points,
        )

    def expected_tranche_loss(
        self, t: float, tranche: Tranche, correlations: np.ndarray | None = None
    ) -> float:
        """
        Expected tranche loss at time `t`.

        Parameters
        ----------
        t : float
            Time horizon in years. ``t=0`` returns 0.0 by construction.
        tranche : Tranche
            Tranche specification.
        correlations : np.ndarray, optional
            Per-obligor correlation override, see
            :meth:`portfolio_loss_distribution`.

        Returns
        -------
        float
            Expected tranche loss in currency units.
        """
        if t <= 0.0:
            return 0.0
        distribution = self.portfolio_loss_distribution(t, correlations)
        return tranche_mod.expected_tranche_loss(
            self._loss_grid, distribution, tranche.attachment, tranche.detachment
        )

    def _expected_tranche_loss_curve(
        self, times: np.ndarray, tranche: Tranche, correlations: np.ndarray | None
    ) -> np.ndarray:
        """Evaluate expected tranche loss at each time in `times` (vector of scalars)."""
        return np.array(
            [self.expected_tranche_loss(t, tranche, correlations) for t in times]
        )

    def price_tranche(
        self,
        tranche: Tranche,
        maturity: float,
        correlations: np.ndarray | None = None,
        valuation_date: date | None = None,
    ) -> TranchePricingResult:
        """
        Price a tranche over its full term and return leg present values.

        The expected tranche loss curve is evaluated once on the union
        of a fine time grid (for protection leg integration) and the
        quarterly payment schedule (for premium leg valuation), to
        avoid redundant loss distribution computation -- the dominant
        cost of the pricing engine.

        Parameters
        ----------
        tranche : Tranche
            Tranche specification.
        maturity : float
            Tranche maturity in years.
        correlations : np.ndarray, optional
            Per-obligor correlation override, see
            :meth:`portfolio_loss_distribution`.
        valuation_date : date, optional
            If supplied, the premium payment schedule is built from
            real calendar dates (ACT/360 accrual, Modified Following
            business-day-adjusted payment dates, standard quarterly
            roll pattern) via
            :func:`credit_copula.tranche.generate_payment_schedule`,
            rather than the calendar-free, evenly-spaced approximation
            used when `valuation_date` is `None`.

        Returns
        -------
        TranchePricingResult
            Pricing outcome including par spread and leg present
            values.

        Raises
        ------
        ValueError
            If `maturity` is non-positive.
        """
        if maturity <= 0.0:
            raise ValueError("maturity must be strictly positive")

        payment_times, accrual_fractions = tranche_mod.generate_payment_schedule(
            maturity, self.payment_frequency, valuation_date=valuation_date
        )
        n_fine_steps = max(int(round(maturity * self.n_integration_steps_per_year)), 1)
        fine_grid = np.linspace(0.0, maturity, n_fine_steps + 1)

        all_times = np.union1d(fine_grid, payment_times)
        if all_times[0] != 0.0:
            all_times = np.concatenate(([0.0], all_times))

        etl_curve = self._expected_tranche_loss_curve(all_times, tranche, correlations)

        protection_pv = tranche_mod.protection_leg_pv(
            etl_curve, all_times, self.discount_curve
        )

        payment_idx = np.searchsorted(all_times, payment_times)
        etl_at_payments = etl_curve[payment_idx]

        period_start_times = np.concatenate(([0.0], payment_times[:-1]))
        period_start_idx = np.searchsorted(all_times, period_start_times)
        etl_at_period_start = etl_curve[period_start_idx]

        annuity = tranche_mod.premium_leg_annuity(
            tranche.notional,
            etl_at_payments,
            payment_times,
            accrual_fractions,
            self.discount_curve,
            expected_tranche_losses_at_period_start=etl_at_period_start,
        )

        spread = tranche_mod.par_spread(protection_pv, annuity)

        return TranchePricingResult(
            par_spread=spread,
            protection_leg_pv=protection_pv,
            risky_annuity=annuity,
            tranche_notional=tranche.notional,
            expected_tranche_loss_curve=etl_curve,
            integration_times=all_times,
        )
