"""
Market quote representation for credit index tranches.

Index tranches are not uniformly quoted as a par running spread. By
market convention (CDX.NA.IG, CDX.NA.HY, and iTraxx Europe):

- the most subordinated ("equity") tranche typically trades with a
  fixed, standardized contractual running spread (e.g. 500 bps for
  CDX.NA.IG 0-3%) plus a quoted **upfront** payment, since a pure par
  running spread for a tranche this risky would imply an impractically
  high periodic coupon;
- more senior tranches typically trade as a pure **par running
  spread**, with no upfront exchanged.

A calibration routine that assumes every quote is a par running spread
will silently misprice or fail to calibrate to upfront-quoted
tranches. This module defines a structured quote representation that
records, per tranche, which convention applies, so that calibration
code can dispatch on it explicitly rather than assuming one
convention throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from credit_copula import tranche as tranche_mod
from credit_copula.conventions import DayCountConvention

__all__ = ["QuoteType", "TrancheQuote"]


class QuoteType(Enum):
    """
    Market quoting convention for a single tranche.

    Attributes
    ----------
    PAR_SPREAD
        The tranche trades on a running spread only; the quoted value
        is the fair (par) running spread that equates the protection
        and premium legs, with no upfront payment.
    UPFRONT
        The tranche trades with a fixed, standardized contractual
        running spread (``TrancheQuote.running_spread``) plus a quoted
        upfront payment (``TrancheQuote.upfront_pct``, as a fraction of
        tranche notional) that brings the contract to fair value at
        inception.
    """

    PAR_SPREAD = "PAR_SPREAD"
    UPFRONT = "UPFRONT"


@dataclass(frozen=True)
class TrancheQuote:
    """
    A single market quote for an index tranche.

    Parameters
    ----------
    index_name : str
        Reference index identifier (e.g. ``"CDX.NA.IG"``).
    attachment_pct : float
        Tranche attachment point, as a fraction of index notional, in
        :math:`[0, 1)`.
    detachment_pct : float
        Tranche detachment point, as a fraction of index notional, in
        :math:`(0, 1]`, with ``detachment_pct > attachment_pct``.
    valuation_date : date
        Date as of which the quote is observed and the contract is to
        be priced.
    maturity_date : date
        Contract maturity date.
    quote_type : QuoteType
        Quoting convention; see :class:`QuoteType`.
    running_spread : float
        Contractual running spread, in decimal form (e.g. ``0.05`` for
        500 bps). For `QuoteType.PAR_SPREAD` quotes, this is the
        quoted fair spread itself; for `QuoteType.UPFRONT` quotes, this
        is the fixed standardized coupon (the quoted, market-clearing
        value is `upfront_pct`, not this field).
    upfront_pct : float, optional
        Quoted upfront payment, as a fraction of tranche notional,
        required for `QuoteType.UPFRONT` quotes and otherwise `None`.
        Positive values denote a payment from protection buyer to
        protection seller at inception (the market convention for
        index tranche equity pieces).
    recovery_rate : float, default=0.40
        Recovery rate assumption underlying the quote.
    currency : str, default="USD"
        Settlement currency.
    notional : float, default=1.0
        Index notional, in currency units, that `attachment_pct` and
        `detachment_pct` are measured against.
    payment_frequency_months : int, default=3
        Premium payment frequency, in months (3 = quarterly).
    day_count : DayCountConvention, default=ACT_360
        Day-count convention for premium accrual.

    Raises
    ------
    ValueError
        If `attachment_pct`/`detachment_pct` are out of range or
        misordered, if `maturity_date` does not strictly follow
        `valuation_date`, or if `upfront_pct` is missing for a
        `QuoteType.UPFRONT` quote (or present for a
        `QuoteType.PAR_SPREAD` quote).
    """

    index_name: str
    attachment_pct: float
    detachment_pct: float
    valuation_date: date
    maturity_date: date
    quote_type: QuoteType
    running_spread: float
    upfront_pct: float | None = None
    recovery_rate: float = 0.40
    currency: str = "USD"
    notional: float = 1.0
    payment_frequency_months: int = 3
    day_count: DayCountConvention = DayCountConvention.ACT_360

    def __post_init__(self) -> None:
        if not (0.0 <= self.attachment_pct < self.detachment_pct <= 1.0):
            raise ValueError(
                "attachment_pct and detachment_pct must satisfy "
                "0 <= attachment_pct < detachment_pct <= 1, "
                f"got ({self.attachment_pct}, {self.detachment_pct})"
            )
        if self.maturity_date <= self.valuation_date:
            raise ValueError("maturity_date must be strictly after valuation_date")
        if not (0.0 <= self.recovery_rate < 1.0):
            raise ValueError("recovery_rate must lie in [0, 1)")
        if self.quote_type is QuoteType.UPFRONT and self.upfront_pct is None:
            raise ValueError("upfront_pct is required for QuoteType.UPFRONT quotes")
        if self.quote_type is QuoteType.PAR_SPREAD and self.upfront_pct is not None:
            raise ValueError("upfront_pct must be None for QuoteType.PAR_SPREAD quotes")

    @property
    def tranche_notional(self) -> float:
        """Tranche notional in currency units: ``(detachment_pct - attachment_pct) * notional``."""
        return (self.detachment_pct - self.attachment_pct) * self.notional

    def quoted_value(self) -> float:
        """
        The single market-quoted number this quote represents.

        Returns
        -------
        float
            `running_spread` for `QuoteType.PAR_SPREAD` quotes, or
            `upfront_pct` for `QuoteType.UPFRONT` quotes -- i.e. the
            value a calibration routine should match.
        """
        if self.quote_type is QuoteType.PAR_SPREAD:
            return self.running_spread
        return self.upfront_pct  # type: ignore[return-value]

    def model_implied_value(self, protection_pv: float, annuity: float) -> float:
        """
        Translate model leg present values into the quantity comparable to this quote.

        For a `QuoteType.PAR_SPREAD` quote, this is the fair running
        spread

        .. math::
            s^{*} = \\frac{V_{\\text{protection}}}{\\text{Annuity}}

        For a `QuoteType.UPFRONT` quote, the contractual running spread
        is fixed at `running_spread`, and the model-implied upfront
        (as a fraction of tranche notional) that brings the contract to
        fair value is

        .. math::
            U^{*} = \\frac{V_{\\text{protection}} - s_{\\text{fixed}} \\cdot \\text{Annuity}}{N}

        consistent with :func:`credit_copula.tranche.mark_to_market`
        evaluated at zero.

        Parameters
        ----------
        protection_pv : float
            Model protection leg present value.
        annuity : float
            Model risky annuity (premium leg PV per unit running
            spread).

        Returns
        -------
        float
            Model-implied par spread or upfront fraction, matching
            this quote's `quote_type`.

        Raises
        ------
        ValueError
            If `quote_type` is `QuoteType.PAR_SPREAD` and `annuity` is
            not strictly positive (no fair spread can be implied).
        """
        if self.quote_type is QuoteType.PAR_SPREAD:
            return tranche_mod.par_spread(protection_pv, annuity)
        return tranche_mod.fair_upfront(
            protection_pv, annuity, self.running_spread, self.tranche_notional
        )
