"""
Date, day-count, business-day, and CDS payment schedule conventions.

This module implements the deterministic calendar layer underlying CDS
and credit index tranche cash flow generation: day-count fractions,
business-day adjustment, and the standard ISDA quarterly premium
schedule (backward generation from the maturity date in 3-month steps,
producing the conventional 20-Mar/20-Jun/20-Sep/20-Dec roll pattern
when the maturity date itself falls on a roll date).

Scope decision: business-day adjustment here uses a weekend-only
calendar (Saturday/Sunday). No jurisdiction-specific holiday calendar
is implemented, since holiday calendars are currency- and
exchange-specific reference data that must be sourced and maintained
externally (e.g. via a market-data vendor or a dedicated calendar
library), not a modelling choice internal to the pricing engine. This
affects only the exact business-day roll of payment dates by at most a
few days around public holidays; day-count fractions and protection
periods, which dominate valuation, are computed from unadjusted period
dates and are unaffected. 

"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

__all__ = [
    "DayCountConvention",
    "BusinessDayConvention",
    "year_fraction",
    "is_business_day",
    "adjust_business_day",
    "add_months",
    "CashflowPeriod",
    "generate_unadjusted_period_dates",
    "build_cds_premium_schedule",
]


class DayCountConvention(Enum):
    """
    Day-count basis for converting a date interval into a year fraction.

    Attributes
    ----------
    ACT_360
        Actual/360: ``(end - start).days / 360``. The market-standard
        convention for CDS and credit index tranche premium accrual.
    ACT_365F
        Actual/365 (Fixed): ``(end - start).days / 365``. Used here for
        converting calendar dates into the year-fraction time variable
        consumed by the discount and credit curves
        (:mod:`credit_copula.market_data`).
    THIRTY_360
        30/360 (Bond Basis, ISDA "30U/360" variant): each month treated
        as having 30 days. Provided for completeness and for
        instruments quoted on this basis; not used by the CDS schedule
        builder in this module, which follows ACT/360 by market
        convention.
    """

    ACT_360 = "ACT/360"
    ACT_365F = "ACT/365F"
    THIRTY_360 = "30/360"


class BusinessDayConvention(Enum):
    """
    Business-day adjustment rule applied to an unadjusted schedule date.

    Attributes
    ----------
    NONE
        No adjustment; the unadjusted date is used as-is.
    FOLLOWING
        Roll forward to the next business day.
    MODIFIED_FOLLOWING
        Roll forward to the next business day, unless doing so would
        cross into the next calendar month, in which case roll backward
        to the preceding business day instead. The standard convention
        for CDS premium payment dates.
    """

    NONE = "NONE"
    FOLLOWING = "FOLLOWING"
    MODIFIED_FOLLOWING = "MODIFIED_FOLLOWING"


def year_fraction(start: date, end: date, convention: DayCountConvention) -> float:
    """
    Compute the year fraction between two dates under a day-count convention.

    Parameters
    ----------
    start : date
        Period start date (inclusive).
    end : date
        Period end date (exclusive), with ``end >= start``.
    convention : DayCountConvention
        Day-count basis.

    Returns
    -------
    float
        Year fraction between `start` and `end`.

    Raises
    ------
    ValueError
        If ``end < start``.
    """
    if end < start:
        raise ValueError("end date must not precede start date")

    if convention is DayCountConvention.ACT_360:
        return (end - start).days / 360.0
    if convention is DayCountConvention.ACT_365F:
        return (end - start).days / 365.0
    if convention is DayCountConvention.THIRTY_360:
        d1 = min(start.day, 30)
        d2 = min(end.day, 30) if d1 == 30 else end.day
        return (
            360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)
        ) / 360.0
    raise ValueError(f"unsupported day-count convention: {convention}")


def is_business_day(d: date) -> bool:
    """
    Whether `d` is a business day under the weekend-only calendar used in this module.

    Parameters
    ----------
    d : date
        Date to test.

    Returns
    -------
    bool
        `True` for Monday through Friday; `False` for Saturday/Sunday.
        No jurisdiction-specific holidays are excluded (see module
        docstring).
    """
    return d.weekday() < 5


def adjust_business_day(d: date, convention: BusinessDayConvention) -> date:
    """
    Apply a business-day adjustment rule to an unadjusted date.

    Parameters
    ----------
    d : date
        Unadjusted date.
    convention : BusinessDayConvention
        Adjustment rule to apply.

    Returns
    -------
    date
        Adjusted date.
    """
    if convention is BusinessDayConvention.NONE:
        return d

    rolled_forward = d
    while not is_business_day(rolled_forward):
        rolled_forward += timedelta(days=1)

    if convention is BusinessDayConvention.FOLLOWING:
        return rolled_forward

    if convention is BusinessDayConvention.MODIFIED_FOLLOWING:
        if rolled_forward.month != d.month:
            rolled_backward = d
            while not is_business_day(rolled_backward):
                rolled_backward -= timedelta(days=1)
            return rolled_backward
        return rolled_forward

    raise ValueError(f"unsupported business-day convention: {convention}")


def add_months(d: date, months: int) -> date:
    """
    Add (or subtract, for negative `months`) a whole number of calendar months to a date.

    Day-of-month is preserved where possible; if the target month has
    fewer days than `d.day` (e.g. adding one month to 31 January), the
    result is clamped to the last day of the target month, the
    standard end-of-month convention for calendar date arithmetic.

    Parameters
    ----------
    d : date
        Starting date.
    months : int
        Number of months to add (negative to subtract).

    Returns
    -------
    date
        Resulting date.
    """
    total_months = d.month - 1 + months
    year = d.year + total_months // 12
    month = total_months % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


@dataclass(frozen=True)
class CashflowPeriod:
    """
    A single premium accrual period of a CDS or tranche premium schedule.

    Attributes
    ----------
    accrual_start : date
        Start of the accrual period (unadjusted).
    accrual_end : date
        End of the accrual period (unadjusted).
    payment_date : date
        Business-day-adjusted date on which the premium for this period
        is paid.
    year_fraction : float
        Accrual year fraction between `accrual_start` and
        `accrual_end`, computed on the period's day-count convention.
    """

    accrual_start: date
    accrual_end: date
    payment_date: date
    year_fraction: float


def generate_unadjusted_period_dates(
    effective_date: date, maturity_date: date, frequency_months: int = 3
) -> list[date]:
    """
    Generate unadjusted accrual period boundary dates via backward generation from maturity.

    Standard CDS and credit index tranche schedules are built backward
    from the maturity date in fixed-frequency steps (3 months, by
    market convention) until reaching or passing the effective
    (protection start) date; the final, partial step produces the
    schedule's stub period. When `maturity_date` itself falls on a
    standard CDS roll date (20 Mar / 20 Jun / 20 Sep / 20 Dec), this
    reproduces the conventional quarterly roll pattern exactly.

    Parameters
    ----------
    effective_date : date
        Protection start date (the first accrual period's start).
    maturity_date : date
        Final accrual period's end date.
    frequency_months : int, default=3
        Number of months between accrual period boundaries (3 =
        quarterly, the CDS market standard).

    Returns
    -------
    list[date]
        Strictly increasing boundary dates, with
        ``result[0] == effective_date`` and
        ``result[-1] == maturity_date``.

    Raises
    ------
    ValueError
        If ``maturity_date <= effective_date``.
    """
    if maturity_date <= effective_date:
        raise ValueError("maturity_date must be strictly after effective_date")

    boundaries = [maturity_date]
    current = maturity_date
    while True:
        current = add_months(current, -frequency_months)
        if current <= effective_date:
            break
        boundaries.append(current)
    boundaries.append(effective_date)
    return sorted(set(boundaries))


def build_cds_premium_schedule(
    effective_date: date,
    maturity_date: date,
    frequency_months: int = 3,
    day_count: DayCountConvention = DayCountConvention.ACT_360,
    business_day_convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
) -> tuple[CashflowPeriod, ...]:
    """
    Build the full premium accrual and payment schedule for a CDS or tranche leg.

    Parameters
    ----------
    effective_date : date
        Protection start date.
    maturity_date : date
        Contract maturity date.
    frequency_months : int, default=3
        Premium payment frequency in months (3 = quarterly).
    day_count : DayCountConvention, default=ACT_360
        Day-count convention for accrual year fractions (ACT/360 is
        the CDS market standard).
    business_day_convention : BusinessDayConvention, default=MODIFIED_FOLLOWING
        Business-day adjustment applied to each payment date. Accrual
        period boundaries themselves are left unadjusted, consistent
        with standard CDS schedule construction, in which only the
        payment date (not the accrual period definition) is rolled to
        a business day.

    Returns
    -------
    tuple[CashflowPeriod, ...]
        One entry per accrual period, in chronological order.
    """
    boundaries = generate_unadjusted_period_dates(effective_date, maturity_date, frequency_months)
    periods = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        payment_date = adjust_business_day(end, business_day_convention)
        periods.append(
            CashflowPeriod(
                accrual_start=start,
                accrual_end=end,
                payment_date=payment_date,
                year_fraction=year_fraction(start, end, day_count),
            )
        )
    return tuple(periods)
