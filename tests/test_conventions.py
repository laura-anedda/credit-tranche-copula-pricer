"""Tests for credit_copula.conventions."""

from __future__ import annotations

from datetime import date

import pytest

from credit_copula.conventions import (
    BusinessDayConvention,
    CashflowPeriod,
    DayCountConvention,
    add_months,
    adjust_business_day,
    build_cds_premium_schedule,
    generate_unadjusted_period_dates,
    is_business_day,
    year_fraction,
)


class TestYearFraction:
    def test_act_360_full_quarter(self) -> None:
        # 20-Mar-2025 to 20-Jun-2025 spans 92 actual days.
        yf = year_fraction(date(2025, 3, 20), date(2025, 6, 20), DayCountConvention.ACT_360)
        assert yf == pytest.approx(92 / 360.0)

    def test_act_365f_full_year(self) -> None:
        yf = year_fraction(date(2024, 1, 1), date(2025, 1, 1), DayCountConvention.ACT_365F)
        assert yf == pytest.approx(366 / 365.0)  # 2024 is a leap year

    def test_thirty_360_full_quarter(self) -> None:
        yf = year_fraction(date(2025, 3, 20), date(2025, 6, 20), DayCountConvention.THIRTY_360)
        assert yf == pytest.approx(90 / 360.0)

    def test_thirty_360_month_end_clamping(self) -> None:
        # 30/360: day-of-month 31 is clamped to 30 on both legs.
        yf = year_fraction(date(2025, 1, 31), date(2025, 3, 31), DayCountConvention.THIRTY_360)
        assert yf == pytest.approx(60 / 360.0)

    def test_rejects_end_before_start(self) -> None:
        with pytest.raises(ValueError):
            year_fraction(date(2025, 6, 20), date(2025, 3, 20), DayCountConvention.ACT_360)

    def test_zero_for_equal_dates(self) -> None:
        d = date(2025, 3, 20)
        assert year_fraction(d, d, DayCountConvention.ACT_360) == pytest.approx(0.0)


class TestBusinessDayCalendar:
    def test_weekday_is_business_day(self) -> None:
        assert is_business_day(date(2025, 3, 20))  # Thursday

    def test_saturday_is_not_business_day(self) -> None:
        assert not is_business_day(date(2025, 3, 22))  # Saturday

    def test_sunday_is_not_business_day(self) -> None:
        assert not is_business_day(date(2025, 3, 23))  # Sunday


class TestAdjustBusinessDay:
    def test_none_convention_is_identity(self) -> None:
        d = date(2025, 3, 22)  # Saturday
        assert adjust_business_day(d, BusinessDayConvention.NONE) == d

    def test_following_rolls_forward_over_weekend(self) -> None:
        saturday = date(2025, 3, 22)
        assert adjust_business_day(saturday, BusinessDayConvention.FOLLOWING) == date(2025, 3, 24)

    def test_modified_following_rolls_forward_within_month(self) -> None:
        saturday = date(2025, 3, 22)
        assert adjust_business_day(saturday, BusinessDayConvention.MODIFIED_FOLLOWING) == date(2025, 3, 24)

    def test_modified_following_rolls_backward_across_month_end(self) -> None:
        # 31-May-2025 is a Saturday; rolling forward would cross into June,
        # so Modified Following instead rolls back to Friday 30-May-2025.
        month_end_saturday = date(2025, 5, 31)
        assert adjust_business_day(month_end_saturday, BusinessDayConvention.MODIFIED_FOLLOWING) == date(2025, 5, 30)

    def test_business_day_is_unchanged(self) -> None:
        thursday = date(2025, 3, 20)
        assert adjust_business_day(thursday, BusinessDayConvention.MODIFIED_FOLLOWING) == thursday


class TestAddMonths:
    def test_simple_addition(self) -> None:
        assert add_months(date(2025, 3, 20), 3) == date(2025, 6, 20)

    def test_simple_subtraction(self) -> None:
        assert add_months(date(2025, 6, 20), -3) == date(2025, 3, 20)

    def test_clamps_to_month_end(self) -> None:
        assert add_months(date(2025, 1, 31), 1) == date(2025, 2, 28)

    def test_crosses_year_boundary(self) -> None:
        assert add_months(date(2025, 11, 20), 3) == date(2026, 2, 20)


class TestGenerateUnadjustedPeriodDates:
    def test_endpoints_match_input(self) -> None:
        boundaries = generate_unadjusted_period_dates(date(2025, 3, 20), date(2030, 3, 20))
        assert boundaries[0] == date(2025, 3, 20)
        assert boundaries[-1] == date(2030, 3, 20)

    def test_standard_quarterly_roll_pattern(self) -> None:
        boundaries = generate_unadjusted_period_dates(date(2025, 3, 20), date(2026, 3, 20))
        assert boundaries == [
            date(2025, 3, 20), date(2025, 6, 20), date(2025, 9, 20),
            date(2025, 12, 20), date(2026, 3, 20),
        ]

    def test_produces_stub_period_for_non_aligned_effective_date(self) -> None:
        # Effective date not on a standard roll: backward generation from
        # maturity produces a short stub as the first period.
        boundaries = generate_unadjusted_period_dates(date(2025, 4, 5), date(2026, 3, 20))
        assert boundaries[0] == date(2025, 4, 5)
        assert boundaries[1] == date(2025, 6, 20)

    def test_rejects_non_increasing_dates(self) -> None:
        with pytest.raises(ValueError):
            generate_unadjusted_period_dates(date(2026, 3, 20), date(2025, 3, 20))


class TestBuildCdsPremiumSchedule:
    def test_schedule_covers_full_term(self) -> None:
        schedule = build_cds_premium_schedule(date(2025, 3, 20), date(2026, 3, 20))
        assert schedule[0].accrual_start == date(2025, 3, 20)
        assert schedule[-1].accrual_end == date(2026, 3, 20)
        assert len(schedule) == 4

    def test_periods_are_contiguous(self) -> None:
        schedule = build_cds_premium_schedule(date(2025, 3, 20), date(2027, 3, 20))
        for previous, current in zip(schedule[:-1], schedule[1:]):
            assert previous.accrual_end == current.accrual_start

    def test_year_fractions_sum_to_actual_days_over_360(self) -> None:
        # ACT/360 systematically inflates the year-fraction sum relative to
        # calendar years, by a factor of approximately 365.25/360 (~1.46%
        # per year) -- expected and correct behaviour of the convention,
        # not an approximation error.
        start, end = date(2025, 3, 20), date(2030, 3, 20)
        schedule = build_cds_premium_schedule(start, end)
        total_year_fraction = sum(period.year_fraction for period in schedule)
        assert total_year_fraction == pytest.approx((end - start).days / 360.0)

    def test_payment_dates_are_business_days(self) -> None:
        schedule = build_cds_premium_schedule(date(2025, 3, 20), date(2026, 3, 20))
        for period in schedule:
            assert is_business_day(period.payment_date)

    def test_returns_cashflow_period_instances(self) -> None:
        schedule = build_cds_premium_schedule(date(2025, 3, 20), date(2025, 9, 20))
        assert all(isinstance(period, CashflowPeriod) for period in schedule)
