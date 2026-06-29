"""Tests for credit_copula.tranche."""

from __future__ import annotations

import numpy as np
import pytest

from credit_copula.market_data import DiscountCurve
from credit_copula.tranche import (
    Tranche,
    expected_tranche_loss,
    fair_upfront,
    generate_payment_schedule,
    mark_to_market,
    par_spread,
    premium_leg_annuity,
    protection_leg_pv,
    tranche_loss_payoff,
)


class TestTranche:
    def test_notional_equals_detachment_minus_attachment(self) -> None:
        tranche = Tranche(attachment=3.0, detachment=7.0)
        assert tranche.notional == pytest.approx(4.0)

    def test_rejects_attachment_geq_detachment(self) -> None:
        with pytest.raises(ValueError):
            Tranche(attachment=7.0, detachment=3.0)

    def test_rejects_negative_attachment(self) -> None:
        with pytest.raises(ValueError):
            Tranche(attachment=-1.0, detachment=3.0)


class TestTrancheLossPayoff:
    def test_zero_loss_below_attachment(self) -> None:
        assert tranche_loss_payoff(2.0, attachment=3.0, detachment=7.0) == pytest.approx(0.0)

    def test_full_notional_above_detachment(self) -> None:
        assert tranche_loss_payoff(10.0, attachment=3.0, detachment=7.0) == pytest.approx(4.0)

    def test_linear_within_tranche(self) -> None:
        assert tranche_loss_payoff(5.0, attachment=3.0, detachment=7.0) == pytest.approx(2.0)

    def test_vectorized_evaluation(self) -> None:
        losses = np.array([0.0, 3.0, 5.0, 7.0, 10.0])
        payoff = tranche_loss_payoff(losses, attachment=3.0, detachment=7.0)
        np.testing.assert_allclose(payoff, [0.0, 0.0, 2.0, 4.0, 4.0])


class TestExpectedTrancheLoss:
    def test_matches_manual_computation(self) -> None:
        loss_grid = np.array([0.0, 1.0, 2.0, 3.0])
        probabilities = np.array([0.5, 0.3, 0.15, 0.05])
        etl = expected_tranche_loss(loss_grid, probabilities, attachment=1.0, detachment=2.0)
        expected = 0.5 * 0.0 + 0.3 * 0.0 + 0.15 * 1.0 + 0.05 * 1.0
        assert etl == pytest.approx(expected)


class TestGeneratePaymentSchedule:
    def test_final_payment_at_maturity(self) -> None:
        payment_times, _ = generate_payment_schedule(maturity=5.0, frequency=4)
        assert payment_times[-1] == pytest.approx(5.0)

    def test_quarterly_accrual_fraction(self) -> None:
        _, accrual = generate_payment_schedule(maturity=5.0, frequency=4)
        np.testing.assert_allclose(accrual, 0.25)

    def test_number_of_payments(self) -> None:
        payment_times, _ = generate_payment_schedule(maturity=3.0, frequency=4)
        assert len(payment_times) == 12

    def test_rejects_non_positive_maturity(self) -> None:
        with pytest.raises(ValueError):
            generate_payment_schedule(maturity=0.0)

    def test_date_based_schedule_final_payment_near_maturity(self) -> None:
        from datetime import date

        payment_times, _ = generate_payment_schedule(
            maturity=5.0, frequency=4, valuation_date=date(2025, 3, 20)
        )
        # ACT/365F time to a date 5 years later is close to, but not
        # exactly, 5.0 (calendar years are not all 365 days).
        assert payment_times[-1] == pytest.approx(5.0, abs=0.01)

    def test_date_based_schedule_number_of_payments(self) -> None:
        from datetime import date

        payment_times, _ = generate_payment_schedule(
            maturity=3.0, frequency=4, valuation_date=date(2025, 3, 20)
        )
        assert len(payment_times) == 12

    def test_date_based_schedule_accrual_fractions_vary_with_actual_days(self) -> None:
        from datetime import date

        _, accrual = generate_payment_schedule(maturity=2.0, frequency=4, valuation_date=date(2025, 3, 20))
        # ACT/360 accrual fractions differ slightly period-to-period because
        # quarterly periods do not all have the same actual day count
        # (e.g. a period spanning February differs from one that does not),
        # unlike the constant-fraction calendar-free path.
        assert not np.allclose(accrual, accrual[0])

    def test_date_based_and_calendar_free_schedules_have_similar_total_accrual(self) -> None:
        from datetime import date

        _, accrual_dates = generate_payment_schedule(maturity=5.0, frequency=4, valuation_date=date(2025, 3, 20))
        _, accrual_float = generate_payment_schedule(maturity=5.0, frequency=4)
        assert np.sum(accrual_dates) == pytest.approx(np.sum(accrual_float), rel=0.02)


class TestLegValuation:
    def test_protection_leg_pv_of_no_loss_is_zero(self) -> None:
        discount_curve = DiscountCurve(tenors=np.array([5.0]), zero_rates=np.array([0.03]))
        times = np.array([0.0, 1.0, 2.0, 5.0])
        etl = np.zeros_like(times)
        assert protection_leg_pv(etl, times, discount_curve) == pytest.approx(0.0)

    def test_protection_leg_pv_undiscounted_equals_total_loss(self) -> None:
        discount_curve = DiscountCurve(tenors=np.array([5.0]), zero_rates=np.array([1e-12]))
        times = np.array([0.0, 2.5, 5.0])
        etl = np.array([0.0, 0.4, 1.0])
        pv = protection_leg_pv(etl, times, discount_curve)
        assert pv == pytest.approx(1.0, rel=1e-6)

    def test_protection_leg_rejects_nonzero_start_time(self) -> None:
        discount_curve = DiscountCurve(tenors=np.array([5.0]), zero_rates=np.array([0.03]))
        with pytest.raises(ValueError):
            protection_leg_pv(np.array([0.0, 1.0]), np.array([0.5, 1.0]), discount_curve)

    def test_premium_leg_annuity_no_loss_matches_simple_annuity(self) -> None:
        discount_curve = DiscountCurve(tenors=np.array([5.0]), zero_rates=np.array([0.0]))
        payment_times = np.array([1.0, 2.0])
        accrual = np.array([1.0, 1.0])
        etl = np.zeros_like(payment_times)
        annuity = premium_leg_annuity(10.0, etl, payment_times, accrual, discount_curve)
        assert annuity == pytest.approx(20.0)

    def test_premium_leg_annuity_reduced_by_expected_loss(self) -> None:
        discount_curve = DiscountCurve(tenors=np.array([5.0]), zero_rates=np.array([0.0]))
        payment_times = np.array([1.0])
        accrual = np.array([1.0])
        etl = np.array([4.0])
        annuity = premium_leg_annuity(10.0, etl, payment_times, accrual, discount_curve)
        assert annuity == pytest.approx(6.0)

    def test_premium_leg_annuity_omits_accrued_on_default_by_default(self) -> None:
        discount_curve = DiscountCurve(tenors=np.array([5.0]), zero_rates=np.array([0.0]))
        annuity = premium_leg_annuity(
            10.0, np.array([4.0]), np.array([1.0]), np.array([1.0]), discount_curve
        )
        assert annuity == pytest.approx(6.0)

    def test_premium_leg_annuity_adds_half_period_accrued_on_default(self) -> None:
        # Coupon-only term: DF(1)*1.0*(10-4) = 6.0.
        # Accrued-on-default term: DF(1)*0.5*1.0*(4-0) = 2.0.
        discount_curve = DiscountCurve(tenors=np.array([5.0]), zero_rates=np.array([0.0]))
        annuity = premium_leg_annuity(
            10.0,
            np.array([4.0]),
            np.array([1.0]),
            np.array([1.0]),
            discount_curve,
            expected_tranche_losses_at_period_start=np.array([0.0]),
        )
        assert annuity == pytest.approx(8.0)

    def test_accrued_on_default_term_scales_with_loss_increment(self) -> None:
        discount_curve = DiscountCurve(tenors=np.array([5.0]), zero_rates=np.array([0.0]))
        no_loss_in_period = premium_leg_annuity(
            10.0, np.array([4.0]), np.array([1.0]), np.array([1.0]), discount_curve,
            expected_tranche_losses_at_period_start=np.array([4.0]),
        )
        # No loss increment within the period: accrued-on-default term is zero,
        # so this matches the coupon-only annuity exactly.
        assert no_loss_in_period == pytest.approx(6.0)


class TestParSpreadAndMarkToMarket:
    def test_par_spread_equates_legs(self) -> None:
        spread = par_spread(protection_pv=5.0, annuity=50.0)
        assert spread == pytest.approx(0.1)

    def test_par_spread_rejects_non_positive_annuity(self) -> None:
        with pytest.raises(ValueError):
            par_spread(protection_pv=5.0, annuity=0.0)

    def test_mark_to_market_zero_at_par_spread(self) -> None:
        protection_pv, annuity = 5.0, 50.0
        spread = par_spread(protection_pv, annuity)
        mtm = mark_to_market(spread, protection_pv, annuity)
        assert mtm == pytest.approx(0.0, abs=1e-12)

    def test_mark_to_market_reflects_upfront(self) -> None:
        mtm = mark_to_market(contractual_spread=0.0, protection_pv=10.0, annuity=0.0, upfront_payment=3.0)
        assert mtm == pytest.approx(7.0)


class TestFairUpfront:
    def test_zero_at_fixed_spread_equal_to_par_spread(self) -> None:
        protection_pv, annuity, notional = 5.0, 50.0, 10.0
        par = par_spread(protection_pv, annuity)
        assert fair_upfront(protection_pv, annuity, par, notional) == pytest.approx(0.0, abs=1e-12)

    def test_positive_upfront_when_fixed_spread_below_par(self) -> None:
        # A below-par fixed coupon (the equity-tranche convention) requires
        # a positive upfront payment from protection buyer to seller.
        protection_pv, annuity, notional = 5.0, 50.0, 10.0
        below_par_spread = par_spread(protection_pv, annuity) / 2.0
        upfront = fair_upfront(protection_pv, annuity, below_par_spread, notional)
        assert upfront > 0.0

    def test_mark_to_market_is_zero_at_fair_upfront(self) -> None:
        protection_pv, annuity, notional, fixed_spread = 6.0, 40.0, 12.0, 0.05
        upfront_pct = fair_upfront(protection_pv, annuity, fixed_spread, notional)
        mtm = mark_to_market(fixed_spread, protection_pv, annuity, upfront_payment=upfront_pct * notional)
        assert mtm == pytest.approx(0.0, abs=1e-10)

    def test_rejects_non_positive_notional(self) -> None:
        with pytest.raises(ValueError):
            fair_upfront(5.0, 50.0, 0.01, 0.0)
