"""Tests for credit_copula.market_data."""

from __future__ import annotations

import numpy as np
import pytest

from credit_copula.market_data import CreditCurve, DiscountCurve, bootstrap_hazard_rates


class TestDiscountCurve:
    def test_discount_factor_at_pillar_is_consistent_with_zero_rate(self) -> None:
        curve = DiscountCurve(tenors=np.array([1.0, 5.0]), zero_rates=np.array([0.02, 0.03]))
        df_5y = curve.discount_factor(5.0)
        assert df_5y == pytest.approx(np.exp(-0.03 * 5.0))

    def test_discount_factor_decreases_with_maturity(self) -> None:
        curve = DiscountCurve(tenors=np.array([1.0, 2.0, 5.0]), zero_rates=np.full(3, 0.03))
        dfs = curve.discount_factor(np.array([1.0, 2.0, 5.0]))
        assert np.all(np.diff(dfs) < 0.0)

    def test_discount_factor_is_strictly_positive(self) -> None:
        curve = DiscountCurve(tenors=np.array([1.0, 10.0, 30.0]), zero_rates=np.array([0.01, 0.05, 0.08]))
        dfs = curve.discount_factor(np.linspace(0.0, 40.0, 50))
        assert np.all(dfs > 0.0)

    def test_zero_at_t_equals_zero_gives_discount_factor_one(self) -> None:
        curve = DiscountCurve(tenors=np.array([1.0]), zero_rates=np.array([0.05]))
        assert curve.discount_factor(0.0) == pytest.approx(1.0)

    def test_rejects_negative_time(self) -> None:
        curve = DiscountCurve(tenors=np.array([1.0]), zero_rates=np.array([0.05]))
        with pytest.raises(ValueError):
            curve.discount_factor(-1.0)

    def test_rejects_non_increasing_tenors(self) -> None:
        with pytest.raises(ValueError):
            DiscountCurve(tenors=np.array([2.0, 1.0]), zero_rates=np.array([0.02, 0.03]))

    def test_zero_rate_at_pillar_matches_input(self) -> None:
        curve = DiscountCurve(tenors=np.array([1.0, 5.0]), zero_rates=np.array([0.02, 0.03]))
        np.testing.assert_allclose(curve.zero_rate(np.array([1.0, 5.0])), [0.02, 0.03])

    def test_zero_rate_at_origin_equals_first_pillar_rate(self) -> None:
        curve = DiscountCurve(tenors=np.array([1.0, 5.0]), zero_rates=np.array([0.02, 0.03]))
        assert curve.zero_rate(0.0) == pytest.approx(0.02)

    def test_interpolation_is_flat_forward_not_linear_zero_rate(self) -> None:
        # Flat-forward interpolation implies the forward rate between
        # pillars is constant, so the discount factor at the midpoint
        # equals the geometric mean of the pillar discount factors --
        # not the discount factor implied by linearly-interpolated zero
        # rates, which would differ whenever the two pillar rates differ.
        curve = DiscountCurve(tenors=np.array([1.0, 5.0]), zero_rates=np.array([0.01, 0.05]))
        midpoint = 3.0
        df_1, df_5 = curve.discount_factor(np.array([1.0, 5.0]))
        weight = (midpoint - 1.0) / (5.0 - 1.0)
        expected_flat_forward_df = df_1 ** (1 - weight) * df_5**weight
        linear_zero_rate_df = np.exp(-(0.01 + weight * (0.05 - 0.01)) * midpoint)
        assert curve.discount_factor(midpoint) == pytest.approx(expected_flat_forward_df)
        assert curve.discount_factor(midpoint) != pytest.approx(linear_zero_rate_df)

    def test_forward_rate_is_constant_within_a_pillar_interval(self) -> None:
        # Under flat-forward interpolation, the instantaneous forward rate
        # -d(ln DF)/dt is constant between pillars; sampling the discount
        # factor at several intermediate points must therefore lie on a
        # single exponential curve (equal forward rate between consecutive
        # sample pairs).
        curve = DiscountCurve(tenors=np.array([1.0, 10.0]), zero_rates=np.array([0.015, 0.04]))
        sample_times = np.array([2.0, 4.0, 6.0, 8.0])
        log_df = np.log(curve.discount_factor(sample_times))
        forward_rates = -np.diff(log_df) / np.diff(sample_times)
        np.testing.assert_allclose(forward_rates, forward_rates[0], rtol=1e-10)

    def test_extrapolation_beyond_last_pillar_uses_terminal_forward_rate(self) -> None:
        curve = DiscountCurve(tenors=np.array([1.0, 5.0]), zero_rates=np.array([0.02, 0.03]))
        df_5 = curve.discount_factor(5.0)
        df_7 = curve.discount_factor(7.0)
        assert df_7 == pytest.approx(df_5 * np.exp(-0.03 * 2.0))


class TestCreditCurve:
    def test_survival_probability_at_zero_is_one(self) -> None:
        curve = CreditCurve(tenors=np.array([5.0]), hazard_rates=np.array([0.02]), recovery_rate=0.4)
        assert curve.survival_probability(0.0) == pytest.approx(1.0)

    def test_survival_probability_matches_constant_hazard_formula(self) -> None:
        curve = CreditCurve(tenors=np.array([10.0]), hazard_rates=np.array([0.05]), recovery_rate=0.4)
        expected = np.exp(-0.05 * 3.0)
        assert curve.survival_probability(3.0) == pytest.approx(expected)

    def test_survival_probability_is_monotonically_decreasing(self) -> None:
        curve = CreditCurve(
            tenors=np.array([1.0, 3.0, 5.0]),
            hazard_rates=np.array([0.01, 0.02, 0.015]),
            recovery_rate=0.4,
        )
        times = np.linspace(0.0, 8.0, 50)
        survival = curve.survival_probability(times)
        assert np.all(np.diff(survival) <= 1e-12)

    def test_default_probability_complements_survival(self) -> None:
        curve = CreditCurve(tenors=np.array([5.0]), hazard_rates=np.array([0.03]), recovery_rate=0.4)
        t = 2.5
        assert curve.default_probability(t) == pytest.approx(1.0 - curve.survival_probability(t))

    def test_default_probability_is_monotonically_increasing(self) -> None:
        curve = CreditCurve(
            tenors=np.array([1.0, 3.0, 5.0]),
            hazard_rates=np.array([0.01, 0.02, 0.015]),
            recovery_rate=0.4,
        )
        times = np.linspace(0.0, 8.0, 50)
        default_probability = curve.default_probability(times)
        assert np.all(np.diff(default_probability) >= -1e-12)

    def test_rejects_negative_hazard_rate(self) -> None:
        with pytest.raises(ValueError):
            CreditCurve(tenors=np.array([1.0]), hazard_rates=np.array([-0.01]), recovery_rate=0.4)

    def test_rejects_invalid_recovery_rate(self) -> None:
        with pytest.raises(ValueError):
            CreditCurve(tenors=np.array([1.0]), hazard_rates=np.array([0.01]), recovery_rate=1.0)

    def test_extrapolates_beyond_last_pillar_using_terminal_hazard(self) -> None:
        curve = CreditCurve(tenors=np.array([5.0]), hazard_rates=np.array([0.02]), recovery_rate=0.4)
        survival_10y = curve.survival_probability(10.0)
        assert survival_10y == pytest.approx(np.exp(-0.02 * 10.0))


class TestBootstrapHazardRates:
    def test_bootstrapped_curve_reprices_input_cds_spreads(self) -> None:
        discount_curve = DiscountCurve(tenors=np.array([1.0, 3.0, 5.0, 10.0]), zero_rates=np.full(4, 0.025))
        cds_tenors = np.array([1.0, 3.0, 5.0])
        cds_spreads = np.array([0.005, 0.009, 0.012])
        recovery_rate = 0.4

        credit_curve = bootstrap_hazard_rates(cds_tenors, cds_spreads, recovery_rate, discount_curve)

        # Re-derive the CDS par spread implied by the bootstrapped curve for
        # each maturity pillar and confirm it reproduces the input quote.
        # The independent verification includes the same accrued-premium-
        # on-default term (midpoint approximation) used by the bootstrap
        # itself, since the curve is calibrated against that valuation.
        for maturity, market_spread in zip(cds_tenors, cds_spreads):
            payment_times = np.linspace(0.25, maturity, int(maturity * 4))
            period_start_times = np.concatenate(([0.0], payment_times[:-1]))
            survival = credit_curve.survival_probability(payment_times)
            survival_at_start = credit_curve.survival_probability(period_start_times)
            discount = discount_curve.discount_factor(payment_times)
            coupon_leg = np.sum(discount * 0.25 * survival)
            accrued_on_default = np.sum(discount * 0.5 * 0.25 * (survival_at_start - survival))
            premium_leg = coupon_leg + accrued_on_default

            grid = np.linspace(0.0, maturity, int(maturity * 16) + 1)
            survival_grid = credit_curve.survival_probability(grid)
            discount_grid = discount_curve.discount_factor(grid)
            protection_leg = (1.0 - recovery_rate) * np.sum(
                discount_grid[1:] * (survival_grid[:-1] - survival_grid[1:])
            )

            implied_spread = protection_leg / premium_leg
            assert implied_spread == pytest.approx(market_spread, rel=1e-3)

    def test_hazard_rates_are_non_negative(self) -> None:
        discount_curve = DiscountCurve(tenors=np.array([1.0, 10.0]), zero_rates=np.full(2, 0.03))
        cds_tenors = np.array([3.0, 7.0])
        cds_spreads = np.array([0.01, 0.015])
        credit_curve = bootstrap_hazard_rates(cds_tenors, cds_spreads, 0.4, discount_curve)
        assert np.all(credit_curve.hazard_rates >= 0.0)
