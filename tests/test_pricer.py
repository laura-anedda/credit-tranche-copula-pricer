"""Integration tests for credit_copula.pricer."""

from __future__ import annotations

import numpy as np
import pytest

from credit_copula.diagnostics import assess_quadrature_convergence, check_expected_tranche_loss_bounds
from credit_copula.market_data import DiscountCurve
from credit_copula.portfolio import CreditPortfolio
from credit_copula.pricer import CDOPricer
from credit_copula.tranche import Tranche


@pytest.fixture
def fast_pricer(small_homogeneous_portfolio: CreditPortfolio, flat_discount_curve: DiscountCurve) -> CDOPricer:
    """A pricer configured with coarse settings for fast unit testing."""
    return CDOPricer(
        portfolio=small_homogeneous_portfolio,
        discount_curve=flat_discount_curve,
        loss_unit=0.6,
        n_quadrature_points=16,
        payment_frequency=4,
        n_integration_steps_per_year=4,
    )


class TestCDOPricer:
    def test_loss_distribution_sums_to_one(self, fast_pricer: CDOPricer) -> None:
        dist = fast_pricer.portfolio_loss_distribution(t=3.0)
        assert np.sum(dist) == pytest.approx(1.0, abs=1e-6)

    def test_loss_grid_matches_n_loss_buckets(self, fast_pricer: CDOPricer) -> None:
        assert fast_pricer.loss_grid.shape == (fast_pricer.n_loss_buckets,)
        assert fast_pricer.loss_grid[0] == pytest.approx(0.0)

    def test_conditional_loss_distribution_sums_to_one(self, fast_pricer: CDOPricer) -> None:
        dist = fast_pricer.conditional_loss_distribution(t=3.0, systemic_factor=0.0)
        assert np.sum(dist) == pytest.approx(1.0, abs=1e-6)

    def test_conditional_loss_distribution_integrates_to_unconditional(self, fast_pricer: CDOPricer) -> None:
        from credit_copula.numerics import gauss_hermite_nodes_weights

        nodes, weights = gauss_hermite_nodes_weights(fast_pricer.n_quadrature_points)
        integrated = sum(
            w * fast_pricer.conditional_loss_distribution(3.0, float(z)) for z, w in zip(nodes, weights)
        )
        unconditional = fast_pricer.portfolio_loss_distribution(3.0)
        np.testing.assert_allclose(integrated, unconditional, atol=1e-8)

    def test_extreme_negative_factor_concentrates_loss_at_maximum(self, fast_pricer: CDOPricer) -> None:
        dist = fast_pricer.conditional_loss_distribution(t=3.0, systemic_factor=-10.0)
        assert dist[-1] == pytest.approx(1.0, abs=1e-4)

    def test_equity_tranche_etl_exceeds_senior_tranche_etl(self, fast_pricer: CDOPricer) -> None:
        equity = Tranche(attachment=0.0, detachment=1.2)
        senior = Tranche(attachment=4.0, detachment=6.0)
        etl_equity = fast_pricer.expected_tranche_loss(3.0, equity)
        etl_senior = fast_pricer.expected_tranche_loss(3.0, senior)
        assert etl_equity > etl_senior

    def test_etl_is_zero_at_time_zero(self, fast_pricer: CDOPricer) -> None:
        tranche = Tranche(attachment=0.0, detachment=2.0)
        assert fast_pricer.expected_tranche_loss(0.0, tranche) == pytest.approx(0.0)

    def test_etl_is_non_decreasing_in_time(self, fast_pricer: CDOPricer) -> None:
        tranche = Tranche(attachment=0.0, detachment=2.0)
        times = np.array([1.0, 2.0, 3.0, 5.0])
        etl = np.array([fast_pricer.expected_tranche_loss(t, tranche) for t in times])
        assert np.all(np.diff(etl) >= -1e-10)

    def test_price_tranche_returns_positive_par_spread(self, fast_pricer: CDOPricer) -> None:
        tranche = Tranche(attachment=0.0, detachment=1.2)
        result = fast_pricer.price_tranche(tranche, maturity=5.0)
        assert result.par_spread > 0.0
        assert result.protection_leg_pv > 0.0
        assert result.risky_annuity > 0.0

    def test_price_tranche_with_real_valuation_date_is_close_to_calendar_free(
        self, fast_pricer: CDOPricer
    ) -> None:
        from datetime import date

        tranche = Tranche(attachment=0.0, detachment=1.2)
        calendar_free = fast_pricer.price_tranche(tranche, maturity=5.0)
        date_based = fast_pricer.price_tranche(tranche, maturity=5.0, valuation_date=date(2025, 3, 20))
        assert date_based.par_spread == pytest.approx(calendar_free.par_spread, rel=0.02)
        assert date_based.par_spread > 0.0

    def test_par_spread_reprices_leg_equality(self, fast_pricer: CDOPricer) -> None:
        tranche = Tranche(attachment=1.2, detachment=2.4)
        result = fast_pricer.price_tranche(tranche, maturity=5.0)
        replicated_premium_pv = result.par_spread * result.risky_annuity
        assert replicated_premium_pv == pytest.approx(result.protection_leg_pv, rel=1e-9)

    def test_risky_annuity_includes_accrued_premium_on_default(self, fast_pricer: CDOPricer) -> None:
        # The annuity returned by price_tranche includes the accrued-
        # premium-on-default term (see tranche.premium_leg_annuity), so it
        # must exceed the coupon-only annuity computed without it for a
        # tranche with strictly positive expected loss.
        from credit_copula import tranche as tranche_mod

        tranche = Tranche(attachment=0.0, detachment=1.2)
        result = fast_pricer.price_tranche(tranche, maturity=5.0)

        payment_times, accrual_fractions = tranche_mod.generate_payment_schedule(
            5.0, fast_pricer.payment_frequency
        )
        etl_at_payments = np.array(
            [fast_pricer.expected_tranche_loss(t, tranche) for t in payment_times]
        )
        coupon_only_annuity = tranche_mod.premium_leg_annuity(
            tranche.notional, etl_at_payments, payment_times, accrual_fractions, fast_pricer.discount_curve
        )
        assert result.risky_annuity > coupon_only_annuity

    def test_equity_tranche_par_spread_exceeds_senior(self, fast_pricer: CDOPricer) -> None:
        equity = Tranche(attachment=0.0, detachment=1.2)
        senior = Tranche(attachment=4.0, detachment=6.0)
        result_equity = fast_pricer.price_tranche(equity, maturity=5.0)
        result_senior = fast_pricer.price_tranche(senior, maturity=5.0)
        assert result_equity.par_spread > result_senior.par_spread

    def test_flat_correlation_override_changes_pricing(self, fast_pricer: CDOPricer) -> None:
        tranche = Tranche(attachment=0.0, detachment=1.2)
        n = fast_pricer.portfolio.n_obligors
        result_low_rho = fast_pricer.price_tranche(tranche, maturity=5.0, correlations=np.full(n, 0.01))
        result_high_rho = fast_pricer.price_tranche(tranche, maturity=5.0, correlations=np.full(n, 0.9))
        # Increasing correlation on an equity-like base tranche reduces its
        # expected loss (and hence par spread), since higher correlation
        # redistributes probability mass away from moderate joint default
        # counts toward the tails.
        assert result_high_rho.par_spread < result_low_rho.par_spread

    def test_rejects_non_positive_maturity(self, fast_pricer: CDOPricer) -> None:
        tranche = Tranche(attachment=0.0, detachment=1.2)
        with pytest.raises(ValueError):
            fast_pricer.price_tranche(tranche, maturity=0.0)

    def test_expected_tranche_loss_stays_within_bounds_over_time(self, fast_pricer: CDOPricer) -> None:
        tranche = Tranche(attachment=0.0, detachment=1.2)
        times = np.linspace(0.0, 5.0, 20)
        etl = np.array([fast_pricer.expected_tranche_loss(t, tranche) for t in times])
        check = check_expected_tranche_loss_bounds(etl, tranche.notional)
        assert check.passed

    def test_quadrature_convergence_for_a_realistic_tranche(self, fast_pricer: CDOPricer) -> None:
        import dataclasses

        tranche = Tranche(attachment=0.0, detachment=1.2)
        values_by_node_count = {
            n: dataclasses.replace(fast_pricer, n_quadrature_points=n).expected_tranche_loss(3.0, tranche)
            for n in (8, 16, 32)
        }
        result = assess_quadrature_convergence(values_by_node_count, tolerance=1e-4)
        assert result.converged
