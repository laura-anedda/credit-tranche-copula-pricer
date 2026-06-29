"""Tests for credit_copula.portfolio."""

from __future__ import annotations

import numpy as np
import pytest

from credit_copula.market_data import CreditCurve
from credit_copula.portfolio import (
    CreditPortfolio,
    Obligor,
    conditional_loss_distribution,
    loss_distribution,
    vasicek_expected_loss_above_threshold,
)


class TestObligor:
    def test_loss_given_default(self, flat_credit_curve: CreditCurve) -> None:
        obligor = Obligor("A", notional=10.0, recovery_rate=0.4, correlation=0.3, credit_curve=flat_credit_curve)
        assert obligor.loss_given_default == pytest.approx(6.0)

    def test_rejects_non_positive_notional(self, flat_credit_curve: CreditCurve) -> None:
        with pytest.raises(ValueError):
            Obligor("A", notional=0.0, recovery_rate=0.4, correlation=0.3, credit_curve=flat_credit_curve)

    def test_rejects_invalid_correlation(self, flat_credit_curve: CreditCurve) -> None:
        with pytest.raises(ValueError):
            Obligor("A", notional=1.0, recovery_rate=0.4, correlation=1.0, credit_curve=flat_credit_curve)


class TestCreditPortfolio:
    def test_total_notional(self, small_homogeneous_portfolio: CreditPortfolio) -> None:
        assert small_homogeneous_portfolio.total_notional == pytest.approx(10.0)

    def test_rejects_empty_portfolio(self) -> None:
        with pytest.raises(ValueError):
            CreditPortfolio(obligors=())


class TestConditionalLossDistribution:
    def test_sums_to_one(self) -> None:
        n = 8
        default_probabilities = np.full(n, 0.1)
        correlations = np.full(n, 0.3)
        lgd_buckets = np.full(n, 1)
        dist = conditional_loss_distribution(default_probabilities, correlations, 0.0, lgd_buckets, n + 1)
        assert np.sum(dist) == pytest.approx(1.0, abs=1e-10)

    def test_zero_default_probability_concentrates_mass_at_zero_loss(self) -> None:
        n = 5
        default_probabilities = np.full(n, 1e-12)
        correlations = np.full(n, 0.2)
        lgd_buckets = np.full(n, 1)
        dist = conditional_loss_distribution(default_probabilities, correlations, 0.0, lgd_buckets, n + 1)
        assert dist[0] == pytest.approx(1.0, abs=1e-9)

    def test_extreme_negative_systemic_factor_drives_near_certain_default(self) -> None:
        # A very negative systemic factor realization should push conditional
        # default probabilities toward 1 for positively correlated obligors,
        # concentrating loss mass at the maximum loss bucket.
        n = 4
        default_probabilities = np.full(n, 0.1)
        correlations = np.full(n, 0.5)
        lgd_buckets = np.full(n, 1)
        dist = conditional_loss_distribution(default_probabilities, correlations, -10.0, lgd_buckets, n + 1)
        assert dist[-1] == pytest.approx(1.0, abs=1e-6)


class TestLossDistribution:
    def test_sums_to_one(self) -> None:
        n = 10
        default_probabilities = np.full(n, 0.08)
        correlations = np.full(n, 0.3)
        lgd = np.full(n, 0.6)
        loss_unit = 0.6
        n_buckets = n + 1
        dist = loss_distribution(default_probabilities, correlations, lgd, loss_unit, n_buckets, n_quadrature_points=24)
        assert np.sum(dist) == pytest.approx(1.0, abs=1e-8)

    def test_expected_loss_matches_sum_of_marginal_default_probabilities(self) -> None:
        # E[portfolio loss] = sum_i LGD_i * Q_i^D(t), independent of correlation,
        # since expectation is linear and unaffected by dependence structure.
        n = 6
        default_probabilities = np.full(n, 0.05)
        correlations = np.full(n, 0.6)
        lgd = np.full(n, 1.0)
        loss_unit = 1.0
        n_buckets = n + 1
        dist = loss_distribution(default_probabilities, correlations, lgd, loss_unit, n_buckets, n_quadrature_points=32)
        loss_grid = np.arange(n_buckets) * loss_unit
        expected_loss = np.sum(dist * loss_grid)
        assert expected_loss == pytest.approx(n * 1.0 * 0.05, abs=1e-3)


class TestVasicekExpectedLossAboveThreshold:
    def test_threshold_zero_recovers_default_probability(self) -> None:
        p = 0.1
        result = vasicek_expected_loss_above_threshold(p, 0.3, 0.0)
        assert result == pytest.approx(p)

    def test_threshold_one_gives_zero(self) -> None:
        result = vasicek_expected_loss_above_threshold(0.1, 0.3, 1.0)
        assert result == pytest.approx(0.0)

    def test_decreasing_in_threshold(self) -> None:
        p, rho = 0.15, 0.4
        thresholds = np.linspace(0.01, 0.99, 20)
        values = [vasicek_expected_loss_above_threshold(p, rho, k) for k in thresholds]
        assert np.all(np.diff(values) <= 1e-12)

    def test_bounded_by_default_probability(self) -> None:
        p, rho = 0.2, 0.5
        for k in np.linspace(0.0, 1.0, 10):
            value = vasicek_expected_loss_above_threshold(p, rho, k)
            assert 0.0 <= value <= p + 1e-9

    def test_rejects_out_of_range_correlation(self) -> None:
        with pytest.raises(ValueError):
            vasicek_expected_loss_above_threshold(0.1, 1.0, 0.5)
