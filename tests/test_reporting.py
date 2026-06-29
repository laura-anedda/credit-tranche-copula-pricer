"""Tests for credit_copula.reporting."""

from __future__ import annotations

import numpy as np
import pytest

from credit_copula.market_data import CreditCurve, DiscountCurve
from credit_copula.portfolio import CreditPortfolio
from credit_copula.pricer import CDOPricer
from credit_copula.reporting import (
    base_correlation_table,
    base_tranche_expected_loss_curve,
    build_pricer,
    calibration_table,
    correlation_sensitivity,
    expected_tranche_loss_curves,
    hazard_rate_sensitivity,
    price_tranche_structure,
    recovery_rate_sensitivity,
)


@pytest.fixture
def fast_pricer(small_homogeneous_portfolio: CreditPortfolio, flat_discount_curve: DiscountCurve) -> CDOPricer:
    return CDOPricer(
        portfolio=small_homogeneous_portfolio,
        discount_curve=flat_discount_curve,
        loss_unit=0.6,
        n_quadrature_points=16,
        payment_frequency=4,
        n_integration_steps_per_year=4,
    )


class TestPriceTrancheStructure:
    def test_returns_one_row_per_tranche(self, fast_pricer: CDOPricer) -> None:
        pillars = [(0.0, 0.12), (0.12, 0.24), (0.24, 0.60)]
        df = price_tranche_structure(fast_pricer, pillars, maturity=5.0)
        assert len(df) == 3
        assert list(df["tranche"]) == ["0%-12%", "12%-24%", "24%-60%"]

    def test_subordinated_tranche_has_wider_spread(self, fast_pricer: CDOPricer) -> None:
        pillars = [(0.0, 0.12), (0.4, 0.6)]
        df = price_tranche_structure(fast_pricer, pillars, maturity=5.0)
        assert df.loc[0, "par_spread_bps"] > df.loc[1, "par_spread_bps"]

    def test_rejects_invalid_pillar_pair(self, fast_pricer: CDOPricer) -> None:
        with pytest.raises(ValueError):
            price_tranche_structure(fast_pricer, [(0.5, 0.2)], maturity=5.0)


class TestExpectedTrancheLossCurves:
    def test_long_format_shape(self, fast_pricer: CDOPricer) -> None:
        pillars = [(0.0, 0.12), (0.12, 0.24)]
        df = expected_tranche_loss_curves(fast_pricer, pillars, maturity=5.0, n_points=10)
        assert set(df["tranche"]) == {"0%-12%", "12%-24%"}
        assert len(df) == 2 * 11  # n_points + 1 time points per tranche

    def test_expected_loss_starts_at_zero(self, fast_pricer: CDOPricer) -> None:
        pillars = [(0.0, 0.12)]
        df = expected_tranche_loss_curves(fast_pricer, pillars, maturity=5.0, n_points=10)
        first_row = df.iloc[0]
        assert first_row["time"] == pytest.approx(0.0)
        assert first_row["expected_loss_pct"] == pytest.approx(0.0)

    def test_expected_loss_fraction_is_bounded(self, fast_pricer: CDOPricer) -> None:
        pillars = [(0.0, 0.12), (0.4, 0.6)]
        df = expected_tranche_loss_curves(fast_pricer, pillars, maturity=5.0, n_points=10)
        assert (df["expected_loss_pct"] >= -1e-9).all()
        assert (df["expected_loss_pct"] <= 1.0 + 1e-9).all()


class TestBaseCorrelationTable:
    def test_recovers_flat_correlation(self, fast_pricer: CDOPricer) -> None:
        # small_homogeneous_portfolio fixture uses a uniform correlation of
        # 0.3 for every obligor; the implied base correlation skew should
        # therefore be flat at 0.3 across all detachment points strictly
        # below full portfolio loss capacity. A detachment point at 100%
        # of total notional (0.6 here, since this fixture's 10 obligors
        # each have LGD=0.6) is deliberately excluded: the base tranche
        # [0, 100%] spans the entire portfolio, whose expected loss is
        # LGD * PD by linearity of expectation -- independent of
        # correlation, which only reshapes the loss distribution, not its
        # mean. Brent's method correctly has no unique root to find there,
        # since the par spread does not vary with correlation at all.
        df = base_correlation_table(fast_pricer, detachment_points_pct=[0.24, 0.36, 0.48], maturity=5.0)
        np.testing.assert_allclose(df["base_correlation"].to_numpy(), 0.3, atol=1e-4)


class TestBuildPricer:
    def test_constructs_homogeneous_portfolio_of_requested_size(self, flat_discount_curve: DiscountCurve) -> None:
        pricer = build_pricer(
            cds_tenors=np.array([1.0, 5.0, 10.0]),
            cds_spreads=np.array([0.005, 0.009, 0.012]),
            recovery_rate=0.4,
            discount_curve=flat_discount_curve,
            n_obligors=15,
            correlation=0.3,
            loss_unit=0.6,
            n_quadrature_points=16,
        )
        assert pricer.portfolio.n_obligors == 15
        assert pricer.portfolio.total_notional == pytest.approx(15.0)


class TestBaseTrancheExpectedLossCurve:
    def test_non_decreasing_and_bounded_by_expected_portfolio_loss(self, fast_pricer: CDOPricer) -> None:
        df = base_tranche_expected_loss_curve(fast_pricer, [0.1, 0.3, 0.6], maturity=5.0)
        losses = df["expected_loss"].to_numpy()
        assert np.all(np.diff(losses) >= -1e-9)
        default_probabilities = fast_pricer.portfolio.default_probabilities(5.0)
        lgd = fast_pricer.portfolio.loss_given_defaults()
        expected_portfolio_loss = float(np.sum(default_probabilities * lgd))
        assert losses[-1] <= expected_portfolio_loss + 1e-6


class TestCorrelationSensitivity:
    def test_long_format_shape_and_columns(self, fast_pricer: CDOPricer) -> None:
        pillars = [(0.0, 0.12), (0.4, 0.6)]
        grid = np.array([0.1, 0.3, 0.5])
        df = correlation_sensitivity(fast_pricer, pillars, maturity=5.0, correlation_grid=grid)
        assert len(df) == len(pillars) * len(grid)
        assert set(df.columns) == {"correlation", "tranche", "par_spread_bps"}

    def test_equity_tranche_spread_decreases_with_correlation(self, fast_pricer: CDOPricer) -> None:
        pillars = [(0.0, 0.12)]
        grid = np.array([0.05, 0.5, 0.9])
        df = correlation_sensitivity(fast_pricer, pillars, maturity=5.0, correlation_grid=grid)
        spreads = df.sort_values("correlation")["par_spread_bps"].to_numpy()
        assert np.all(np.diff(spreads) <= 0.0)


class TestRecoveryRateSensitivity:
    def test_long_format_shape(self, flat_discount_curve: DiscountCurve) -> None:
        pillars = [(0.0, 0.3)]
        df = recovery_rate_sensitivity(
            cds_tenors=np.array([1.0, 5.0, 10.0]),
            cds_spreads=np.array([0.005, 0.009, 0.012]),
            discount_curve=flat_discount_curve,
            n_obligors=10,
            correlation=0.3,
            loss_unit=0.6,
            tranche_pillars_pct=pillars,
            maturity=5.0,
            recovery_grid=np.array([0.2, 0.4, 0.6]),
            n_quadrature_points=16,
        )
        assert len(df) == 3
        assert set(df.columns) == {"recovery_rate", "tranche", "par_spread_bps"}

    def test_recovery_rate_change_produces_a_different_spread(self, flat_discount_curve: DiscountCurve) -> None:
        # Recovery rate sensitivity is not guaranteed to be monotonic here:
        # since the credit curve is re-bootstrapped from the *same* market
        # CDS spreads at each trial recovery, a higher assumed recovery
        # forces the bootstrap to imply a correspondingly higher hazard
        # rate to match the same quote, and the two effects act in
        # opposite directions on tranche expected loss. This test only
        # confirms the sweep is non-degenerate (sensitive to its input).
        pillars = [(0.0, 0.3)]
        df = recovery_rate_sensitivity(
            cds_tenors=np.array([1.0, 5.0, 10.0]),
            cds_spreads=np.array([0.005, 0.009, 0.012]),
            discount_curve=flat_discount_curve,
            n_obligors=10,
            correlation=0.3,
            loss_unit=0.6,
            tranche_pillars_pct=pillars,
            maturity=5.0,
            recovery_grid=np.array([0.2, 0.6]),
            n_quadrature_points=16,
        )
        spreads = df.sort_values("recovery_rate")["par_spread_bps"].to_numpy()
        assert spreads[0] != pytest.approx(spreads[1])


class TestHazardRateSensitivity:
    def test_higher_multiplier_increases_spread(self, fast_pricer: CDOPricer) -> None:
        pillars = [(0.0, 0.3)]
        grid = np.array([0.5, 1.0, 1.5])
        df = hazard_rate_sensitivity(fast_pricer, pillars, maturity=5.0, multiplier_grid=grid)
        spreads = df.sort_values("hazard_rate_multiplier")["par_spread_bps"].to_numpy()
        assert np.all(np.diff(spreads) > 0.0)

    def test_multiplier_one_reproduces_base_pricing(self, fast_pricer: CDOPricer) -> None:
        pillars = [(0.0, 0.3)]
        df = hazard_rate_sensitivity(fast_pricer, pillars, maturity=5.0, multiplier_grid=np.array([1.0]))
        reference = price_tranche_structure(fast_pricer, pillars, maturity=5.0)
        assert df["par_spread_bps"].iloc[0] == pytest.approx(reference["par_spread_bps"].iloc[0], rel=1e-9)


class TestCalibrationTable:
    def test_table_matches_residuals_object(self) -> None:
        table, residuals = calibration_table(["A", "B"], [100.0, 50.0], [105.0, 48.0])
        assert list(table["tranche"]) == ["A", "B"]
        np.testing.assert_allclose(table["absolute_error_bps"].to_numpy(), residuals.absolute_error_bps)
