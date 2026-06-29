"""Integration tests for credit_copula.base_correlation."""

from __future__ import annotations

import numpy as np
import pytest

from credit_copula.base_correlation import (
    bootstrap_base_correlation,
    bootstrap_base_correlation_from_standard_tranches,
)
from credit_copula.diagnostics import check_correlation_bounds
from credit_copula.market_data import DiscountCurve
from credit_copula.portfolio import CreditPortfolio
from credit_copula.pricer import CDOPricer
from credit_copula.tranche import Tranche


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


class TestBootstrapBaseCorrelation:
    def test_bootstrapped_correlation_reprices_input_spread(self, fast_pricer: CDOPricer) -> None:
        # Generate self-consistent "market" quotes by pricing at a known flat
        # correlation, then confirm the bootstrap recovers that correlation.
        true_rho = 0.4
        n = fast_pricer.portfolio.n_obligors
        detachment = 2.4
        tranche = Tranche(attachment=0.0, detachment=detachment)
        reference_result = fast_pricer.price_tranche(
            tranche, maturity=5.0, correlations=np.full(n, true_rho)
        )

        curve = bootstrap_base_correlation(
            fast_pricer,
            detachment_points=np.array([detachment]),
            market_par_spreads=np.array([reference_result.par_spread]),
            maturity=5.0,
        )

        assert curve.correlations[0] == pytest.approx(true_rho, abs=1e-4)
        assert check_correlation_bounds(curve.correlations).passed

    def test_rejects_non_increasing_detachment_points(self, fast_pricer: CDOPricer) -> None:
        with pytest.raises(ValueError):
            bootstrap_base_correlation(
                fast_pricer,
                detachment_points=np.array([3.0, 2.0]),
                market_par_spreads=np.array([0.05, 0.02]),
                maturity=5.0,
            )

    def test_interpolate_returns_exact_pillar_value(self, fast_pricer: CDOPricer) -> None:
        n = fast_pricer.portfolio.n_obligors
        tranche = Tranche(attachment=0.0, detachment=2.4)
        reference_result = fast_pricer.price_tranche(tranche, maturity=5.0, correlations=np.full(n, 0.35))

        curve = bootstrap_base_correlation(
            fast_pricer,
            detachment_points=np.array([2.4]),
            market_par_spreads=np.array([reference_result.par_spread]),
            maturity=5.0,
        )
        assert curve.interpolate(2.4) == pytest.approx(curve.correlations[0])

    def _bootstrap_two_pillar_curve(self, fast_pricer: CDOPricer):
        # Derive achievable "market" quotes from the model itself at two
        # distinct flat correlations, rather than inventing arbitrary
        # target spreads that may not be reachable at any correlation in
        # the search bracket.
        n = fast_pricer.portfolio.n_obligors
        detachments = np.array([1.2, 3.6])
        true_correlations = [0.2, 0.5]
        market_spreads = np.array(
            [
                fast_pricer.price_tranche(
                    Tranche(0.0, float(d)), maturity=5.0, correlations=np.full(n, rho)
                ).par_spread
                for d, rho in zip(detachments, true_correlations)
            ]
        )
        return bootstrap_base_correlation(fast_pricer, detachments, market_spreads, maturity=5.0)

    def test_interpolate_between_pillars_is_linear(self, fast_pricer: CDOPricer) -> None:
        curve = self._bootstrap_two_pillar_curve(fast_pricer)
        midpoint = curve.interpolate(2.4)
        expected = 0.5 * (curve.correlations[0] + curve.correlations[1])
        assert midpoint == pytest.approx(expected)

    def test_interpolate_extrapolates_flat_beyond_range(self, fast_pricer: CDOPricer) -> None:
        curve = self._bootstrap_two_pillar_curve(fast_pricer)
        assert curve.interpolate(10.0) == pytest.approx(curve.correlations[-1])
        assert curve.interpolate(0.0) == pytest.approx(curve.correlations[0])


class TestBootstrapBaseCorrelationFromStandardTranches:
    def test_recovers_flat_correlation_across_contiguous_pillars(self, fast_pricer: CDOPricer) -> None:
        true_rho = 0.35
        n = fast_pricer.portfolio.n_obligors
        pillars = [(0.0, 1.2), (1.2, 2.4), (2.4, 4.8)]
        market_spreads = [
            fast_pricer.price_tranche(
                Tranche(a, d), maturity=5.0, correlations=np.full(n, true_rho)
            ).par_spread
            for a, d in pillars
        ]

        curve, diagnostics = bootstrap_base_correlation_from_standard_tranches(
            fast_pricer, pillars, market_spreads, maturity=5.0
        )

        np.testing.assert_allclose(curve.correlations, true_rho, atol=1e-4)
        assert np.all(diagnostics.converged)
        assert np.all(np.abs(diagnostics.residual) < 1e-6)

    def test_rejects_non_contiguous_pillars(self, fast_pricer: CDOPricer) -> None:
        with pytest.raises(ValueError):
            bootstrap_base_correlation_from_standard_tranches(
                fast_pricer,
                tranche_pillars=[(0.0, 1.2), (1.5, 2.4)],
                market_par_spreads=[0.05, 0.02],
                maturity=5.0,
            )

    def test_rejects_non_zero_first_attachment(self, fast_pricer: CDOPricer) -> None:
        with pytest.raises(ValueError):
            bootstrap_base_correlation_from_standard_tranches(
                fast_pricer,
                tranche_pillars=[(0.6, 1.2)],
                market_par_spreads=[0.05],
                maturity=5.0,
            )

    def test_first_pillar_matches_direct_base_tranche_bootstrap(self, fast_pricer: CDOPricer) -> None:
        # For the most subordinated pillar, the standard-tranche stripping
        # algorithm reduces to the direct base-tranche bootstrap, since the
        # tranche [0, D_1] is itself a base tranche with no prior pillar to
        # subtract.
        n = fast_pricer.portfolio.n_obligors
        detachment = 1.2
        reference_result = fast_pricer.price_tranche(
            Tranche(0.0, detachment), maturity=5.0, correlations=np.full(n, 0.3)
        )

        direct_curve = bootstrap_base_correlation(
            fast_pricer,
            detachment_points=np.array([detachment]),
            market_par_spreads=np.array([reference_result.par_spread]),
            maturity=5.0,
        )
        stripped_curve, _ = bootstrap_base_correlation_from_standard_tranches(
            fast_pricer, [(0.0, detachment)], [reference_result.par_spread], maturity=5.0
        )
        assert stripped_curve.correlations[0] == pytest.approx(direct_curve.correlations[0], abs=1e-6)
