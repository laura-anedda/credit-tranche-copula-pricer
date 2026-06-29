"""Tests for credit_copula.copula."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from credit_copula.copula import conditional_default_probability, default_barrier


class TestDefaultBarrier:
    def test_matches_inverse_normal_cdf(self) -> None:
        q = 0.1
        assert default_barrier(q) == pytest.approx(norm.ppf(q))

    def test_rejects_default_probability_of_one(self) -> None:
        with pytest.raises(ValueError):
            default_barrier(1.0)

    def test_rejects_negative_default_probability(self) -> None:
        with pytest.raises(ValueError):
            default_barrier(-0.1)


class TestConditionalDefaultProbability:
    def test_recovers_marginal_probability_at_zero_correlation_limit(self) -> None:
        # With correlation -> 0, the conditional default probability should
        # converge to the unconditional one for any systemic factor value,
        # since idiosyncratic risk dominates entirely.
        q = 0.05
        cond = conditional_default_probability(q, 1e-9, systemic_factor=2.0)
        assert cond == pytest.approx(q, abs=1e-4)

    def test_decreasing_in_systemic_factor_for_positive_correlation(self) -> None:
        q = 0.1
        rho = 0.4
        z = np.linspace(-3.0, 3.0, 25)
        cond = conditional_default_probability(q, rho, z)
        assert np.all(np.diff(cond) < 0.0)

    def test_bounded_in_unit_interval(self) -> None:
        q = 0.2
        rho = 0.5
        z = np.linspace(-10.0, 10.0, 50)
        cond = conditional_default_probability(q, rho, z)
        assert np.all(cond >= 0.0) and np.all(cond <= 1.0)

    def test_integrates_back_to_marginal_probability(self) -> None:
        # E_Z[p(Z)] should equal the unconditional default probability,
        # by the tower property / construction of the copula barrier.
        from credit_copula.numerics import gauss_hermite_nodes_weights

        q = 0.15
        rho = 0.35
        nodes, weights = gauss_hermite_nodes_weights(48)
        cond = conditional_default_probability(q, rho, nodes)
        recovered = np.sum(weights * cond)
        assert recovered == pytest.approx(q, abs=1e-6)

    def test_rejects_correlation_of_one(self) -> None:
        with pytest.raises(ValueError):
            conditional_default_probability(0.1, 1.0, 0.0)

    def test_rejects_invalid_default_probability(self) -> None:
        with pytest.raises(ValueError):
            conditional_default_probability(1.5, 0.3, 0.0)
