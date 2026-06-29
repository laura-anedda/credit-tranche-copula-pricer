"""Tests for credit_copula.numerics."""

from __future__ import annotations

import numpy as np
import pytest

from credit_copula.numerics import RootDiagnostics, gauss_hermite_nodes_weights, solve_root_brent


class TestGaussHermiteNodesWeights:
    def test_weights_sum_to_one(self) -> None:
        _, weights = gauss_hermite_nodes_weights(32)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-12)

    def test_integrates_constant_function_exactly(self) -> None:
        nodes, weights = gauss_hermite_nodes_weights(16)
        integral = np.sum(weights * np.ones_like(nodes) * 3.0)
        assert integral == pytest.approx(3.0)

    def test_integrates_mean_of_standard_normal_to_zero(self) -> None:
        nodes, weights = gauss_hermite_nodes_weights(32)
        integral = np.sum(weights * nodes)
        assert integral == pytest.approx(0.0, abs=1e-12)

    def test_integrates_variance_of_standard_normal_to_one(self) -> None:
        nodes, weights = gauss_hermite_nodes_weights(32)
        integral = np.sum(weights * nodes**2)
        assert integral == pytest.approx(1.0, rel=1e-10)

    def test_rejects_non_positive_n_points(self) -> None:
        with pytest.raises(ValueError):
            gauss_hermite_nodes_weights(0)


class TestSolveRootBrent:
    def test_finds_known_root(self) -> None:
        root = solve_root_brent(lambda x: x**2 - 2.0, 0.0, 2.0)
        assert root == pytest.approx(np.sqrt(2.0))

    def test_raises_when_root_not_bracketed(self) -> None:
        with pytest.raises(ValueError):
            solve_root_brent(lambda x: x**2 + 1.0, -1.0, 1.0)

    def test_returns_endpoint_when_exactly_zero(self) -> None:
        root = solve_root_brent(lambda x: x - 1.0, 1.0, 2.0)
        assert root == pytest.approx(1.0)

    def test_full_output_returns_root_diagnostics(self) -> None:
        result = solve_root_brent(lambda x: x**2 - 2.0, 0.0, 2.0, full_output=True)
        assert isinstance(result, RootDiagnostics)
        assert result.root == pytest.approx(np.sqrt(2.0))
        assert result.converged is True
        assert result.iterations > 0
        assert result.residual == pytest.approx(0.0, abs=1e-8)

    def test_full_output_at_exact_endpoint_root(self) -> None:
        result = solve_root_brent(lambda x: x - 1.0, 1.0, 2.0, full_output=True)
        assert isinstance(result, RootDiagnostics)
        assert result.root == pytest.approx(1.0)
        assert result.converged is True
        assert result.iterations == 0
