"""Tests for credit_copula.diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from credit_copula.diagnostics import (
    CalibrationDiagnostics,
    assess_quadrature_convergence,
    base_correlation_curvature,
    check_base_correlation_monotonicity,
    check_base_tranche_convexity,
    check_correlation_bounds,
    check_expected_loss_time_monotonicity,
    check_expected_tranche_loss_bounds,
    check_probabilities_non_negative,
    check_probability_mass_conservation,
    compute_pricing_residuals,
    estimate_loss_discretization_error,
    generate_calibration_warnings,
    maximum_absolute_error,
    root_mean_square_error,
    summarize_calibration,
)


class TestPricingResiduals:
    def test_zero_residuals_when_market_equals_model(self) -> None:
        residuals = compute_pricing_residuals(["A", "B"], [100.0, 50.0], [100.0, 50.0])
        np.testing.assert_allclose(residuals.absolute_error_bps, 0.0)
        np.testing.assert_allclose(residuals.relative_error_pct, 0.0)

    def test_relative_error_matches_definition(self) -> None:
        residuals = compute_pricing_residuals(["A"], [100.0], [110.0])
        assert residuals.absolute_error_bps[0] == pytest.approx(10.0)
        assert residuals.relative_error_pct[0] == pytest.approx(10.0)

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError):
            compute_pricing_residuals(["A", "B"], [100.0], [100.0, 110.0])


class TestErrorSummaryStatistics:
    def test_rmse_of_constant_residual(self) -> None:
        assert root_mean_square_error(np.array([3.0, 3.0, 3.0])) == pytest.approx(3.0)

    def test_rmse_zero_for_perfect_fit(self) -> None:
        assert root_mean_square_error(np.zeros(5)) == pytest.approx(0.0)

    def test_max_absolute_error(self) -> None:
        assert maximum_absolute_error(np.array([-2.0, 5.0, -7.0])) == pytest.approx(7.0)


class TestExpectedLossTimeMonotonicity:
    def test_passes_for_non_decreasing_curve(self) -> None:
        times = np.array([0.0, 1.0, 2.0])
        etl = np.array([0.0, 0.1, 0.25])
        result = check_expected_loss_time_monotonicity(times, etl)
        assert result.passed
        assert result.violation_indices.size == 0

    def test_fails_for_decreasing_curve(self) -> None:
        times = np.array([0.0, 1.0, 2.0])
        etl = np.array([0.0, 0.2, 0.1])
        result = check_expected_loss_time_monotonicity(times, etl)
        assert not result.passed
        assert result.violation_indices.size == 1

    def test_tolerates_negligible_noise(self) -> None:
        times = np.array([0.0, 1.0, 2.0])
        etl = np.array([0.0, 0.1, 0.1 - 1e-12])
        result = check_expected_loss_time_monotonicity(times, etl, tolerance=1e-8)
        assert result.passed

    def test_rejects_non_increasing_times(self) -> None:
        times = np.array([0.0, 2.0, 1.0])
        etl = np.array([0.0, 0.1, 0.2])
        with pytest.raises(ValueError):
            check_expected_loss_time_monotonicity(times, etl)


class TestBaseTrancheConvexity:
    def test_passes_for_concave_increasing_curve(self) -> None:
        # E[min(L,K)] = E[L] * (1 - exp(-K)) for an exponential-like loss
        # tail is non-decreasing and concave in K, matching the required
        # shape of a base tranche expected-loss curve.
        detachments = np.array([1.0, 2.0, 3.0, 4.0])
        expected_losses = 1.0 - np.exp(-detachments)
        result = check_base_tranche_convexity(detachments, expected_losses)
        assert result.passed

    def test_fails_for_convex_increasing_curve(self) -> None:
        # Monotonically increasing but convex (second derivative > 0):
        # violates the required concavity of E[min(L,K)].
        detachments = np.array([1.0, 2.0, 3.0, 4.0])
        expected_losses = np.exp(detachments)
        result = check_base_tranche_convexity(detachments, expected_losses)
        assert not result.passed

    def test_fails_for_non_monotonic_curve(self) -> None:
        detachments = np.array([1.0, 2.0, 3.0, 4.0])
        expected_losses = -((detachments - 2.5) ** 2)  # decreasing on part of the domain
        result = check_base_tranche_convexity(detachments, expected_losses)
        assert not result.passed


class TestBaseCorrelationMonotonicity:
    def test_passes_for_increasing_skew(self) -> None:
        result = check_base_correlation_monotonicity(
            np.array([1.0, 2.0, 3.0]), np.array([0.2, 0.3, 0.4])
        )
        assert result.passed

    def test_fails_for_non_monotonic_skew(self) -> None:
        result = check_base_correlation_monotonicity(
            np.array([1.0, 2.0, 3.0]), np.array([0.3, 0.2, 0.4])
        )
        assert not result.passed


class TestBaseCorrelationCurvature:
    def test_zero_for_linear_curve(self) -> None:
        detachments = np.array([1.0, 2.0, 3.0, 4.0])
        correlations = 0.1 + 0.05 * detachments
        curvature = base_correlation_curvature(detachments, correlations)
        assert curvature == pytest.approx(0.0, abs=1e-10)

    def test_positive_for_erratic_curve(self) -> None:
        detachments = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        correlations = np.array([0.2, 0.5, 0.15, 0.55, 0.1])
        curvature = base_correlation_curvature(detachments, correlations)
        assert curvature > 0.0

    def test_returns_zero_for_fewer_than_three_pillars(self) -> None:
        assert base_correlation_curvature(np.array([1.0, 2.0]), np.array([0.2, 0.3])) == 0.0


class TestGenerateCalibrationWarnings:
    def test_no_warnings_for_clean_calibration(self) -> None:
        residuals = compute_pricing_residuals(["A", "B"], [100.0, 50.0], [101.0, 49.5])
        warnings = generate_calibration_warnings(residuals)
        assert warnings == []

    def test_warns_on_high_rmse(self) -> None:
        residuals = compute_pricing_residuals(["A", "B"], [100.0, 50.0], [150.0, 90.0])
        warnings = generate_calibration_warnings(residuals, rmse_threshold_bps=5.0)
        assert any("RMSE" in w for w in warnings)

    def test_warns_on_convexity_violation(self) -> None:
        residuals = compute_pricing_residuals(["A"], [100.0], [100.0])
        convexity_check = check_base_tranche_convexity(
            np.array([1.0, 2.0, 3.0]), np.exp(np.array([1.0, 2.0, 3.0]))
        )
        warnings = generate_calibration_warnings(residuals, base_tranche_convexity_check=convexity_check)
        assert any("arbitrage" in w for w in warnings)


class TestSummarizeCalibration:
    def test_reports_rmse_and_max_error(self) -> None:
        diagnostics = summarize_calibration(
            tranche_labels=["A", "B"],
            market_spreads_bps=np.array([100.0, 50.0]),
            model_spreads_bps=np.array([103.0, 49.0]),
            pillar_iterations=np.array([5, 7]),
            pillar_converged=np.array([True, True]),
            pillar_residuals=np.array([1e-12, 1e-13]),
            correlation_bounds=(1e-4, 0.999),
            root_tolerance=1e-10,
        )
        assert isinstance(diagnostics, CalibrationDiagnostics)
        assert diagnostics.rmse_bps == pytest.approx(root_mean_square_error(np.array([3.0, -1.0])))
        assert diagnostics.max_absolute_error_bps == pytest.approx(3.0)

    def test_is_reliable_when_converged_and_no_warnings(self) -> None:
        diagnostics = summarize_calibration(
            tranche_labels=["A"],
            market_spreads_bps=np.array([100.0]),
            model_spreads_bps=np.array([100.1]),
            pillar_iterations=np.array([5]),
            pillar_converged=np.array([True]),
            pillar_residuals=np.array([1e-12]),
            correlation_bounds=(1e-4, 0.999),
            root_tolerance=1e-10,
        )
        assert diagnostics.is_reliable
        assert diagnostics.all_pillars_converged

    def test_is_not_reliable_when_a_pillar_fails_to_converge(self) -> None:
        diagnostics = summarize_calibration(
            tranche_labels=["A", "B"],
            market_spreads_bps=np.array([100.0, 50.0]),
            model_spreads_bps=np.array([100.1, 50.1]),
            pillar_iterations=np.array([5, 200]),
            pillar_converged=np.array([True, False]),
            pillar_residuals=np.array([1e-12, 0.5]),
            correlation_bounds=(1e-4, 0.999),
            root_tolerance=1e-10,
        )
        assert not diagnostics.all_pillars_converged
        assert not diagnostics.is_reliable
        assert any("convergence" in w for w in diagnostics.warnings)

    def test_warnings_include_check_driven_messages(self) -> None:
        convexity_check = check_base_tranche_convexity(
            np.array([1.0, 2.0, 3.0]), np.exp(np.array([1.0, 2.0, 3.0]))
        )
        diagnostics = summarize_calibration(
            tranche_labels=["A"],
            market_spreads_bps=np.array([100.0]),
            model_spreads_bps=np.array([100.1]),
            pillar_iterations=np.array([5]),
            pillar_converged=np.array([True]),
            pillar_residuals=np.array([1e-12]),
            correlation_bounds=(1e-4, 0.999),
            root_tolerance=1e-10,
            base_tranche_convexity_check=convexity_check,
        )
        assert any("arbitrage" in w for w in diagnostics.warnings)
        assert not diagnostics.is_reliable


class TestProbabilityMassConservation:
    def test_passes_for_normalized_distribution(self) -> None:
        result = check_probability_mass_conservation(np.array([0.3, 0.3, 0.4]))
        assert result.passed

    def test_fails_for_unnormalized_distribution(self) -> None:
        result = check_probability_mass_conservation(np.array([0.3, 0.3, 0.3]), tolerance=1e-6)
        assert not result.passed
        assert result.max_violation == pytest.approx(-0.1)


class TestProbabilitiesNonNegative:
    def test_passes_for_non_negative_array(self) -> None:
        assert check_probabilities_non_negative(np.array([0.0, 0.1, 0.9])).passed

    def test_fails_for_negative_value(self) -> None:
        result = check_probabilities_non_negative(np.array([0.1, -0.01, 0.9]))
        assert not result.passed
        assert result.max_violation == pytest.approx(0.01)

    def test_tolerates_negligible_negative_noise(self) -> None:
        result = check_probabilities_non_negative(np.array([0.1, -1e-13, 0.9]))
        assert result.passed


class TestExpectedTrancheLossBounds:
    def test_passes_within_bounds(self) -> None:
        result = check_expected_tranche_loss_bounds(np.array([0.0, 0.5, 1.0]), tranche_notional=1.0)
        assert result.passed

    def test_fails_above_notional(self) -> None:
        result = check_expected_tranche_loss_bounds(np.array([0.5, 1.2]), tranche_notional=1.0)
        assert not result.passed
        assert result.max_violation == pytest.approx(0.2)

    def test_fails_below_zero(self) -> None:
        result = check_expected_tranche_loss_bounds(np.array([-0.1, 0.5]), tranche_notional=1.0)
        assert not result.passed
        assert result.max_violation == pytest.approx(0.1)


class TestCorrelationBounds:
    def test_passes_within_unit_interval(self) -> None:
        assert check_correlation_bounds(np.array([0.1, 0.5, 0.99])).passed

    def test_fails_above_one(self) -> None:
        result = check_correlation_bounds(np.array([0.5, 1.1]))
        assert not result.passed
        assert result.max_violation == pytest.approx(0.1)

    def test_fails_below_zero(self) -> None:
        result = check_correlation_bounds(np.array([-0.05, 0.5]))
        assert not result.passed
        assert result.max_violation == pytest.approx(0.05)


class TestEstimateLossDiscretizationError:
    def test_zero_error_when_lgd_exactly_on_grid(self) -> None:
        result = estimate_loss_discretization_error(
            default_probabilities=np.array([0.1, 0.2]),
            loss_given_defaults=np.array([0.5, 1.0]),
            loss_unit=0.5,
        )
        assert result.absolute_error == pytest.approx(0.0, abs=1e-12)

    def test_nonzero_error_when_lgd_off_grid(self) -> None:
        result = estimate_loss_discretization_error(
            default_probabilities=np.array([0.1]),
            loss_given_defaults=np.array([0.6]),
            loss_unit=0.5,
        )
        # 0.6 rounds to the nearest multiple of 0.5, i.e. 0.5.
        assert result.discretized_expected_loss == pytest.approx(0.1 * 0.5)
        assert result.exact_expected_loss == pytest.approx(0.1 * 0.6)
        assert result.absolute_error < 0.0

    def test_relative_error_is_percentage_of_exact(self) -> None:
        result = estimate_loss_discretization_error(
            default_probabilities=np.array([1.0]),
            loss_given_defaults=np.array([0.6]),
            loss_unit=0.5,
        )
        assert result.relative_error_pct == pytest.approx(
            100.0 * result.absolute_error / result.exact_expected_loss
        )


class TestAssessQuadratureConvergence:
    def test_converged_when_last_difference_within_tolerance(self) -> None:
        result = assess_quadrature_convergence({16: 0.1000, 32: 0.1001, 64: 0.10011}, tolerance=1e-3)
        assert result.converged

    def test_not_converged_when_last_difference_exceeds_tolerance(self) -> None:
        result = assess_quadrature_convergence({16: 0.10, 32: 0.20}, tolerance=1e-6)
        assert not result.converged

    def test_node_counts_sorted_increasing(self) -> None:
        result = assess_quadrature_convergence({64: 0.3, 16: 0.1, 32: 0.2})
        np.testing.assert_array_equal(result.node_counts, [16, 32, 64])

    def test_rejects_fewer_than_two_node_counts(self) -> None:
        with pytest.raises(ValueError):
            assess_quadrature_convergence({16: 0.1})
