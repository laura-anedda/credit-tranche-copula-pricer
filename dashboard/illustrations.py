"""
Monte Carlo sampling of the one-factor Gaussian copula latent-variable structure.

Draws joint realizations of the latent asset values
:math:`X_i = \\sqrt{\\rho}\\,Z + \\sqrt{1-\\rho}\\,\\varepsilon_i` defined in
:mod:`credit_copula.copula`, for direct visualization of the
systematic/idiosyncratic decomposition and the resulting pairwise
dependence structure. Tranche pricing itself is computed semi-
analytically via Gauss-Hermite quadrature over the systematic factor
(:mod:`credit_copula.portfolio`); the simulation here is a separate,
deterministic (seeded) sampling routine used only to render the
dependence-structure visualizations in the dashboard.
"""

from __future__ import annotations

import numpy as np

__all__ = ["simulate_latent_pair", "simulate_systemic_decomposition"]


def simulate_latent_pair(
    correlation: float, n_samples: int = 3000, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate paired latent asset values under the one-factor Gaussian copula.

    Draws `n_samples` realizations of two obligors' latent asset values

    .. math::
        X_i = \\sqrt{\\rho}\\, Z + \\sqrt{1-\\rho}\\, \\varepsilon_i, \\qquad
        X_j = \\sqrt{\\rho}\\, Z + \\sqrt{1-\\rho}\\, \\varepsilon_j

    with a shared systemic factor `Z` and independent idiosyncratic
    factors :math:`\\varepsilon_i, \\varepsilon_j`, all standard normal.
    By construction, :math:`\\text{Corr}(X_i, X_j) = \\rho`.

    Parameters
    ----------
    correlation : float
        Common one-factor loading :math:`\\rho \\in [0, 1)` applied to
        both obligors.
    n_samples : int, default=3000
        Number of joint draws.
    seed : int, default=7
        Seed for the NumPy random generator, for reproducibility.

    Returns
    -------
    x_i, x_j : np.ndarray
        Simulated latent asset values, shape ``(n_samples,)`` each.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_samples)
    eps_i = rng.standard_normal(n_samples)
    eps_j = rng.standard_normal(n_samples)
    sqrt_rho = np.sqrt(correlation)
    sqrt_one_minus_rho = np.sqrt(1.0 - correlation)
    x_i = sqrt_rho * z + sqrt_one_minus_rho * eps_i
    x_j = sqrt_rho * z + sqrt_one_minus_rho * eps_j
    return x_i, x_j


def simulate_systemic_decomposition(
    correlation: float, n_samples: int = 2000, seed: int = 7
) -> dict[str, np.ndarray]:
    """
    Simulate the systemic/idiosyncratic decomposition of a single obligor's latent variable.

    Parameters
    ----------
    correlation : float
        One-factor loading :math:`\\rho \\in [0, 1)`.
    n_samples : int, default=2000
        Number of draws.
    seed : int, default=7
        Random seed.

    Returns
    -------
    dict[str, np.ndarray]
        Keys ``"systemic"`` (:math:`\\sqrt{\\rho}\\, Z`), ``"idiosyncratic"``
        (:math:`\\sqrt{1-\\rho}\\, \\varepsilon`), and ``"latent"``
        (their sum, :math:`X`), each of shape ``(n_samples,)``.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_samples)
    eps = rng.standard_normal(n_samples)
    systemic = np.sqrt(correlation) * z
    idiosyncratic = np.sqrt(1.0 - correlation) * eps
    return {"systemic": systemic, "idiosyncratic": idiosyncratic, "latent": systemic + idiosyncratic}
