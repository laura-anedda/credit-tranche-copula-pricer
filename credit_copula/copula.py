"""
One-factor Gaussian copula dependence model (Li, 2000).

This module implements the latent asset-value representation of joint
default dependence used throughout the synthetic CDO and index tranche
market. Each obligor's default time is driven by a standardized
Gaussian asset-value process decomposed into a common systemic factor
and an idiosyncratic, obligor-specific factor:

.. math::
    X_i = \\sqrt{\\rho_i}\\, Z + \\sqrt{1 - \\rho_i}\\, \\varepsilon_i,
    \\qquad Z, \\varepsilon_i \\overset{\\text{i.i.d.}}{\\sim} \\mathcal{N}(0, 1)

Obligor `i` is modelled as having defaulted by time `t` if and only if
its asset value has fallen below a default barrier calibrated to match
the marginal (unconditional) default probability term structure implied
by the single-name CDS market:

.. math::
    \\{X_i \\le t\\} \\;:=\\; \\{X_i \\le K_i(t)\\}, \\qquad
    K_i(t) = \\Phi^{-1}\\!\\big(Q_i^{D}(t)\\big)

where :math:`Q_i^{D}(t) = 1 - Q_i(t)` is the cumulative default
probability and :math:`\\Phi` is the standard normal CDF. This barrier
construction guarantees, by the probability integral transform, that
the marginal default probability of obligor `i` exactly matches the
CDS-implied curve regardless of the assumed correlation, which is the
essential calibration property that makes the copula approach
practical for index tranche pricing.

"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["default_barrier", "conditional_default_probability"]


def default_barrier(default_probability: np.ndarray | float) -> np.ndarray:
    """
    Compute the Gaussian copula default barrier :math:`K = \\Phi^{-1}(Q^D)`.

    Parameters
    ----------
    default_probability : array_like or float
        Marginal (unconditional) cumulative default probability,
        :math:`Q^D(t) \\in [0, 1)`.

    Returns
    -------
    np.ndarray
        Default barrier(s) on the standard normal scale.

    Raises
    ------
    ValueError
        If any default probability lies outside :math:`[0, 1)`.

    Notes
    -----
    A default probability of exactly 1.0 maps to :math:`+\\infty` under
    the inverse normal CDF; this is disallowed here since it represents
    a degenerate (certain default) input that is not meaningful for
    finite-maturity CDS-implied curves.
    """
    q = np.asarray(default_probability, dtype=np.float64)
    if np.any(q < 0.0) or np.any(q >= 1.0):
        raise ValueError("default_probability must lie in [0, 1)")
    return norm.ppf(q)


def conditional_default_probability(
    default_probability: np.ndarray,
    correlation: np.ndarray | float,
    systemic_factor: np.ndarray | float,
) -> np.ndarray:
    """
    Conditional default probability given a realization of the systemic factor.

    Under the one-factor Gaussian copula, conditioning on a realization
    `z` of the common factor `Z` renders all obligors' default
    indicators independent. The conditional default probability is

    .. math::
        p_i(t \\mid Z = z) = \\Phi\\!\\left(
            \\frac{\\Phi^{-1}\\big(Q_i^{D}(t)\\big) - \\sqrt{\\rho_i}\\, z}
                 {\\sqrt{1 - \\rho_i}}
        \\right)

    This conditional independence is the key analytical device that
    makes the portfolio loss distribution tractable: integrating the
    (conditionally independent) portfolio loss distribution over the
    distribution of `Z` recovers the full, dependent joint loss
    distribution (see :mod:`credit_copula.portfolio`).

    Parameters
    ----------
    default_probability : np.ndarray
        Marginal cumulative default probability(ies)
        :math:`Q_i^{D}(t) \\in [0, 1)`, broadcastable against
        `correlation`.
    correlation : np.ndarray or float
        Factor loading(s) :math:`\\rho_i \\in [0, 1)`, i.e. the squared
        correlation of obligor `i`'s asset value with the systemic
        factor.
    systemic_factor : np.ndarray or float
        Realization(s) `z` of the standard normal systemic factor `Z`.

    Returns
    -------
    np.ndarray
        Conditional default probability, broadcast over the input
        shapes.

    Raises
    ------
    ValueError
        If `correlation` lies outside :math:`[0, 1)` or
        `default_probability` lies outside :math:`[0, 1)`.

    Notes
    -----
    A correlation of exactly 1.0 would make the denominator
    :math:`\\sqrt{1-\\rho_i}` vanish, collapsing the obligor's default
    indicator onto a deterministic function of `Z` alone; this
    degenerate case is excluded by construction. Numerically, the
    formula is evaluated directly without further safeguards since
    `correlation` is bounded away from 1 by validation.
    """
    q = np.asarray(default_probability, dtype=np.float64)
    rho = np.asarray(correlation, dtype=np.float64)
    z = np.asarray(systemic_factor, dtype=np.float64)

    if np.any(q < 0.0) or np.any(q >= 1.0):
        raise ValueError("default_probability must lie in [0, 1)")
    if np.any(rho < 0.0) or np.any(rho >= 1.0):
        raise ValueError("correlation must lie in [0, 1)")

    barrier = norm.ppf(q)
    numerator = barrier - np.sqrt(rho) * z
    denominator = np.sqrt(1.0 - rho)
    return norm.cdf(numerator / denominator)
