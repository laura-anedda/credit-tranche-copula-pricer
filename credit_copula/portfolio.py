"""
Portfolio composition and credit loss distribution construction.

This module provides:

1. The `Obligor` and `CreditPortfolio` data structures describing a
   credit index or bespoke synthetic CDO reference portfolio.
2. The semi-analytic recursive algorithm of Andersen, Sidenius & Basu
   (2003) for constructing the discretized portfolio loss distribution
   conditional on the systemic factor, and its unconditional
   counterpart obtained by Gauss-Hermite integration.
3. The Large Homogeneous Portfolio (LHP) closed-form approximation
   (Vasicek, 1987/2002) for the expected loss above a threshold,
   applicable when portfolio granularity is high and obligor
   parameters can be treated as homogeneous (or homogenized via
   notional/spread-weighted averages).

The two approaches trade off accuracy against speed: the ASB recursion
is exact (up to loss discretization error) for a finite, heterogeneous
portfolio and is the standard approach for pricing bespoke and index
tranches; the LHP approximation is a closed-form limit valid as the
number of obligors grows large and idiosyncratic risk diversifies away,
useful for quick analytics, sensitivity approximations, and as a sanity
check on the recursive method.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import multivariate_normal, norm

from credit_copula import copula, numerics
from credit_copula.market_data import CreditCurve

__all__ = [
    "Obligor",
    "CreditPortfolio",
    "discretize_loss_given_default",
    "conditional_loss_distribution",
    "loss_distribution",
    "vasicek_expected_loss_above_threshold",
]


@dataclass(frozen=True)
class Obligor:
    """
    A single reference entity within a credit index or CDO portfolio.

    Parameters
    ----------
    name : str
        Reference entity identifier.
    notional : float
        Reference notional, in currency units. Must be strictly
        positive.
    recovery_rate : float
        Assumed fractional recovery rate upon default, in
        :math:`[0, 1)`. The loss given default (LGD) for this obligor
        is ``notional * (1 - recovery_rate)``.
    correlation : float
        One-factor Gaussian copula factor loading
        :math:`\\rho_i \\in [0, 1)`, representing the squared
        correlation of this obligor's latent asset value with the
        systemic factor `Z`.
    credit_curve : CreditCurve
        Bootstrapped survival probability curve for this obligor.

    Raises
    ------
    ValueError
        If `notional` is non-positive, `recovery_rate` is outside
        :math:`[0, 1)`, or `correlation` is outside :math:`[0, 1)`.
    """

    name: str
    notional: float
    recovery_rate: float
    correlation: float
    credit_curve: CreditCurve

    def __post_init__(self) -> None:
        if self.notional <= 0.0:
            raise ValueError("notional must be strictly positive")
        if not (0.0 <= self.recovery_rate < 1.0):
            raise ValueError("recovery_rate must lie in [0, 1)")
        if not (0.0 <= self.correlation < 1.0):
            raise ValueError("correlation must lie in [0, 1)")

    @property
    def loss_given_default(self) -> float:
        """Loss given default in currency units: ``notional * (1 - R)``."""
        return self.notional * (1.0 - self.recovery_rate)


@dataclass(frozen=True)
class CreditPortfolio:
    """
    A static, fully-funded reference portfolio of `Obligor` instances.

    Parameters
    ----------
    obligors : tuple[Obligor, ...]
        The constituent reference entities. Must be non-empty.

    Raises
    ------
    ValueError
        If `obligors` is empty.
    """

    obligors: tuple[Obligor, ...]

    def __post_init__(self) -> None:
        if len(self.obligors) == 0:
            raise ValueError("obligors must be non-empty")

    @property
    def total_notional(self) -> float:
        """Sum of all constituent notionals."""
        return float(sum(o.notional for o in self.obligors))

    @property
    def n_obligors(self) -> int:
        """Number of reference entities in the portfolio."""
        return len(self.obligors)

    def default_probabilities(self, t: float) -> np.ndarray:
        """Marginal cumulative default probabilities of all obligors at time `t`."""
        return np.array([o.credit_curve.default_probability(t) for o in self.obligors])

    def correlations(self) -> np.ndarray:
        """One-factor copula correlations of all obligors."""
        return np.array([o.correlation for o in self.obligors])

    def loss_given_defaults(self) -> np.ndarray:
        """Currency loss-given-default amounts of all obligors."""
        return np.array([o.loss_given_default for o in self.obligors])


def discretize_loss_given_default(
    loss_given_defaults: np.ndarray, loss_unit: float
) -> np.ndarray:
    """
    Map continuous loss-given-default amounts onto an integer grid of
    `loss_unit`-sized buckets, by rounding to the nearest integer
    multiple.

    This discretization is the standard implementation device for the
    Andersen-Sidenius-Basu recursive loss distribution algorithm, which
    requires losses to lie on a common, equally-spaced grid so that
    individual-name loss contributions can be combined by integer index
    shifts. The discretization error introduced is controlled by the
    choice of `loss_unit`: a finer unit reduces bias at the cost of a
    proportionally larger number of loss buckets (and hence recursion
    cost).

    Parameters
    ----------
    loss_given_defaults : np.ndarray
        Continuous per-obligor loss-given-default amounts (currency
        units).
    loss_unit : float
        Size of one discretization bucket (currency units). Must be
        strictly positive.

    Returns
    -------
    np.ndarray
        Integer number of buckets per obligor (minimum 1, so that an
        obligor with positive LGD always contributes some loss upon
        default).
    """
    buckets = np.round(loss_given_defaults / loss_unit).astype(np.int64)
    return np.maximum(buckets, 1)


def conditional_loss_distribution(
    default_probabilities: np.ndarray,
    correlations: np.ndarray,
    systemic_factor: float,
    lgd_buckets: np.ndarray,
    n_buckets: int,
) -> np.ndarray:
    """
    Discretized portfolio loss distribution conditional on the systemic factor.

    Implements the recursive convolution algorithm of Andersen, Sidenius
    & Basu (2003). Conditional on a realization `z` of the systemic
    factor, obligor defaults are independent Bernoulli trials with
    conditional default probabilities :math:`p_i(z)` (see
    :func:`credit_copula.copula.conditional_default_probability`). The
    distribution of total portfolio loss (measured in integer multiples
    of a fixed loss unit) is built up name-by-name via the recursion

    .. math::
        f_0(k) = \\mathbb{1}\\{k = 0\\}

    .. math::
        f_i(k) = f_{i-1}(k)\\,(1 - p_i) + f_{i-1}(k - \\ell_i)\\, p_i

    where :math:`f_i(k)` is the probability that the cumulative loss
    from the first `i` obligors equals `k` loss units, and
    :math:`\\ell_i` is obligor `i`'s loss-given-default expressed as an
    integer number of loss units. After processing all `N` obligors,
    :math:`f_N(\\cdot)` is the conditional portfolio loss distribution.

    Parameters
    ----------
    default_probabilities : np.ndarray
        Marginal cumulative default probabilities :math:`Q_i^D(t)`,
        shape ``(N,)``.
    correlations : np.ndarray
        One-factor copula correlations :math:`\\rho_i`, shape ``(N,)``.
    systemic_factor : float
        Realization `z` of the systemic factor `Z`.
    lgd_buckets : np.ndarray
        Integer loss-given-default, in loss units, per obligor, shape
        ``(N,)``. See :func:`discretize_loss_given_default`.
    n_buckets : int
        Number of loss buckets to track, i.e. the support of the
        returned distribution is ``{0, 1, ..., n_buckets - 1}`` loss
        units. Must be at least ``max(lgd_buckets) + 1``; losses that
        would exceed this range are not separately tracked (in
        practice `n_buckets` is chosen to cover the full portfolio
        loss-given-default, so this truncation never occurs).

    Returns
    -------
    np.ndarray
        Conditional probability mass function over loss buckets, shape
        ``(n_buckets,)``, summing to 1.0 up to floating-point precision.

    Notes
    -----
    Computational complexity is :math:`O(N \\cdot L)` where `L` is
    `n_buckets`, since each of the `N` recursion steps performs a
    single vectorized shift-and-combine operation over the loss grid.
    This is the standard complexity of the ASB algorithm and is
    dramatically more efficient than direct enumeration of the
    :math:`2^N` default scenarios.
    """
    cond_default_prob = copula.conditional_default_probability(
        default_probabilities, correlations, systemic_factor
    )
    distribution = np.zeros(n_buckets, dtype=np.float64)
    distribution[0] = 1.0

    for default_prob, lgd_units in zip(cond_default_prob, lgd_buckets):
        lgd_units = int(lgd_units)
        survived_contribution = distribution * (1.0 - default_prob)
        defaulted_contribution = np.zeros(n_buckets, dtype=np.float64)
        if lgd_units < n_buckets:
            defaulted_contribution[lgd_units:] = distribution[: n_buckets - lgd_units] * default_prob
        distribution = survived_contribution + defaulted_contribution

    return distribution


def loss_distribution(
    default_probabilities: np.ndarray,
    correlations: np.ndarray,
    loss_given_defaults: np.ndarray,
    loss_unit: float,
    n_buckets: int,
    n_quadrature_points: int = 32,
) -> np.ndarray:
    """
    Unconditional portfolio loss distribution via Gauss-Hermite integration.

    Integrates the conditional loss distribution
    (:func:`conditional_loss_distribution`) over the standard normal
    distribution of the systemic factor `Z`:

    .. math::
        f(k) = \\int_{-\\infty}^{\\infty} f(k \\mid z)\\, \\varphi(z)\\, dz
             \\approx \\sum_{j=1}^{M} w_j\\, f(k \\mid z_j)

    where :math:`\\{z_j, w_j\\}_{j=1}^{M}` are Gauss-Hermite quadrature
    nodes and weights for the normal density (see
    :func:`credit_copula.numerics.gauss_hermite_nodes_weights`).

    Parameters
    ----------
    default_probabilities : np.ndarray
        Marginal cumulative default probabilities, shape ``(N,)``.
    correlations : np.ndarray
        One-factor copula correlations, shape ``(N,)``.
    loss_given_defaults : np.ndarray
        Continuous per-obligor loss-given-default amounts (currency
        units), shape ``(N,)``.
    loss_unit : float
        Discretization bucket size (currency units).
    n_buckets : int
        Number of loss buckets in the support grid.
    n_quadrature_points : int, default=32
        Number of Gauss-Hermite quadrature nodes. 32 nodes typically
        give double-precision accuracy for the smooth conditional loss
        distribution functions encountered here.

    Returns
    -------
    np.ndarray
        Unconditional probability mass function over loss buckets,
        shape ``(n_buckets,)``, summing to 1.0 up to quadrature and
        discretization error.

    Notes
    -----
    Quadrature accuracy degrades for very high correlations approaching
    1, where the conditional loss distribution becomes a near-step
    function of `z`; in that regime a larger `n_quadrature_points` is
    recommended. For the correlation ranges typically calibrated to
    index tranche markets (0.1-0.6), 32 nodes are more than sufficient.
    """
    lgd_buckets = discretize_loss_given_default(loss_given_defaults, loss_unit)
    nodes, weights = numerics.gauss_hermite_nodes_weights(n_quadrature_points)

    total_distribution = np.zeros(n_buckets, dtype=np.float64)
    for node, weight in zip(nodes, weights):
        total_distribution += weight * conditional_loss_distribution(
            default_probabilities, correlations, node, lgd_buckets, n_buckets
        )
    return total_distribution


def vasicek_expected_loss_above_threshold(
    default_probability: float, correlation: float, threshold: float
) -> float:
    """
    Closed-form expected fractional loss above a threshold under the LHP limit.

    In the Large Homogeneous Portfolio limit (infinitely many obligors
    with identical default probability `p` and correlation
    :math:`\\rho`), the law of large numbers collapses the conditional
    portfolio default rate onto its conditional expectation, so the
    fractional default rate becomes a deterministic function of the
    systemic factor:

    .. math::
        l(z) = \\Phi\\!\\left(
            \\frac{\\Phi^{-1}(p) - \\sqrt{\\rho}\\, z}{\\sqrt{1 - \\rho}}
        \\right)

    This is the Vasicek (1987, 2002) large portfolio limit. The quantity
    priced here is :math:`E[(l(Z) - K)^+]`, the building block for
    expected tranche loss (an arbitrary tranche payoff is a difference
    of two such call-option-like payoffs on the portfolio default rate).
    Since :math:`l(z)` is strictly decreasing in `z` (for
    :math:`\\rho > 0`), the event :math:`\\{l(Z) > K\\}` is equivalent to
    :math:`\\{Z < z^\\*\\}` with

    .. math::
        z^\\* = \\frac{\\Phi^{-1}(p) - \\sqrt{1 - \\rho}\\, \\Phi^{-1}(K)}{\\sqrt{\\rho}}

    and the expectation reduces, via the Owen (1980) identity for the
    integral of a normal CDF against a normal density, to a bivariate
    normal CDF evaluation:

    .. math::
        E[(l(Z) - K)^+] = \\Phi_2\\!\\big(z^\\*, \\Phi^{-1}(p);\\, \\sqrt{\\rho}\\big)
                            - K\\, \\Phi(z^\\*)

    where :math:`\\Phi_2(\\cdot, \\cdot; r)` is the standard bivariate
    normal CDF with correlation `r`.

    Parameters
    ----------
    default_probability : float
        Homogeneous marginal cumulative default probability
        :math:`p = Q^D(T) \\in (0, 1)`.
    correlation : float
        Homogeneous one-factor copula correlation
        :math:`\\rho \\in (0, 1)`.
    threshold : float
        Fractional default rate threshold :math:`K \\in [0, 1]`.

    Returns
    -------
    float
        :math:`E[(l(Z) - K)^+] \\in [0, 1]`.

    Raises
    ------
    ValueError
        If inputs lie outside their required domains.

    Notes
    -----
    For :math:`K = 0`, :math:`E[(l(Z) - 0)^+] = E[l(Z)] = p`, recovering
    the marginal default probability, which provides a useful
    closed-form sanity check on the implementation.

    References
    ----------
    .. [Vasicek2002] Vasicek, O. (2002). "Loan Portfolio Value." Risk
       Magazine, December 2002.
    .. [OKane2003] O'Kane, D., & Schloegl, L. (2003). "A Note on the
       Large Homogeneous Portfolio Approximation with the Student-t
       Copula." Lehman Brothers Quantitative Credit Research.
    """
    if not (0.0 < default_probability < 1.0):
        raise ValueError("default_probability must lie in (0, 1)")
    if not (0.0 < correlation < 1.0):
        raise ValueError("correlation must lie in (0, 1)")
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must lie in [0, 1]")

    if threshold == 0.0:
        return default_probability
    if threshold == 1.0:
        return 0.0

    c_p = norm.ppf(default_probability)
    sqrt_rho = np.sqrt(correlation)
    sqrt_one_minus_rho = np.sqrt(1.0 - correlation)
    z_star = (c_p - sqrt_one_minus_rho * norm.ppf(threshold)) / sqrt_rho

    bivariate_cdf = multivariate_normal(
        mean=[0.0, 0.0], cov=[[1.0, sqrt_rho], [sqrt_rho, 1.0]]
    ).cdf([z_star, c_p])

    expected_excess = bivariate_cdf - threshold * norm.cdf(z_star)
    # Clip to [0, 1] to absorb negligible numerical noise from the
    # bivariate normal CDF evaluation near the domain boundary.
    return float(np.clip(expected_excess, 0.0, 1.0))
