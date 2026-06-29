"""
Numerical methods shared across the copula pricing framework.

This module isolates the generic numerical building blocks --
Gauss-Hermite quadrature for integration against a Gaussian density,
and a thin, well-documented wrapper around Brent's root-finding method
-- from the financial pricing logic in other modules. Separating these
concerns keeps the pricing code declarative and makes the numerical
choices (quadrature order, convergence tolerance) independently
testable and tunable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import brentq

__all__ = ["gauss_hermite_nodes_weights", "solve_root_brent", "RootDiagnostics"]


@dataclass(frozen=True)
class RootDiagnostics:
    """
    Convergence diagnostics for a single Brent root-finding call.

    Attributes
    ----------
    root : float
        The solved root.
    iterations : int
        Number of iterations performed by the solver.
    converged : bool
        Whether the solver reported convergence within `max_iterations`.
    function_calls : int
        Number of objective function evaluations performed.
    residual : float
        Value of the objective function evaluated at `root`. Should be
        close to zero at convergence; its magnitude relative to the
        requested tolerance is a direct measure of calibration quality
        and is surfaced in calibration diagnostics (see
        :mod:`credit_copula.diagnostics`).
    """

    root: float
    iterations: int
    converged: bool
    function_calls: int
    residual: float


def gauss_hermite_nodes_weights(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Quadrature nodes and weights for integrating against the standard
    normal density.

    The one-factor Gaussian copula requires evaluating integrals of the
    form

    .. math::
        I = \\int_{-\\infty}^{\\infty} f(z)\\, \\varphi(z)\\, dz

    where :math:`\\varphi(z) = \\tfrac{1}{\\sqrt{2\\pi}} e^{-z^2/2}` is the
    standard normal density (the distribution of the common systemic
    factor `Z`). This is exactly the "probabilist's" Gauss-Hermite
    quadrature rule, which is exact for `f` polynomial of degree up to
    ``2*n_points - 1`` and converges geometrically fast for the smooth,
    rapidly-decaying integrands arising in conditional default
    probability calculations.

    The probabilist's Hermite polynomials (as opposed to the "physicist's"
    convention with weight :math:`e^{-x^2}`) are used so that no change
    of variables is required: the returned nodes and weights directly
    satisfy :math:`I \\approx \\sum_k w_k f(z_k)`.

    Parameters
    ----------
    n_points : int
        Number of quadrature nodes. Values in the range 20-40 typically
        achieve double-precision accuracy for the smooth conditional
        default probability functions encountered in single-factor
        copula models.

    Returns
    -------
    nodes : np.ndarray
        Quadrature abscissae (realizations of the systemic factor `Z`).
    weights : np.ndarray
        Quadrature weights, summing to 1.0 (since they already
        integrate the normal density to unity).

    Raises
    ------
    ValueError
        If `n_points` is not a positive integer.

    Notes
    -----
    Internally uses :func:`numpy.polynomial.hermite_e.hermegauss`, which
    returns nodes and weights for the weight function :math:`e^{-x^2/2}`
    (unnormalized). The weights are rescaled here by
    :math:`1/\\sqrt{2\\pi}` to integrate the *normalized* Gaussian density.
    """
    if n_points <= 0:
        raise ValueError("n_points must be a positive integer")
    nodes, raw_weights = hermegauss(n_points)
    weights = raw_weights / np.sqrt(2.0 * np.pi)
    return nodes, weights


def solve_root_brent(
    func: Callable[[float], float],
    lower: float,
    upper: float,
    tolerance: float = 1.0e-10,
    max_iterations: int = 200,
    full_output: bool = False,
) -> float | RootDiagnostics:
    """
    Solve `func(x) = 0` on `[lower, upper]` using Brent's method.

    This is a thin wrapper around `scipy.optimize.brentq` used
    throughout the package for hazard rate bootstrapping and base
    correlation implication, both of which involve solving a single
    monotone (or at least single-root) scalar equation. Brent's method
    combines bisection, secant and inverse quadratic interpolation
    steps, giving guaranteed convergence (inherited from bisection)
    with superlinear convergence speed in the well-behaved cases that
    arise in copula calibration.

    Parameters
    ----------
    func : Callable[[float], float]
        Objective function. Must change sign over `[lower, upper]`.
    lower : float
        Lower bracket bound.
    upper : float
        Upper bracket bound.
    tolerance : float, default=1e-10
        Absolute tolerance on the root (`xtol` parameter of `brentq`).
    max_iterations : int, default=200
        Maximum number of iterations before raising.
    full_output : bool, default=False
        If `True`, return a :class:`RootDiagnostics` instance carrying
        the iteration count, convergence flag, function call count, and
        residual at the root, in addition to the root itself. Used by
        the calibration diagnostics layer (see
        :mod:`credit_copula.diagnostics` and
        :mod:`credit_copula.base_correlation`) to report convergence
        quality alongside calibrated parameters. If `False` (the
        default), only the root is returned, preserving the original
        scalar-returning signature.

    Returns
    -------
    float or RootDiagnostics
        The root of `func` within `[lower, upper]` if `full_output` is
        `False`; otherwise a :class:`RootDiagnostics` instance.

    Raises
    ------
    ValueError
        If `func(lower)` and `func(upper)` do not bracket a root
        (i.e. do not have opposite signs).

    Notes
    -----
    The caller is responsible for choosing an economically meaningful
    bracket (e.g. correlation in :math:`[0, 1)`, hazard rate in
    :math:`[0, +\\infty)` truncated to a finite upper bound). No
    automatic bracket expansion is performed, since silently widening
    the search range can mask calibration failures that should instead
    be surfaced to the caller.
    """
    f_lower = func(lower)
    f_upper = func(upper)

    if f_lower == 0.0:
        return lower if not full_output else RootDiagnostics(lower, 0, True, 1, f_lower)
    if f_upper == 0.0:
        return upper if not full_output else RootDiagnostics(upper, 0, True, 1, f_upper)
    if np.sign(f_lower) == np.sign(f_upper):
        raise ValueError(
            "func does not bracket a root on [lower, upper]: "
            f"f(lower)={f_lower}, f(upper)={f_upper}"
        )

    if not full_output:
        return brentq(func, lower, upper, xtol=tolerance, maxiter=max_iterations)

    root, results = brentq(
        func, lower, upper, xtol=tolerance, maxiter=max_iterations, full_output=True
    )
    return RootDiagnostics(
        root=root,
        iterations=results.iterations,
        converged=results.converged,
        function_calls=results.function_calls,
        residual=func(root),
    )
