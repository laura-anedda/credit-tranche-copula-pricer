"""
Credit index tranche pricer based on the one-factor Gaussian copula model.

This package implements the standard market methodology for pricing
synthetic CDO tranches and credit index tranches (e.g. CDX, iTraxx),
following the framework of Li (2000) for dependence modelling and
Andersen, Sidenius & Basu (2003) for portfolio loss distribution
construction. Large Homogeneous Portfolio (LHP) closed-form pricing
(Vasicek, 1987/2002) and base correlation bootstrapping (McGinty et al.,
2004) are also provided.

Modules
-------
conventions
    Day-count, business-day, and CDS premium schedule construction.
market_data
    Discount and credit (survival) curve construction and bootstrapping.
numerics
    Numerical integration, root-finding and interpolation utilities.
copula
    One-factor Gaussian copula conditional default probabilities.
portfolio
    Conditional and unconditional portfolio loss distributions.
tranche
    Expected tranche loss, protection/premium leg valuation, par spreads.
quotes
    Structured market quote representation for index tranches.
base_correlation
    Base correlation bootstrapping from market tranche quotes.
pricer
    High-level orchestration objects tying the above together.
reporting
    Pure tabular (pandas) reporting layer for presentation surfaces
    such as the standalone HTML dashboard in `dashboard/`.
diagnostics
    Calibration accuracy, no-arbitrage, and stability diagnostics.
"""

from credit_copula import (
    base_correlation,
    conventions,
    copula,
    diagnostics,
    market_data,
    numerics,
    portfolio,
    pricer,
    quotes,
    reporting,
    tranche,
)

__all__ = [
    "base_correlation",
    "conventions",
    "copula",
    "diagnostics",
    "market_data",
    "numerics",
    "portfolio",
    "pricer",
    "quotes",
    "reporting",
    "tranche",
]
