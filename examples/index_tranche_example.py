"""
Worked example: pricing CDX/iTraxx-style index tranches.

This script demonstrates the end-to-end workflow supported by the
`credit_copula` package:

1. Build a risk-free discount curve.
2. Bootstrap a single-name CDS survival curve from market par spreads.
3. Construct a homogeneous reference portfolio of 125 obligors (the
   standard CDX/iTraxx index constituent count).
4. Price a set of standard tranches (0-3%, 3-7%, 7-10%, 10-15%, 15-30%)
   under the one-factor Gaussian copula using the Andersen-Sidenius-
   Basu recursive loss distribution.
5. Bootstrap a base correlation curve from the resulting par spreads.

Run with::

    python examples/index_tranche_example.py
"""

from __future__ import annotations

import numpy as np

from credit_copula.base_correlation import bootstrap_base_correlation
from credit_copula.market_data import DiscountCurve, bootstrap_hazard_rates
from credit_copula.portfolio import CreditPortfolio, Obligor
from credit_copula.pricer import CDOPricer
from credit_copula.tranche import Tranche


def main() -> None:
    # 1. Risk-free discount curve (flat 3% for illustration purposes).
    discount_curve = DiscountCurve(
        tenors=np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0]),
        zero_rates=np.full(6, 0.03),
    )

    # 2. Bootstrap a representative investment-grade CDS curve.
    cds_tenors = np.array([1.0, 3.0, 5.0, 7.0, 10.0])
    cds_spreads = np.array([0.0040, 0.0065, 0.0090, 0.0105, 0.0120])
    recovery_rate = 0.40
    credit_curve = bootstrap_hazard_rates(cds_tenors, cds_spreads, recovery_rate, discount_curve)

    # 3. Build a 125-name homogeneous index portfolio with a uniform
    # one-factor correlation of 25%, broadly consistent with historical
    # CDX.IG / iTraxx Europe compound correlation levels.
    n_obligors = 125
    obligor_notional = 1.0
    obligors = tuple(
        Obligor(
            name=f"Index_Constituent_{i:03d}",
            notional=obligor_notional,
            recovery_rate=recovery_rate,
            correlation=0.25,
            credit_curve=credit_curve,
        )
        for i in range(n_obligors)
    )
    portfolio = CreditPortfolio(obligors=obligors)
    index_notional = portfolio.total_notional

    pricer = CDOPricer(
        portfolio=portfolio,
        discount_curve=discount_curve,
        loss_unit=0.25,
        n_quadrature_points=32,
        payment_frequency=4,
        n_integration_steps_per_year=12,
    )

    # 4. Price the standard CDX.IG tranche structure (attachment points
    # expressed as a percentage of index notional).
    tranche_pillars = [
        (0.00, 0.03),
        (0.03, 0.07),
        (0.07, 0.10),
        (0.10, 0.15),
        (0.15, 0.30),
    ]
    maturity = 5.0

    print(f"Index notional: {index_notional:.2f}")
    print(f"5y index-average survival probability: {credit_curve.survival_probability(maturity):.4f}")
    print()
    print(f"{'Tranche':>12} {'Par Spread (bps)':>18} {'Risky PV01':>12}")

    par_spreads = []
    for attach_pct, detach_pct in tranche_pillars:
        tranche = Tranche(
            attachment=attach_pct * index_notional, detachment=detach_pct * index_notional
        )
        result = pricer.price_tranche(tranche, maturity)
        par_spreads.append(result.par_spread)
        label = f"{attach_pct:.0%}-{detach_pct:.0%}"
        print(f"{label:>12} {result.par_spread * 1e4:>18.1f} {result.risky_pv01:>12.3f}")

    # 5. Bootstrap base correlation. Base correlation is calibrated to
    # the par spreads of *base* tranches [0, K] -- not the standard
    # tranches [A, D] priced above -- so the "market" base tranche
    # spreads are computed here directly from the heterogeneous (25%
    # correlation) portfolio. In practice these base tranche spreads
    # would instead be observed market quotes, or derived from standard
    # tranche quotes via the base-tranche stripping identity
    # Tranche[A, D] = BaseTranche[0, D] - BaseTranche[0, A].
    detachment_points = np.array([d for _, d in tranche_pillars]) * index_notional
    base_tranche_spreads = np.array(
        [
            pricer.price_tranche(Tranche(0.0, detachment), maturity).par_spread
            for detachment in detachment_points
        ]
    )
    base_correlation_curve = bootstrap_base_correlation(
        pricer, detachment_points, base_tranche_spreads, maturity
    )

    print()
    print(f"{'Detachment':>12} {'Base Correlation':>18}")
    for detachment, rho in zip(
        base_correlation_curve.detachment_points, base_correlation_curve.correlations
    ):
        print(f"{detachment / index_notional:>11.0%} {rho:>18.4f}")


if __name__ == "__main__":
    main()
