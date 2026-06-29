"""Shared pytest fixtures for the credit_copula test suite."""

from __future__ import annotations

import numpy as np
import pytest

from credit_copula.market_data import CreditCurve, DiscountCurve
from credit_copula.portfolio import CreditPortfolio, Obligor


@pytest.fixture
def flat_discount_curve() -> DiscountCurve:
    """A flat 3% continuously-compounded discount curve out to 10 years."""
    return DiscountCurve(tenors=np.array([1.0, 2.0, 5.0, 10.0]), zero_rates=np.full(4, 0.03))


@pytest.fixture
def flat_credit_curve() -> CreditCurve:
    """A single-pillar 200bp flat hazard rate credit curve, 40% recovery."""
    return CreditCurve(
        tenors=np.array([10.0]), hazard_rates=np.array([0.02]), recovery_rate=0.4
    )


@pytest.fixture
def small_homogeneous_portfolio(flat_credit_curve: CreditCurve) -> CreditPortfolio:
    """A small, fully homogeneous 10-name portfolio for fast unit testing."""
    obligors = tuple(
        Obligor(
            name=f"Obligor_{i}",
            notional=1.0,
            recovery_rate=0.4,
            correlation=0.3,
            credit_curve=flat_credit_curve,
        )
        for i in range(10)
    )
    return CreditPortfolio(obligors=obligors)
