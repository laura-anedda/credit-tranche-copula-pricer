"""Tests for credit_copula.quotes."""

from __future__ import annotations

from datetime import date

import pytest

from credit_copula.quotes import QuoteType, TrancheQuote


def _base_kwargs(**overrides):
    kwargs = dict(
        index_name="CDX.NA.IG",
        attachment_pct=0.03,
        detachment_pct=0.07,
        valuation_date=date(2025, 3, 20),
        maturity_date=date(2030, 3, 20),
        quote_type=QuoteType.PAR_SPREAD,
        running_spread=0.0250,
        notional=125.0,
    )
    kwargs.update(overrides)
    return kwargs


class TestTrancheQuoteValidation:
    def test_accepts_valid_par_spread_quote(self) -> None:
        quote = TrancheQuote(**_base_kwargs())
        assert quote.quote_type is QuoteType.PAR_SPREAD

    def test_accepts_valid_upfront_quote(self) -> None:
        quote = TrancheQuote(
            **_base_kwargs(
                attachment_pct=0.0, detachment_pct=0.03,
                quote_type=QuoteType.UPFRONT, running_spread=0.05, upfront_pct=0.32,
            )
        )
        assert quote.upfront_pct == pytest.approx(0.32)

    def test_rejects_misordered_attachment_detachment(self) -> None:
        with pytest.raises(ValueError):
            TrancheQuote(**_base_kwargs(attachment_pct=0.07, detachment_pct=0.03))

    def test_rejects_maturity_before_valuation(self) -> None:
        with pytest.raises(ValueError):
            TrancheQuote(**_base_kwargs(maturity_date=date(2020, 1, 1)))

    def test_rejects_upfront_quote_missing_upfront_pct(self) -> None:
        with pytest.raises(ValueError):
            TrancheQuote(**_base_kwargs(quote_type=QuoteType.UPFRONT))

    def test_rejects_par_spread_quote_with_upfront_pct(self) -> None:
        with pytest.raises(ValueError):
            TrancheQuote(**_base_kwargs(upfront_pct=0.1))

    def test_rejects_invalid_recovery_rate(self) -> None:
        with pytest.raises(ValueError):
            TrancheQuote(**_base_kwargs(recovery_rate=1.0))


class TestTrancheNotionalAndQuotedValue:
    def test_tranche_notional(self) -> None:
        quote = TrancheQuote(**_base_kwargs())
        assert quote.tranche_notional == pytest.approx((0.07 - 0.03) * 125.0)

    def test_quoted_value_for_par_spread(self) -> None:
        quote = TrancheQuote(**_base_kwargs(running_spread=0.025))
        assert quote.quoted_value() == pytest.approx(0.025)

    def test_quoted_value_for_upfront(self) -> None:
        quote = TrancheQuote(
            **_base_kwargs(
                attachment_pct=0.0, detachment_pct=0.03,
                quote_type=QuoteType.UPFRONT, running_spread=0.05, upfront_pct=0.32,
            )
        )
        assert quote.quoted_value() == pytest.approx(0.32)


class TestModelImpliedValue:
    def test_par_spread_matches_tranche_par_spread_function(self) -> None:
        quote = TrancheQuote(**_base_kwargs())
        implied = quote.model_implied_value(protection_pv=5.0, annuity=50.0)
        assert implied == pytest.approx(0.1)

    def test_upfront_matches_tranche_fair_upfront_function(self) -> None:
        quote = TrancheQuote(
            **_base_kwargs(
                attachment_pct=0.0, detachment_pct=0.03,
                quote_type=QuoteType.UPFRONT, running_spread=0.05, upfront_pct=0.30,
                notional=100.0,
            )
        )
        # tranche_notional = 0.03 * 100 = 3.0
        implied = quote.model_implied_value(protection_pv=0.2, annuity=2.0)
        assert implied == pytest.approx((0.2 - 0.05 * 2.0) / 3.0)
