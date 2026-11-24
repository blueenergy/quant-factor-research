"""The primer file must cover every factor the registry knows about."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.factor.alpha158_features import factor_catalog  # noqa: E402
from analysis.factor.alpha_factor_backtest import factor_set_catalog  # noqa: E402
from analysis.factor.primers import (  # noqa: E402
    FACTOR_PRIMERS,
    FAMILY_PRIMERS,
    OPERATOR_PRIMERS,
    primer_for,
)

FIELDS = ("intuition", "reading", "example", "caveats")

#: Registry documents that are not alpha158/alpha8 and therefore have to be
#: listed by hand. Mirrors ``VAULT_DEFS`` in the registry backfill.
EXPLICIT_IDS = [
    "fm1.rev_yoy", "fm1.rev_cagr", "fm1.rev_yoy_std", "fm1.profit_yoy",
    "fm1.profit_positive_ratio", "fm1.roe", "fm1.roe_trend", "fm1.rd_ratio",
    "fm1.pe_ttm", "fm1.pb", "fm1.momentum_20d", "fm1.momentum_60d",
    "fm1.volatility_20d", "fm1.turnover_20d", "fm1.log_market_cap",
    "vault.bias_turn_21d_252d", "vault.return_21d", "vault.beta_60d_000300",
    "vault.gpm_ttm", "vault.yoy_roe", "vault.q_ia", "vault.q_roe",
    "vault.beta_252", "vault.analyst_fy1", "vault.analyst_fy2",
    "alpha101.1", "alpha101.12", "alpha101.33", "alpha101.101",
    "group.value", "group.quality", "group.growth",
    "group.value_quality_growth", "group.course_44",
]


def test_every_alpha158_operator_has_a_primer():
    operators = {entry["operator"] for entry in factor_catalog()}
    missing = sorted(operators - set(OPERATOR_PRIMERS))
    assert missing == []


def test_every_alpha8_factor_has_a_primer():
    names = {entry["name"] for entry in factor_set_catalog("alpha8")}
    missing = sorted(name for name in names
                     if f"alpha8.{name}" not in FACTOR_PRIMERS)
    assert missing == []


@pytest.mark.parametrize("factor_id", EXPLICIT_IDS)
def test_explicit_factor_ids_resolve_to_their_own_primer(factor_id):
    primer = primer_for(factor_id)
    assert primer["scope"] == "factor", factor_id
    for field in FIELDS:
        assert primer[field].strip(), f"{factor_id}.{field} is empty"


def test_window_is_substituted_into_operator_templates():
    primer = primer_for("alpha158.STD5", operator="STD", window=5)
    assert primer["scope"] == "factor"
    assert "{d}" not in primer["intuition"]
    assert "5" in primer["intuition"]


def test_operator_template_wins_over_family_fallback():
    primer = primer_for("alpha158.RANK20", operator="RANK", family="区间位置")
    assert primer["scope"] == "factor"
    assert primer["intuition"] != FAMILY_PRIMERS["区间位置"]["intuition"]


def test_family_fallback_is_used_when_nothing_else_matches():
    primer = primer_for("some.new.factor", operator=None, family="quality")
    assert primer["scope"] == "family"
    assert primer["intuition"] == FAMILY_PRIMERS["quality"]["intuition"]


def test_unknown_factor_gets_an_empty_primer():
    primer = primer_for("some.new.factor")
    assert primer["scope"] == "none"
    assert primer["intuition"] == ""
    assert primer["refs"] == []


DETAIL_IDS = [
    "fm1.momentum_20d", "fm1.momentum_60d", "fm1.volatility_20d",
    "fm1.log_market_cap", "fm1.turnover_20d", "vault.bias_turn_21d_252d",
]


@pytest.mark.parametrize("factor_id", DETAIL_IDS)
def test_long_form_detail_has_five_sections(factor_id):
    primer = primer_for(factor_id)
    sections = primer.get("detail") or []
    assert len(sections) == 5, factor_id
    for section in sections:
        assert section["heading"].strip(), factor_id
        assert section["body"].strip(), f"{factor_id}/{section['heading']}"


def test_only_the_six_in_use_factors_carry_long_form_detail():
    with_detail = {key for key, primer in FACTOR_PRIMERS.items() if primer.get("detail")}
    assert with_detail == set(DETAIL_IDS)


def test_every_primer_is_complete_and_links_are_https():
    for key, primer in list(OPERATOR_PRIMERS.items()) + list(FACTOR_PRIMERS.items()):
        for field in FIELDS:
            assert primer[field].strip(), f"{key}.{field} is empty"
        for ref in primer.get("refs", []):
            assert ref["title"].strip(), f"{key} ref has no title"
            assert ref["url"].startswith("https://"), f"{key} ref url: {ref['url']}"
    for family, primer in FAMILY_PRIMERS.items():
        for field in FIELDS:
            assert primer[field].strip(), f"{family}.{field} is empty"
