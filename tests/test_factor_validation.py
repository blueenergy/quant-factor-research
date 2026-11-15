"""Comparison metrics and verdicts for the factor validation harness."""

import numpy as np
import pandas as pd
import pytest

from analysis.factor.factor_validation import compare, summarize


def _long(values, factor="alpha101_101", date="20240628"):
    symbols = [f"{index:06d}.SZ" for index in range(len(values))]
    return pd.DataFrame(
        {
            "trade_date": date,
            "symbol": symbols,
            "factor": factor,
            "value": values,
        }
    )


def test_compare_identical_series_scores_perfectly():
    values = np.linspace(-1.0, 1.0, 100)
    comparison = compare(_long(values), _long(values))
    row = comparison.iloc[0]
    assert row["spearman"] == pytest.approx(1.0)
    assert row["median_abs_rel_error"] == pytest.approx(0.0)
    assert row["top50_overlap"] == pytest.approx(1.0)


def test_compare_reversed_series_scores_minus_one():
    values = np.linspace(-1.0, 1.0, 100)
    comparison = compare(_long(values), _long(values[::-1]))
    assert comparison.iloc[0]["spearman"] == pytest.approx(-1.0)


def test_compare_reports_missing_overlap():
    ours = _long([1.0, 2.0, np.nan, 4.0] * 25)
    theirs = _long([np.nan, 2.0, 3.0, 4.0] * 25)
    comparison = compare(ours, theirs)
    row = comparison.iloc[0]
    assert row["n_both"] == 50
    assert row["n_ours"] == 75
    assert row["n_theirs"] == 75


def _comparison_rows(factor, spearman, rel_error=0.0, minimum=None):
    values = [spearman] * 10
    values[0] = spearman if minimum is None else minimum
    return pd.DataFrame(
        {
            "factor": factor,
            "trade_date": [f"2024{index:04d}" for index in range(10)],
            "spearman": values,
            "median_abs_rel_error": [rel_error] * 10,
            "top50_overlap": [1.0] * 10,
            "n_both": [100] * 10,
        }
    )


def test_raw_factor_passes_when_mean_and_error_are_tight():
    summary = summarize(_comparison_rows("return_21d", 0.995, rel_error=0.001))
    assert summary.iloc[0]["verdict"] == "PASS"


def test_raw_factor_with_loose_error_is_a_caveat_not_a_failure():
    summary = summarize(_comparison_rows("return_21d", 0.995, rel_error=0.05))
    assert summary.iloc[0]["verdict"] == "PASS_WITH_CAVEAT"


def test_rank_factor_requires_the_minimum_date_to_hold():
    summary = summarize(_comparison_rows("alpha101_33", 0.98, minimum=0.70))
    assert summary.iloc[0]["verdict"] == "PASS_WITH_CAVEAT"


def test_rank_factor_below_the_floor_fails_semantics():
    summary = summarize(_comparison_rows("alpha101_33", 0.60))
    assert summary.iloc[0]["verdict"] == "FAIL_SEMANTICS"


def test_financial_factor_is_allowed_a_larger_caveat_band():
    summary = summarize(_comparison_rows("yoy_roe", 0.91))
    assert summary.iloc[0]["verdict"] == "PASS_WITH_CAVEAT"
