"""Alpha158 expression evaluation.

The rolling operators are checked against brute-force implementations rather
than against golden numbers, so a regression shows up as a disagreement with
the definition instead of as a diff nobody can adjudicate.
"""

import numpy as np
import pandas as pd
import pytest

from analysis.factor.alpha158_features import (
    DEFAULT_WINDOWS,
    FAMILY_KBAR,
    FAMILY_PRICE,
    FAMILY_TREND,
    FAMILY_VOLATILITY,
    FAMILY_VOLUME,
    FAMILY_VOLUME_PRICE,
    ROLLING_OPERATORS,
    Evaluator,
    alpha158_feature_config,
    build_matrices,
    compute_alpha158,
    compute_expressions,
    factor_catalog,
    kbar_features,
    parse_expression,
    price_features,
    rolling_features,
    tokenize,
)


def _panel(symbols=("A", "B"), n_days=90, seed=11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for offset, symbol in enumerate(symbols):
        price = 100.0 + 25.0 * offset
        for day in range(n_days):
            price *= 1.0 + rng.normal(0.0, 0.02)
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": f"2025{day // 28 + 1:02d}{day % 28 + 1:02d}",
                    "open": price * 0.99,
                    "high": price * 1.03,
                    "low": price * 0.96,
                    "close": price,
                    "volume": float(rng.integers(500, 5000)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def evaluator():
    return Evaluator(build_matrices(_panel()))


def _evaluate(evaluator, expression):
    return evaluator.evaluate(parse_expression(expression))


# ===================== Factor definitions =====================
def test_default_config_has_158_uniquely_named_factors():
    fields, names = alpha158_feature_config()
    assert len(fields) == 158
    assert len(set(names)) == 158


def test_config_splits_into_kbar_price_and_rolling():
    assert len(kbar_features()[0]) == 9
    assert len(price_features()[0]) == 4
    assert len(rolling_features()[0]) == len(ROLLING_OPERATORS) * len(DEFAULT_WINDOWS)


def test_low_operator_is_named_min_to_match_qlib():
    _fields, names = rolling_features(windows=(5,))
    assert "MIN5" in names
    assert "LOW5" not in names


def test_every_expression_parses():
    fields, _names = alpha158_feature_config()
    for expression in fields:
        assert parse_expression(expression) is not None


# ===================== Parsing =====================
def test_tokenizer_splits_fields_numbers_and_operators():
    kinds = [token.kind for token in tokenize("($close-$open)/1e-12")]
    assert kinds == ["op", "field", "op", "field", "op", "op", "number"]


def test_multiplication_binds_tighter_than_addition(evaluator):
    result = _evaluate(evaluator, "1+2*3")
    assert result.to_numpy()[0, 0] == pytest.approx(7.0)


def test_parentheses_override_precedence(evaluator):
    result = _evaluate(evaluator, "(1+2)*3")
    assert result.to_numpy()[0, 0] == pytest.approx(9.0)


def test_unary_minus_is_supported(evaluator):
    close = evaluator.matrices["close"]
    assert _evaluate(evaluator, "-$close").equals(-close)


def test_unknown_operator_is_rejected(evaluator):
    with pytest.raises(ValueError, match="unsupported operator"):
        _evaluate(evaluator, "Bogus($close, 5)")


def test_trailing_tokens_are_rejected():
    with pytest.raises(ValueError, match="unconsumed"):
        parse_expression("$close 5")


def test_missing_base_field_raises_key_error(evaluator):
    with pytest.raises(KeyError):
        _evaluate(evaluator, "$vwap/$close")


# ===================== Rolling operators =====================
def test_ref_shifts_within_each_symbol(evaluator):
    close = evaluator.matrices["close"]
    assert _evaluate(evaluator, "Ref($close, 3)").equals(close.shift(3))


@pytest.mark.parametrize(
    "expression,reference",
    [
        ("Mean($close, 5)", lambda m: m["close"].rolling(5).mean()),
        ("Sum($close, 5)", lambda m: m["close"].rolling(5).sum()),
        ("Std($close, 10)", lambda m: m["close"].rolling(10).std()),
        ("Max($high, 20)", lambda m: m["high"].rolling(20).max()),
        ("Min($low, 20)", lambda m: m["low"].rolling(20).min()),
        ("Quantile($close, 10, 0.8)", lambda m: m["close"].rolling(10).quantile(0.8)),
        ("Rank($close, 10)", lambda m: m["close"].rolling(10).rank(pct=True)),
    ],
)
def test_simple_rolling_operators_match_pandas(evaluator, expression, reference):
    result = _evaluate(evaluator, expression)
    expected = reference(evaluator.matrices)
    pd.testing.assert_frame_equal(result, expected, check_names=False)


def test_corr_matches_pairwise_rolling_correlation(evaluator):
    matrices = evaluator.matrices
    result = _evaluate(evaluator, "Corr($close, Log($volume+1), 10)")
    expected = matrices["close"].rolling(10).corr(np.log(matrices["volume"] + 1), pairwise=False)
    pd.testing.assert_frame_equal(result, expected, check_names=False)


@pytest.mark.parametrize("window", [5, 10, 30])
def test_slope_matches_least_squares_fit(evaluator, window):
    close = evaluator.matrices["close"]
    result = _evaluate(evaluator, f"Slope($close, {window})")
    expected = _brute_force_regression(close, window, "slope")
    np.testing.assert_allclose(result.to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-9)


def test_rsquare_matches_least_squares_fit(evaluator):
    close = evaluator.matrices["close"]
    result = _evaluate(evaluator, "Rsquare($close, 10)")
    expected = _brute_force_regression(close, 10, "rsquare")
    np.testing.assert_allclose(result.to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-9)


def test_resi_is_the_last_bar_residual(evaluator):
    close = evaluator.matrices["close"]
    result = _evaluate(evaluator, "Resi($close, 10)")
    expected = _brute_force_regression(close, 10, "resi")
    np.testing.assert_allclose(result.to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-9)


def _brute_force_regression(frame: pd.DataFrame, window: int, what: str) -> pd.DataFrame:
    x = np.arange(window, dtype=float)
    out = {}
    for column in frame.columns:
        y = frame[column].to_numpy(dtype=float)
        values = np.full(len(y), np.nan)
        for end in range(window - 1, len(y)):
            chunk = y[end - window + 1 : end + 1]
            if np.isnan(chunk).any():
                continue
            slope, intercept = np.polyfit(x, chunk, 1)
            if what == "slope":
                values[end] = slope
            elif what == "resi":
                values[end] = chunk[-1] - (intercept + slope * (window - 1))
            else:
                residual = ((chunk - (intercept + slope * x)) ** 2).sum()
                total = ((chunk - chunk.mean()) ** 2).sum()
                values[end] = 1.0 - residual / total if total > 0 else np.nan
        out[column] = values
    return pd.DataFrame(out, index=frame.index)


def test_idxmax_counts_from_the_oldest_bar_in_the_window(evaluator):
    high = evaluator.matrices["high"]
    result = _evaluate(evaluator, "IdxMax($high, 10)")
    expected = high.rolling(10).apply(lambda a: float(np.argmax(a)), raw=True)
    pd.testing.assert_frame_equal(result, expected, check_names=False)


def test_idxmin_counts_from_the_oldest_bar_in_the_window(evaluator):
    low = evaluator.matrices["low"]
    result = _evaluate(evaluator, "IdxMin($low, 10)")
    expected = low.rolling(10).apply(lambda a: float(np.argmin(a)), raw=True)
    pd.testing.assert_frame_equal(result, expected, check_names=False)


def test_idxmax_is_nan_while_the_window_is_incomplete(evaluator):
    result = _evaluate(evaluator, "IdxMax($high, 10)")
    assert result.iloc[:9].isna().all().all()
    assert result.iloc[9:].notna().all().all()


def test_greater_and_less_are_elementwise(evaluator):
    matrices = evaluator.matrices
    assert _evaluate(evaluator, "Greater($open, $close)").equals(
        np.maximum(matrices["open"], matrices["close"])
    )
    assert _evaluate(evaluator, "Less($open, $close)").equals(
        np.minimum(matrices["open"], matrices["close"])
    )


def test_comparison_yields_one_and_zero_for_averaging(evaluator):
    close = evaluator.matrices["close"]
    result = _evaluate(evaluator, "$close>Ref($close, 1)")
    assert set(np.unique(result.to_numpy()[1:])) <= {0.0, 1.0}
    # The first bar has no predecessor, so the comparison is undefined, not False.
    assert result.iloc[0].isna().all()
    expected_ups = (close > close.shift(1)).sum().sum()
    assert result.sum().sum() == pytest.approx(float(expected_ups))


def test_log_of_non_positive_input_is_nan(evaluator):
    result = _evaluate(evaluator, "Log($close-$close)")
    assert result.isna().all().all()


def test_division_by_zero_is_nan_not_infinite(evaluator):
    result = _evaluate(evaluator, "$close/($close-$close)")
    assert not np.isinf(result.to_numpy()).any()
    assert result.isna().all().all()


# ===================== Warm-up and isolation =====================
def test_rolling_windows_require_full_history():
    panel = _panel(n_days=90)
    matrices = build_matrices(panel)
    result = Evaluator(matrices).evaluate(parse_expression("Mean($close, 60)"))
    assert result.iloc[:59].isna().all().all()
    assert result.iloc[59:].notna().all().all()


def test_one_symbol_history_never_leaks_into_another():
    """A symbol whose history is entirely NaN must not borrow the other's values."""
    panel = _panel(symbols=("A", "B"), n_days=90)
    panel.loc[panel["symbol"] == "B", "close"] = np.nan
    result = Evaluator(build_matrices(panel)).evaluate(parse_expression("Mean($close, 20)"))
    assert result["B"].isna().all()
    assert result["A"].iloc[19:].notna().all()


def test_factors_use_no_future_information():
    """Truncating the panel must not change factor values on the surviving dates."""
    panel = _panel(n_days=90)
    full, names, _ = compute_alpha158(panel)
    cutoff = sorted(panel["trade_date"].unique())[70]
    truncated, _, _ = compute_alpha158(panel[panel["trade_date"] <= cutoff].copy())

    key = ["symbol", "trade_date"]
    merged = full[full["trade_date"] <= cutoff].merge(truncated, on=key, suffixes=("_full", "_cut"))
    for name in names:
        np.testing.assert_allclose(
            merged[f"{name}_full"].to_numpy(dtype=float),
            merged[f"{name}_cut"].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"{name} changed when future bars were removed",
        )


# ===================== compute_alpha158 =====================
def test_compute_alpha158_skips_factors_whose_base_field_is_absent():
    """Our panel has no turnover, so VWAP0 must be dropped, not approximated."""
    _out, names, skipped = compute_alpha158(_panel())
    assert "VWAP0" not in names
    assert len(names) == 157
    # The reason travels with the skip so a report can explain the missing one.
    assert skipped == [{"name": "VWAP0", "reason": "needs $vwap"}]


def test_compute_alpha158_reports_no_skips_when_every_factor_computes():
    _out, names, skipped = compute_alpha158(_panel(), only=["MA20", "KMID"])
    assert names == ["KMID", "MA20"]
    assert skipped == []


def test_compute_alpha158_preserves_panel_rows_and_order():
    panel = _panel()
    out, names, _ = compute_alpha158(panel)
    assert len(out) == len(panel)
    pd.testing.assert_series_equal(out["symbol"], panel["symbol"])
    pd.testing.assert_series_equal(out["close"], panel["close"])
    assert set(names).issubset(out.columns)


def test_compute_alpha158_values_match_direct_evaluation():
    panel = _panel()
    out, _names, _ = compute_alpha158(panel)
    evaluator = Evaluator(build_matrices(panel))
    expected = evaluator.evaluate(parse_expression("Mean($close, 20)/$close"))
    for symbol in ("A", "B"):
        rows = out[out["symbol"] == symbol].set_index("trade_date")["MA20"]
        np.testing.assert_allclose(
            rows.to_numpy(dtype=float),
            expected[symbol].reindex(rows.index).to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
        )


def test_compute_alpha158_handles_unsorted_input():
    panel = _panel()
    shuffled = panel.sample(frac=1.0, random_state=5).reset_index(drop=True)
    out, _, _ = compute_alpha158(shuffled)
    reference, _, _ = compute_alpha158(panel)
    merged = out.merge(reference, on=["symbol", "trade_date"], suffixes=("_a", "_b"))
    np.testing.assert_allclose(
        merged["MA20_a"].to_numpy(dtype=float),
        merged["MA20_b"].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )


def test_only_restricts_evaluation_without_changing_values():
    panel = _panel()
    full, _, _ = compute_alpha158(panel)
    subset, names, _ = compute_alpha158(panel, only=["MA20", "KMID"])
    assert names == ["KMID", "MA20"]
    assert "RSV60" not in subset.columns
    for name in names:
        np.testing.assert_allclose(
            subset[name].to_numpy(dtype=float), full[name].to_numpy(dtype=float), rtol=1e-12
        )


def test_only_ignores_names_that_are_not_alpha158_factors():
    _out, names, _ = compute_alpha158(_panel(), only=["MA20", "not_a_factor"])
    assert names == ["MA20"]


def test_symbol_with_no_usable_prices_stays_in_the_panel_as_nan():
    panel = _panel(symbols=("A", "B"))
    panel.loc[panel["symbol"] == "B", "close"] = np.nan
    out, names, _ = compute_alpha158(panel)
    assert len(out) == len(panel)
    assert out.loc[out["symbol"] == "B", "MA20"].isna().all()
    assert out.loc[out["symbol"] == "A", "MA20"].notna().any()


def test_kbar_factors_are_defined_on_the_first_bar():
    """Candle-shape factors need no history, so they must not be NaN at the start."""
    out, _names, _ = compute_alpha158(_panel())
    assert out.groupby("symbol")["KMID"].first().notna().all()


# ===================== factor_catalog =====================
def test_catalog_covers_exactly_what_the_evaluator_computes():
    fields, names = alpha158_feature_config()
    catalog = factor_catalog()

    assert [entry["name"] for entry in catalog] == names
    assert [entry["expression"] for entry in catalog] == fields


def test_catalog_entries_are_fully_populated():
    for entry in factor_catalog():
        assert entry["family"], entry
        assert entry["description"], entry
        assert "{d}" not in entry["description"], entry
        assert entry["operator"] in entry["name"], entry


def test_catalog_reads_the_window_off_the_name():
    by_name = {entry["name"]: entry for entry in factor_catalog()}

    assert by_name["MA20"]["window"] == 20
    assert by_name["CORR60"]["window"] == 60
    assert "60 日" in by_name["CORR60"]["description"]
    # A kbar name ending in a digit is a variant, not a window.
    assert by_name["KMID2"]["window"] is None
    assert by_name["KSFT2"]["window"] is None


def test_catalog_separates_the_min_operator_from_the_low_price():
    by_name = {entry["name"]: entry for entry in factor_catalog()}

    # ``LOW`` the rolling operator is published as ``MIN``; ``LOW0`` is a price level.
    assert by_name["MIN60"]["expression"] == "Min($low, 60)/$close"
    assert by_name["MIN60"]["operator"] == "MIN"
    assert by_name["LOW0"]["expression"] == "$low/$close"
    assert by_name["LOW0"]["family"] == FAMILY_PRICE


def test_catalog_families_group_factors_a_user_would_compare():
    by_name = {entry["name"]: entry for entry in factor_catalog()}

    assert by_name["MA20"]["family"] == FAMILY_TREND
    assert by_name["STD20"]["family"] == FAMILY_VOLATILITY
    assert by_name["CORR10"]["family"] == FAMILY_VOLUME_PRICE
    assert by_name["VSUMP5"]["family"] == FAMILY_VOLUME
    assert by_name["KMID"]["family"] == FAMILY_KBAR
    # WVMA measures volume dispersion, so it belongs with volatility, not volume.
    assert by_name["WVMA30"]["family"] == FAMILY_VOLATILITY


def test_catalog_follows_a_custom_window_set():
    catalog = factor_catalog(windows=(7,))
    windows = {entry["window"] for entry in catalog if entry["family"] != FAMILY_KBAR}
    assert windows == {0, 7}


# ===================== Alpha101 operators =====================
def _matrix(values, index=("d1", "d2", "d3", "d4", "d5"), columns=("A", "B", "C")):
    return pd.DataFrame(values, index=list(index), columns=list(columns))


def _evaluate_matrix(frames, expression):
    return Evaluator(frames).evaluate(parse_expression(expression))


def test_sign_reads_the_direction():
    frame = _matrix([[-2.0, 0.0, 3.0]] * 5)
    result = _evaluate_matrix({"close": frame}, "Sign($close)")
    assert result.iloc[0].tolist() == [-1.0, 0.0, 1.0]


def test_delta_is_the_change_from_d_days_ago():
    frame = _matrix([[1.0, 10.0, 100.0], [2.0, 20.0, 100.0], [4.0, 40.0, 100.0], [7.0, 70.0, 100.0], [11.0, 110.0, 100.0]])
    result = _evaluate_matrix({"close": frame}, "Delta($close, 2)")
    assert result.iloc[2].tolist() == [3.0, 30.0, 0.0]
    assert np.isnan(result.iloc[0]).all()
    assert np.isnan(result.iloc[1]).all()


def test_csrank_matches_pandas_row_wise_pct_rank():
    frame = _matrix(
        [[1.0, 3.0, 2.0], [4.0, np.nan, 1.0], [np.nan, np.nan, np.nan], [5.0, 5.0, 5.0], [2.0, 1.0, 3.0]]
    )
    result = _evaluate_matrix({"close": frame}, "CSRank($close)")
    expected = frame.rank(axis=1, pct=True)
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_where_keeps_a_missing_condition_missing():
    frame = _matrix([[99.0, 100.0, 101.0], [np.nan, 60.0, 40.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    result = _evaluate_matrix({"close": frame}, "Where($close > 100, 1, -1)")
    assert result.iloc[0].tolist() == [-1.0, -1.0, 1.0]
    assert np.isnan(result.iloc[1, 0])
    assert result.iloc[1, 1:].tolist() == [-1.0, -1.0]


def test_ternary_parses_and_nests_right_associatively():
    frame = _matrix([[40.0, 60.0, 110.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    evaluator = Evaluator({"close": frame})
    assert parse_expression("$close > 100 ? 1 : 0")[0] == "ternary"
    result = evaluator.evaluate(parse_expression("$close > 100 ? 1 : $close > 50 ? 2 : 3"))
    assert result.iloc[0].tolist() == [3.0, 2.0, 1.0]


def test_signed_power_keeps_the_sign():
    frame = _matrix([[-4.0, -1.0, 0.0], [1.0, 4.0, 9.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    result = _evaluate_matrix({"close": frame}, "SignedPower($close, 2)")
    assert result.iloc[0].tolist() == [-16.0, -1.0, 0.0]
    assert result.iloc[1].tolist() == [1.0, 16.0, 81.0]


def test_ts_argmax_counts_back_from_today():
    frame = _matrix([[1.0, 5.0, 9.0], [2.0, 4.0, 8.0], [3.0, 3.0, 7.0], [2.0, 2.0, 6.0], [1.0, 1.0, 5.0]])
    result = _evaluate_matrix({"close": frame}, "TsArgMax($close, 5)")
    # Column A peaks at position 2 (0 = oldest), i.e. two bars back from today.
    assert result.iloc[-1, 0] == 2.0
    # Column B peaks at position 0, four bars back.
    assert result.iloc[-1, 1] == 4.0
    assert np.isnan(result.iloc[:4].to_numpy()).all()


def test_ts_argmax_rejects_a_window_with_a_hole():
    frame = _matrix([[1.0, 1.0, 1.0], [np.nan, 2.0, 2.0], [3.0, 3.0, 3.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]])
    result = _evaluate_matrix({"close": frame}, "TsArgMax($close, 5)")
    assert np.isnan(result.iloc[-1, 0])
    assert not np.isnan(result.iloc[-1, 1])


def test_ts_argmax_breaks_ties_toward_the_most_recent_bar():
    # The maximum repeats on the last three bars. Alpha101/Tushare count the
    # newest occurrence as "0 days ago", not the oldest.
    frame = _matrix([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])
    result = _evaluate_matrix({"close": frame}, "TsArgMax($close, 5)")
    assert result.iloc[-1, 0] == 0.0


def test_covariance_matches_pandas_rolling_cov():
    x = _matrix([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0], [4.0, 2.0], [5.0, 1.0]], columns=("A", "B"))
    y = _matrix([[2.0, 1.0], [4.0, 2.0], [5.0, 3.0], [4.0, 4.0], [5.0, 5.0]], columns=("A", "B"))
    result = _evaluate_matrix({"x": x, "y": y}, "Covariance($x, $y, 3)")
    expected = x.rolling(3, min_periods=3).cov(y, pairwise=False)
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_build_matrices_accepts_custom_fields():
    panel = _panel()
    panel["returns"] = panel["close"] / 100.0 - 1.0
    matrices = build_matrices(panel, fields=("close", "returns"))
    assert set(matrices) == {"close", "returns"}


def test_compute_expressions_skips_a_factor_with_a_missing_field():
    panel = _panel()
    result, computed, skipped = compute_expressions(
        panel,
        [("needs_vwap", "$close/$vwap"), ("plain", "$close")],
        fields=("close", "vwap"),
    )
    assert computed == ["plain"]
    assert skipped == [{"name": "needs_vwap", "reason": "needs $vwap"}]
    pd.testing.assert_series_equal(
        result["plain"].reset_index(drop=True),
        panel["close"].reset_index(drop=True),
        check_names=False,
    )
