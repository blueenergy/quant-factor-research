import json
import math
import sys

import numpy as np
import pandas as pd
import pytest

from analysis.core.score_backtester import BacktestConfig
from analysis.core.trade_fee_model import TradeFeeModel
from analysis.factor.alpha_factor_backtest import (
    ALPHA158_MIN_WARMUP_DAYS,
    ALPHA_FACTOR_COLUMNS,
    PROGRESS_STAGES,
    STAGE_BUILDING_DATASET,
    STAGE_COST_ADJUSTMENT,
    STAGE_LOADING_PRICES,
    CostAssumptions,
    FactorBacktestError,
    FactorBacktestParams,
    FactorBudgetError,
    PanelFactorBacktester,
    PointInTimeUniverse,
    build_dataset,
    build_price_cache,
    compute_alpha_factors,
    compute_factor_set,
    evaluate,
    load_mongo_panel,
    net_return_table,
    normalize_panel,
    panel_to_score_records,
    run_factor_backtest,
    screen_factors,
    screen_rank_ic,
    shift_calendar_days,
)
from analysis.factor.ic_metrics import per_date_rank_ic


def _trade_date(day_index: int) -> str:
    """Sequential 8-digit dates; the backtester treats them as opaque sortable keys."""
    return f"2025{day_index // 28 + 1:02d}{day_index % 28 + 1:02d}"


def _raw_panel(symbols=("000001.SZ",), n_days=40) -> pd.DataFrame:
    """Deterministic panel: open rises 0.10 per day, each symbol offset by 1.00."""
    rows = []
    for offset, symbol in enumerate(symbols):
        for day in range(n_days):
            base = 10.0 + offset + day * 0.1
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": _trade_date(day),
                    "open": base,
                    "high": base + 0.5,
                    "low": base - 0.5,
                    "close": base + 0.2,
                    "vol": 1000 + day * 10 + offset * 100,
                }
            )
    return pd.DataFrame(rows)


# ===================== normalize_panel =====================
def test_normalize_panel_maps_aliases():
    panel = normalize_panel(_raw_panel())
    assert "symbol" in panel.columns
    assert "volume" in panel.columns
    assert "ts_code" not in panel.columns
    assert panel["trade_date"].iloc[0] == "20250101"


def test_normalize_panel_strips_date_separators():
    raw = _raw_panel()
    raw["trade_date"] = "2025-01-02"
    assert normalize_panel(raw)["trade_date"].iloc[0] == "20250102"


def test_normalize_panel_rejects_missing_columns():
    raw = _raw_panel().drop(columns=["close"])
    with pytest.raises(ValueError, match="close"):
        normalize_panel(raw)


def test_normalize_panel_drops_non_positive_close():
    raw = _raw_panel(n_days=5)
    raw.loc[2, "close"] = 0
    assert len(normalize_panel(raw)) == 4


# ===================== compute_alpha_factors =====================
def test_warmup_rows_stay_nan_instead_of_filled():
    panel = compute_alpha_factors(normalize_panel(_raw_panel(n_days=40)))
    # alpha2 needs the previous close, alpha5 a 20-day mean.
    assert math.isnan(panel["alpha2"].iloc[0])
    assert panel["alpha5"].iloc[:19].isna().all()
    assert panel["alpha5"].iloc[19:].notna().all()


def test_factors_do_not_leak_across_symbols():
    panel = compute_alpha_factors(normalize_panel(_raw_panel(symbols=("000001.SZ", "000002.SZ"))))
    second = panel[panel["symbol"] == "000002.SZ"].reset_index(drop=True)
    # The first bar of the second symbol must not borrow the first symbol's tail.
    assert math.isnan(second["alpha2"].iloc[0])
    assert second["alpha7"].iloc[:5].isna().all()


def test_zero_range_bar_yields_nan_not_inflated_ratio():
    raw = _raw_panel(n_days=10)
    flat = raw["close"].iloc[5]
    raw.loc[5, ["open", "high", "low", "close"]] = flat
    panel = compute_alpha_factors(normalize_panel(raw))

    assert math.isnan(panel["alpha1"].iloc[5])
    assert math.isnan(panel["alpha3"].iloc[5])
    # Neighbouring bars keep finite values.
    assert np.isfinite(panel["alpha1"].iloc[4])


def test_all_expected_factor_columns_exist():
    panel = compute_alpha_factors(normalize_panel(_raw_panel()))
    for column in ALPHA_FACTOR_COLUMNS:
        assert column in panel.columns


# ===================== panel -> backtester inputs =====================
def test_score_records_use_the_bar_own_date():
    panel = compute_alpha_factors(normalize_panel(_raw_panel(n_days=30)))
    records = panel_to_score_records(panel, ["alpha1"])
    assert records[0]["score_date"] == panel["trade_date"].iloc[0]
    assert records[0]["symbol"] == "000001.SZ"


def test_score_records_convert_nan_to_none():
    panel = compute_alpha_factors(normalize_panel(_raw_panel(n_days=30)))
    records = panel_to_score_records(panel, ["alpha1", "alpha5"])
    assert records[0]["alpha5"] is None
    assert isinstance(records[0]["alpha1"], float)


def test_price_cache_is_sorted_and_falls_back_to_close_for_nan_open():
    raw = _raw_panel(n_days=6)
    raw.loc[3, "open"] = np.nan
    panel = normalize_panel(raw).sample(frac=1.0, random_state=0)

    cache = build_price_cache(panel)
    rows = cache["000001.SZ"]
    assert [row["trade_date"] for row in rows] == sorted(row["trade_date"] for row in rows)
    assert rows[3]["open"] == rows[3]["close"]


def test_score_row_from_doc_passes_arbitrary_factor_columns():
    config = BacktestConfig(score_columns=("alpha1", "alpha4"))
    backtester = PanelFactorBacktester({}, config)
    row = backtester._score_row_from_doc({"alpha1": "0.25", "alpha4": None, "composite_score": 90})

    assert row == {"alpha1": 0.25, "alpha4": None}
    assert "composite_score" not in row


def test_future_price_lookup_matches_the_base_class_slice():
    panel = normalize_panel(_raw_panel(n_days=30))
    cache = build_price_cache(panel)
    backtester = PanelFactorBacktester(cache, BacktestConfig(score_columns=("alpha1",)))
    history = cache["000001.SZ"]

    for score_date in (_trade_date(0), _trade_date(7), _trade_date(29)):
        fast = backtester._load_future_prices("000001.SZ", score_date, 6)
        expected = backtester._slice_future_prices(history, score_date, 6)
        assert fast == expected


def test_future_price_lookup_returns_empty_for_unknown_symbol():
    backtester = PanelFactorBacktester({}, BacktestConfig(score_columns=("alpha1",)))
    assert backtester._load_future_prices("999999.SZ", "20250101", 5) == []


# ===================== end to end =====================
def test_dataset_enters_at_next_open_without_lookahead():
    panel = compute_alpha_factors(normalize_panel(_raw_panel(n_days=30)))
    config = BacktestConfig(horizons=(1,), score_columns=("alpha1",), min_names_per_date=1)
    dataset = build_dataset(panel, ["alpha1"], config)

    row = dataset[dataset["score_date"] == _trade_date(10)].iloc[0]
    # Entry fills at the next session's open, exit one horizon later at its open.
    assert row["entry_date"] == _trade_date(11)
    assert row["entry_price"] == pytest.approx(10.0 + 11 * 0.1)
    expected = (10.0 + 12 * 0.1) / (10.0 + 11 * 0.1) - 1
    assert row["forward_return_1d"] == pytest.approx(expected)


def test_evaluate_reports_ic_and_quantiles_per_horizon():
    symbols = tuple(f"00000{i}.SZ" for i in range(8))
    panel = compute_alpha_factors(normalize_panel(_raw_panel(symbols=symbols, n_days=40)))
    config = BacktestConfig(
        horizons=(1, 5),
        quantiles=4,
        score_columns=("alpha1", "alpha5"),
        min_names_per_date=5,
    )
    dataset = build_dataset(panel, ["alpha1", "alpha5"], config)
    report = evaluate(dataset, ["alpha1", "alpha5"], config)

    assert report["symbols"] == len(symbols)
    assert set(report["factors"]) == {"alpha1", "alpha5"}
    for horizon in ("1", "5"):
        block = report["factors"]["alpha1"][horizon]
        assert block["ic"]["observations"] >= 1
        assert block["nonoverlap"]["step"] == int(horizon)
        assert block["nonoverlap"]["observations"] >= 1
        assert "mde80" in block["nonoverlap"]
        assert "top_minus_bottom_spread" in block["quantiles"]


def test_evaluate_flags_empty_dataset():
    config = BacktestConfig(score_columns=("alpha1",))
    assert "error" in evaluate(pd.DataFrame(), ["alpha1"], config)


# ===================== cost model =====================
def test_round_trip_rate_includes_fees_and_both_slippage_legs():
    costs = CostAssumptions(fee=TradeFeeModel.from_params(), slippage_bps_one_way=10.0)
    # Default fees: 1bp buy + 1bp sell commission + 5bp stamp tax = 7bp, plus 20bp slippage.
    assert costs.round_trip_rate() == pytest.approx(0.0007 + 0.0020)


def test_round_trip_rate_scales_with_turnover():
    fee = TradeFeeModel.from_params()
    half = CostAssumptions(fee=fee, slippage_bps_one_way=10.0, turnover_per_rebalance=0.5)
    full = CostAssumptions(fee=fee, slippage_bps_one_way=10.0, turnover_per_rebalance=1.0)
    assert half.round_trip_rate() == pytest.approx(full.round_trip_rate() / 2)


def test_net_return_table_charges_one_round_trip_long_and_two_long_short():
    topk = pd.DataFrame(
        [
            {
                "score_column": "alpha5",
                "horizon": 10,
                "top_k": 20,
                "mean_top_vs_universe": 0.01,
                "mean_top_minus_bottom": 0.02,
                "observations": 100,
            }
        ]
    )
    costs = CostAssumptions(fee=TradeFeeModel.from_params(), slippage_bps_one_way=10.0)
    net = net_return_table(topk, costs)
    row = net.iloc[0]
    cost = costs.round_trip_rate()

    assert row["long_excess_net"] == pytest.approx(0.01 - cost)
    assert row["long_short_net"] == pytest.approx(0.02 - 2 * cost)
    # 10-day rebalancing compounds 25.2 times a year.
    assert row["long_short_net_annual"] == pytest.approx((0.02 - 2 * cost) * 252 / 10)


def test_net_return_table_handles_empty_input():
    costs = CostAssumptions(fee=TradeFeeModel.from_params())
    assert net_return_table(pd.DataFrame(), costs).empty


# ===================== Mongo source =====================
class _FakeConstituents:
    """Stand-in for quant_data.index_constituents."""

    def __init__(self, docs):
        self.docs = list(docs)

    def distinct(self, field, query=None):
        query = query or {}
        return sorted(
            {d.get(field) for d in self.docs if all(d.get(k) == v for k, v in query.items())}
        )

    def find(self, query=None, projection=None):
        query = query or {}
        return [d for d in self.docs if all(d.get(k) == v for k, v in query.items())]

    def find_one(self, query=None, projection=None, sort=None):
        matches = [d for d in self.docs if _matches(d, query or {})]
        if not matches:
            return None
        for key, direction in reversed(sort or []):
            matches.sort(key=lambda d: d.get(key) or "", reverse=direction < 0)
        return matches[0]


def _matches(doc, query):
    for key, cond in query.items():
        value = doc.get(key)
        if isinstance(cond, dict):
            if "$lte" in cond and not (value and value <= cond["$lte"]):
                return False
        elif value != cond:
            return False
    return True


def _constituent_docs():
    """Two quarterly snapshots; AAA is dropped and CCC added in the second."""
    first = [("AAA.SZ", "20250101"), ("BBB.SZ", "20250101")]
    second = [("BBB.SZ", "20250401"), ("CCC.SZ", "20250401")]
    return [
        {"index_code": "hs300", "symbol": sym, "weight_trade_date": wtd, "update_date": wtd}
        for sym, wtd in first + second
    ]


def test_shift_calendar_days():
    assert shift_calendar_days("20250301", -60) == "20241231"
    assert shift_calendar_days("20250101", 1) == "20250102"


def test_point_in_time_universe_switches_snapshot_on_the_rebalance_date():
    universe = PointInTimeUniverse(_FakeConstituents(_constituent_docs()), "hs300", "20250101", "20250630")

    assert universe.symbols == {"AAA.SZ", "BBB.SZ", "CCC.SZ"}
    assert universe.members_on("20250315") == {"AAA.SZ", "BBB.SZ"}
    assert universe.members_on("20250401") == {"BBB.SZ", "CCC.SZ"}
    # A dropped name must not linger after the switch.
    assert "AAA.SZ" not in universe.members_on("20250501")


def test_point_in_time_universe_uses_snapshot_in_force_before_the_window():
    universe = PointInTimeUniverse(_FakeConstituents(_constituent_docs()), "hs300", "20250301", "20250331")
    assert universe.members_on("20250301") == {"AAA.SZ", "BBB.SZ"}


def test_point_in_time_universe_is_empty_for_unknown_index():
    universe = PointInTimeUniverse(_FakeConstituents(_constituent_docs()), "csi999", "20250101", "20250630")
    assert universe.symbols == set()
    assert universe.members_on("20250101") == set()


class _FakeAccessor:
    """Returns hfq frames shaped like AdjustedPriceDataAccess.load_adjusted_ohlc."""

    def __init__(self, per_symbol):
        self.per_symbol = per_symbol
        self.calls = []

    def load_adjusted_ohlc(self, symbol, start_date, end_date, adjust="hfq"):
        self.calls.append((symbol, start_date, end_date, adjust))
        frame = self.per_symbol.get(symbol)
        return frame.copy() if frame is not None else pd.DataFrame()


def _hfq_frame(n_days=8, base=20.0):
    return pd.DataFrame(
        [
            {
                "trade_date": _trade_date(day),
                "open": base + day,
                "high": base + day + 0.5,
                "low": base + day - 0.5,
                "close": base + day + 0.2,
                "volume": 1000 + day,
            }
            for day in range(n_days)
        ]
    )


@pytest.fixture
def mongo_panel(monkeypatch):
    import analysis.factor.alpha_factor_backtest as module

    accessor = _FakeAccessor({sym: _hfq_frame() for sym in ("AAA.SZ", "BBB.SZ", "CCC.SZ")})
    monkeypatch.setattr(module, "AdjustedPriceDataAccess", lambda db=None: accessor)
    db = {"index_constituents": _FakeConstituents(_constituent_docs())}
    return accessor, db


def test_mongo_panel_flags_membership_point_in_time(mongo_panel):
    accessor, db = mongo_panel
    # The window spans both snapshots, but the fake prices only cover January.
    panel = load_mongo_panel("hs300", "20250101", "20250630", db=db, workers=2)

    scored = panel[panel["in_pool"]]
    # CCC only joins in the April snapshot, so it must never be scored in January.
    assert set(scored["symbol"]) == {"AAA.SZ", "BBB.SZ"}
    # Its prices are still loaded, because a held position may need to exit.
    assert "CCC.SZ" in set(panel["symbol"])


def test_mongo_panel_keeps_warmup_rows_out_of_the_cross_section(mongo_panel):
    accessor, db = mongo_panel
    start = _trade_date(3)
    panel = load_mongo_panel("hs300", start, "20250131", db=db, workers=2)

    assert panel["trade_date"].min() < start
    assert panel[panel["in_pool"]]["trade_date"].min() == start


def test_mongo_panel_requests_hfq_prices_with_warmup(mongo_panel):
    accessor, db = mongo_panel
    load_mongo_panel("hs300", "20250301", "20250630", db=db, warmup_days=60, workers=2)

    symbols = {call[0] for call in accessor.calls}
    assert symbols == {"AAA.SZ", "BBB.SZ", "CCC.SZ"}
    for _symbol, fetch_start, end_date, adjust in accessor.calls:
        assert adjust == "hfq"
        assert fetch_start == "20241231"
        assert end_date == "20250630"


def test_mongo_panel_raises_for_index_without_snapshots(mongo_panel):
    _accessor, db = mongo_panel
    with pytest.raises(ValueError, match="no index_constituents snapshot"):
        load_mongo_panel("csi999", "20250101", "20250131", db=db)


def test_score_records_respect_the_in_pool_flag():
    panel = compute_alpha_factors(normalize_panel(_raw_panel(symbols=("AAA.SZ", "BBB.SZ"), n_days=25)))
    panel["in_pool"] = panel["symbol"] == "AAA.SZ"

    records = panel_to_score_records(panel, ["alpha1"])
    assert {r["symbol"] for r in records} == {"AAA.SZ"}


def test_price_cache_ignores_the_in_pool_flag():
    panel = normalize_panel(_raw_panel(symbols=("AAA.SZ", "BBB.SZ"), n_days=10))
    panel["in_pool"] = panel["symbol"] == "AAA.SZ"

    # Excluded names still need prices so open positions can be exited.
    assert set(build_price_cache(panel)) == {"AAA.SZ", "BBB.SZ"}


# ===================== factor sets =====================
def test_alpha8_factor_set_returns_the_legacy_columns():
    panel, factors, skipped = compute_factor_set(normalize_panel(_raw_panel(n_days=30)), "alpha8")
    assert factors == list(ALPHA_FACTOR_COLUMNS)
    assert set(ALPHA_FACTOR_COLUMNS).issubset(panel.columns)
    assert skipped == []


def test_alpha158_factor_set_returns_every_computable_factor():
    panel, factors, skipped = compute_factor_set(normalize_panel(_raw_panel(n_days=80)), "alpha158")
    # VWAP0 needs turnover, which the OHLCV panel does not carry.
    assert len(factors) == 157
    assert {"KMID", "MA20", "VSUMD60"}.issubset(panel.columns)
    assert skipped == [{"name": "VWAP0", "reason": "needs $vwap"}]


def test_unknown_factor_set_is_rejected():
    with pytest.raises(ValueError, match="unknown factor set"):
        compute_factor_set(normalize_panel(_raw_panel(n_days=10)), "alpha360")


# ===================== IC screening =====================
def _screening_panel(n_symbols=12, n_days=45, seed=4):
    """Panel with enough cross-sectional spread for IC to be defined each day."""
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_symbols):
        price = 10.0 + index
        for day in range(n_days):
            price *= 1.0 + rng.normal(0.0, 0.03)
            rows.append(
                {
                    "symbol": f"{index:06d}.SZ",
                    "trade_date": _trade_date(day),
                    "open": price,
                    "high": price * 1.02,
                    "low": price * 0.98,
                    "close": price * 1.01,
                    "volume": float(rng.integers(1000, 9000)),
                }
            )
    return compute_alpha_factors(normalize_panel(pd.DataFrame(rows)))


def _screening_dataset(panel, factors, horizon=5):
    config = BacktestConfig(
        horizons=(horizon,), score_columns=tuple(factors), min_names_per_date=5
    )
    return build_dataset(panel, factors, config), config


def test_screen_rank_ic_matches_the_reference_implementation():
    factors = ["alpha1", "alpha3", "alpha4"]
    panel = _screening_panel()
    dataset, _ = _screening_dataset(panel, factors)

    fast = screen_rank_ic(dataset, factors, "forward_return_5d", min_names=5)
    for factor in factors:
        reference = per_date_rank_ic(dataset, factor, "forward_return_5d", 5)
        assert not reference.empty
        pd.testing.assert_frame_equal(
            fast[factor].reset_index(drop=True), reference.reset_index(drop=True)
        )


def test_screen_rank_ic_matches_reference_for_factors_with_gaps():
    """A factor with NaNs must be ranked on its own rows, not the full cross-section."""
    factors = ["alpha1", "alpha5"]
    panel = _screening_panel()
    dataset, _ = _screening_dataset(panel, factors)
    assert dataset["alpha5"].isna().any()

    fast = screen_rank_ic(dataset, factors, "forward_return_5d", min_names=5)
    reference = per_date_rank_ic(dataset, "alpha5", "forward_return_5d", 5)
    pd.testing.assert_frame_equal(
        fast["alpha5"].reset_index(drop=True), reference.reset_index(drop=True)
    )


def test_screen_rank_ic_drops_dates_below_the_minimum_cross_section():
    panel = _screening_panel(n_symbols=12)
    dataset, _ = _screening_dataset(panel, ["alpha1"])
    wide = screen_rank_ic(dataset, ["alpha1"], "forward_return_5d", min_names=5)
    narrow = screen_rank_ic(dataset, ["alpha1"], "forward_return_5d", min_names=20)
    assert not wide["alpha1"].empty
    assert narrow["alpha1"].empty


def test_screen_rank_ic_ignores_columns_absent_from_the_dataset():
    panel = _screening_panel()
    dataset, _ = _screening_dataset(panel, ["alpha1"])
    result = screen_rank_ic(dataset, ["alpha1", "nope"], "forward_return_5d", min_names=5)
    assert not result["alpha1"].empty
    assert result["nope"].empty


def test_screen_factors_ranks_by_absolute_t_statistic():
    factors = ["alpha1", "alpha3", "alpha4"]
    panel = _screening_panel()
    dataset, _ = _screening_dataset(panel, factors)

    screen = screen_factors(dataset, factors, horizon=5, min_names=5)
    assert list(screen.columns[:3]) == ["factor", "horizon", "ic_mean"]
    assert {"t_nonoverlap", "nonoverlap_observations", "mde80"} <= set(screen.columns)
    magnitudes = screen["t_stat"].abs().dropna().tolist()
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_build_dataset_carries_every_factor_column():
    """The placeholder used for forward returns must not survive into the output."""
    factors = ["alpha1", "alpha3", "alpha4"]
    panel = _screening_panel()
    dataset, _ = _screening_dataset(panel, factors)
    assert set(factors).issubset(dataset.columns)
    assert "__scored__" not in dataset.columns
    assert dataset["forward_return_5d"].notna().any()


def test_build_dataset_factor_values_come_from_the_scoring_bar():
    panel = _screening_panel()
    dataset, _ = _screening_dataset(panel, ["alpha1"])
    row = dataset.iloc[0]
    source = panel[
        (panel["symbol"] == row["symbol"]) & (panel["trade_date"] == row["score_date"])
    ].iloc[0]
    assert row["alpha1"] == pytest.approx(source["alpha1"])


# ===================== FactorBacktestParams =====================
def _mongo_params(**overrides):
    base = dict(source="mongo", index_code="hs300", start_date="20240101", end_date="20241231")
    base.update(overrides)
    return FactorBacktestParams(**base)


def test_params_normalize_horizons_and_top_k():
    params = _mongo_params(horizons=(20, 5, 5, 1), top_k=(50, 20, 20))
    assert params.horizons == (1, 5, 20)
    assert params.top_k == (20, 50)


def test_params_raise_alpha158_warmup_to_cover_the_longest_window():
    params = _mongo_params(factor_set="alpha158", warmup_days=60)
    assert params.warmup_days == ALPHA158_MIN_WARMUP_DAYS
    # alpha8 has no 60-bar window, so its warm-up is left alone.
    assert _mongo_params(factor_set="alpha8", warmup_days=60).warmup_days == 60


def test_params_keep_an_explicit_warmup_longer_than_the_floor():
    assert _mongo_params(factor_set="alpha158", warmup_days=400).warmup_days == 400


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"source": "parquet"}, "source must be one of"),
        ({"factor_set": "alpha42"}, "factor_set must be one of"),
        ({"start_date": "2024-01-01"}, "start_date must be YYYYMMDD"),
        ({"start_date": "20241231", "end_date": "20240101"}, "is after end_date"),
        ({"horizons": ()}, "at least one horizon"),
        ({"horizons": (0, 5)}, "horizons must be >= 1"),
        ({"screen_horizon": 3}, "not among horizons"),
        ({"quantiles": 1}, "quantiles must be >= 2"),
        ({"quantiles": 5, "min_names": 4}, "cannot fill 5 quantile buckets"),
        ({"top_k": (0,)}, "top_k values must be >= 1"),
        ({"screen_top": 0}, "screen_top must be >= 1"),
        ({"warmup_days": -1}, "warmup_days must be >= 0"),
        ({"workers": 0}, "workers must be >= 1"),
        ({"slippage_bps": -1.0}, "slippage_bps must be >= 0"),
        ({"turnover": 0.0}, "turnover must be in"),
        ({"turnover": 1.5}, "turnover must be in"),
        ({"memory_budget_gb": 0}, "memory_budget_gb must be > 0"),
    ],
)
def test_params_reject_invalid_values(overrides, message):
    with pytest.raises(FactorBacktestError, match=message):
        _mongo_params(**overrides)


def test_params_require_the_fields_their_source_reads():
    with pytest.raises(FactorBacktestError, match="requires: index_code, start_date, end_date"):
        FactorBacktestParams(source="mongo")
    with pytest.raises(FactorBacktestError, match="requires a panel path"):
        FactorBacktestParams(source="panel")


def test_params_reject_factor_names_outside_the_set():
    with pytest.raises(FactorBacktestError, match="unknown alpha158 factors: NOPE, ZZZ99"):
        _mongo_params(factor_set="alpha158", factors=("MA20", "ZZZ99", "NOPE"))
    with pytest.raises(FactorBacktestError, match="unknown alpha8 factors: MA20"):
        _mongo_params(factor_set="alpha8", factors=("alpha1", "MA20"))


def test_params_accept_real_factor_names():
    assert _mongo_params(factor_set="alpha158", factors=["MA20", "CORR10"]).factors == ("MA20", "CORR10")


def test_params_from_mapping_rejects_keys_the_pipeline_would_ignore():
    payload = {"source": "mongo", "index_code": "hs300", "start_date": "20240101",
               "end_date": "20241231", "factorset": "alpha158"}
    with pytest.raises(FactorBacktestError, match="unknown parameters: factorset"):
        FactorBacktestParams.from_mapping(payload)


def test_params_survive_a_json_round_trip():
    original = _mongo_params(factor_set="alpha158", factors=("MA20",), horizons=(1, 10))
    restored = FactorBacktestParams.from_mapping(json.loads(json.dumps(original.to_dict())))
    assert restored == original


def test_params_factor_count_reflects_the_set_or_the_shortlist():
    assert _mongo_params(factor_set="alpha8").factor_count == len(ALPHA_FACTOR_COLUMNS)
    assert _mongo_params(factor_set="alpha158").factor_count == 158
    assert _mongo_params(factor_set="alpha158", factors=("MA20", "CORR10")).factor_count == 2


# ===================== compute budget =====================
def test_budget_rejects_the_whole_set_over_a_long_history():
    params = _mongo_params(factor_set="alpha158", index_code="csi1000",
                           start_date="20100101", end_date="20260101")
    with pytest.raises(FactorBudgetError, match="over the 6.0 GB budget"):
        params.check_budget()


def test_budget_admits_the_two_stage_workflow_it_recommends():
    # Stage 1: every factor, short window. Stage 2: a shortlist, long window.
    wide = _mongo_params(factor_set="alpha158", index_code="csi1000",
                         start_date="20250101", end_date="20260701")
    deep = _mongo_params(factor_set="alpha158", index_code="csi1000", factors=("MA20", "CORR10"),
                         start_date="20100101", end_date="20260101")
    assert wide.check_budget() < wide.memory_budget_gb
    assert deep.check_budget() < deep.memory_budget_gb


def test_budget_scales_with_universe_size():
    def estimate(index_code):
        return _mongo_params(factor_set="alpha158", index_code=index_code,
                             start_date="20240101", end_date="20250101",
                             memory_budget_gb=100).estimated_memory_gb()

    assert estimate("csi2000") > estimate("csi1000") > estimate("hs300")


def test_budget_is_skipped_when_the_panel_size_is_unknowable(tmp_path):
    csv_params = FactorBacktestParams(source="panel", panel=str(tmp_path / "p.csv"),
                                      factor_set="alpha158")
    assert csv_params.estimated_memory_gb() is None
    assert csv_params.check_budget() is None
    # An index nobody sized cannot be guarded either.
    assert _mongo_params(index_code="sse50", factor_set="alpha158").estimated_memory_gb() is None


def test_budget_counts_the_warmup_that_gets_loaded():
    short = _mongo_params(start_date="20250101", end_date="20250201", warmup_days=0)
    long_warmup = _mongo_params(start_date="20250101", end_date="20250201", warmup_days=365)
    assert long_warmup.years > short.years


# ===================== run_factor_backtest =====================
@pytest.fixture
def panel_csv(tmp_path):
    path = tmp_path / "panel.csv"
    _raw_panel(symbols=[f"{600000 + i:06d}.SH" for i in range(40)], n_days=60).to_csv(path, index=False)
    return str(path)


def _run_params(panel_csv, **overrides):
    base = dict(source="panel", panel=panel_csv, factor_set="alpha8", horizons=(1, 5),
                quantiles=5, min_names=10, top_k=(5,))
    base.update(overrides)
    return FactorBacktestParams(**base)


def test_run_factor_backtest_returns_a_json_serializable_report(panel_csv):
    report = run_factor_backtest(_run_params(panel_csv))

    assert set(report) >= {"factors", "screen", "source", "cost_assumptions", "net_returns", "params"}
    assert report["source"]["factor_set"] == "alpha8"
    assert report["source"]["point_in_time_universe"] is False
    assert report["params"]["horizons"] == [1, 5]
    # A queue worker stores this document as-is, so nothing may be a numpy scalar.
    json.dumps(report)


def test_run_factor_backtest_reports_progress_in_pipeline_order(panel_csv):
    seen = []
    run_factor_backtest(_run_params(panel_csv), lambda stage, message: seen.append(stage))

    assert set(seen) <= set(PROGRESS_STAGES)
    first_seen = [stage for i, stage in enumerate(seen) if stage not in seen[:i]]
    assert first_seen == [s for s in PROGRESS_STAGES if s in first_seen]
    assert first_seen[0] == STAGE_LOADING_PRICES
    assert first_seen[-1] == STAGE_COST_ADJUSTMENT


def test_run_factor_backtest_passes_messages_alongside_the_stage(panel_csv):
    seen = []
    run_factor_backtest(_run_params(panel_csv), lambda stage, message: seen.append((stage, message)))
    building = [m for stage, m in seen if stage == STAGE_BUILDING_DATASET]
    assert any("horizons=(1, 5)" in m for m in building)


def test_run_factor_backtest_works_without_a_progress_callback(panel_csv):
    assert run_factor_backtest(_run_params(panel_csv), None)["factors"]


def test_run_factor_backtest_needs_no_scipy(panel_csv, monkeypatch):
    """The worker image installs requirements-docker.txt, which omits scipy.

    pandas' ``corr(method="spearman")`` imports scipy.stats on the spot, so a
    Spearman written that way runs here and dies in the image. The host does
    have scipy, so blocking the import is the only way this suite can tell.
    """
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.stats", None)

    report = run_factor_backtest(_run_params(panel_csv), None)

    assert report["factors"]
    assert report["net_returns"]


def test_run_factor_backtest_enforces_the_budget_before_reading_prices():
    params = _mongo_params(factor_set="alpha158", index_code="csi1000",
                           start_date="20100101", end_date="20260101")
    # No db is wired up, so anything past the guard would fail differently.
    with pytest.raises(FactorBudgetError):
        run_factor_backtest(params, db={})


def test_run_factor_backtest_rejects_a_factor_the_panel_cannot_produce(panel_csv):
    # VWAP0 needs $vwap, which a panel without turnover cannot supply.
    params = _run_params(panel_csv, factor_set="alpha158", factors=("VWAP0",))
    with pytest.raises(FactorBacktestError, match="none of the requested factors exist"):
        run_factor_backtest(params)


def test_run_factor_backtest_reports_a_factor_the_panel_cannot_supply(panel_csv):
    """A partly computable request must name what it dropped, and why."""
    report = run_factor_backtest(
        _run_params(panel_csv, factor_set="alpha158", factors=("MA20", "VWAP0"), screen_top=20)
    )

    assert "VWAP0" not in report["evaluated_columns"]
    assert report["skipped_factors"] == [{"name": "VWAP0", "reason": "needs $vwap"}]
    json.dumps(report)


def test_run_factor_backtest_skips_nothing_when_the_whole_set_computes(panel_csv):
    assert run_factor_backtest(_run_params(panel_csv))["skipped_factors"] == []


def test_run_factor_backtest_narrows_diagnostics_to_the_screen_survivors(panel_csv):
    params = _run_params(panel_csv, screen_top=3)
    report = run_factor_backtest(params)

    assert len(report["factors"]) == 3
    # The screen itself still covers every factor, so the UI can show the full ranking.
    assert len(report["screen"]["ranked"]) == len(ALPHA_FACTOR_COLUMNS)
    assert list(report["factors"]) == [row["factor"] for row in report["screen"]["ranked"][:3]]


def test_run_factor_backtest_attaches_metadata_for_either_factor_set(panel_csv):
    report = run_factor_backtest(_run_params(panel_csv, factor_set="alpha158",
                                            factors=("MA20", "STD20"), screen_top=20))
    meta = {entry["name"]: entry for entry in report["factor_meta"]}
    assert meta["MA20"]["expression"] == "Mean($close, 20)/$close"
    assert meta["MA20"]["window"] == 20

    alpha8_meta = {entry["name"]: entry for entry in run_factor_backtest(_run_params(panel_csv))["factor_meta"]}
    assert set(alpha8_meta) == set(ALPHA_FACTOR_COLUMNS)
    assert alpha8_meta["alpha5"]["window"] == 20


def test_screen_factors_survives_a_factor_with_no_variation(panel_csv):
    # A constant factor yields zero IC dispersion, so its t stat is None.
    panel = normalize_panel(pd.read_csv(panel_csv, dtype={"trade_date": str, "ts_code": str}))
    panel = compute_alpha_factors(panel)
    panel["flat"] = 1.0
    config = BacktestConfig(horizons=(5,), quantiles=5, score_columns=("alpha1", "flat"),
                            min_names_per_date=10, top_k_sizes=(5,))
    dataset = build_dataset(panel, ["alpha1", "flat"], config)

    screen = screen_factors(dataset, ["alpha1", "flat"], 5, 10)

    assert set(screen["factor"]) == {"alpha1", "flat"}
    assert screen["factor"].iloc[-1] == "flat"
    assert pd.isna(screen["abs_t"].iloc[-1])
