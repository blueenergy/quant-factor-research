#!/usr/bin/env python3
"""Evaluate price-volume factors through ``ScoreBacktester``.

Unlike ``growth_factor_baseline.py``, which reads scores out of Mongo, this
module starts from an OHLCV panel, derives Alpha158-style factors from it, and
feeds them into the shared backtester so a raw factor gets judged by the same
yardstick as a production score column: rank IC across horizons, quantile
monotonicity, and top-K spreads.

Two things IC alone cannot answer are added on top:

- Horizon comparison, so a signal that only lives one day is separated from one
  that survives a 10-20 day holding period.
- Cost-adjusted net return, because at |IC| ~ 0.03 the round-trip cost of daily
  rebalancing consumes roughly the entire gross spread.

The panel comes either from a CSV dump or from Mongo. The Mongo path is
preferred: it reads backward-adjusted prices and resolves index membership
point-in-time, so neither dividend jumps nor a survivorship-biased universe
leaks into the forward returns.

Run:
    python analysis/alpha_factor_backtest.py --source mongo --index hs300 \
        --start-date 20250101 --end-date 20260728 --horizons 1 5 10 20
    python analysis/alpha_factor_backtest.py --panel /path/to/panel.csv.gz
"""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from dataclasses import fields as dataclasses_fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from stock_data_access.index_constituents import (
    asof_snapshot_symbols,
    normalize_yyyymmdd,
    snapshot_symbols_for_keys,
)
from stock_data_access.prices import AdjustedPriceDataAccess

from analysis.core.score_backtester import BacktestConfig, ScoreBacktester
from analysis.core.trade_fee_model import TradeFeeModel
from analysis.factor.alpha158_features import (
    FAMILY_KBAR,
    FAMILY_MOMENTUM,
    FAMILY_TREND,
    FAMILY_VOLUME,
    FAMILY_VOLUME_PRICE,
)
from analysis.factor.ic_metrics import (
    nonoverlap_summary,
    per_date_rank_ic,
    per_year_ic,
    quantile_monotonicity,
    return_col,
    summarize_ic,
)

TRADING_DAYS_PER_YEAR = 252

# Column aliases accepted in the input panel, mapped to the canonical name.
_PANEL_ALIASES = {
    "ts_code": "symbol",
    "code": "symbol",
    "date": "trade_date",
    "vol": "volume",
}

_REQUIRED_PANEL_COLUMNS = ("symbol", "trade_date", "open", "high", "low", "close", "volume")


# ===================== Factor construction =====================
def compute_alpha_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """Derive per-symbol price-volume factors from a long OHLCV panel.

    Warm-up rows stay NaN instead of being filled: a filled zero would enter the
    cross-section as a real rank and bias the IC. Zero-range bars (one-price
    limit moves) turn ratio denominators into NaN rather than being nudged by an
    epsilon, which would otherwise blow those factors up by orders of magnitude.
    """
    df = panel.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    grouped = df.groupby("symbol", sort=False)

    hl_range = (df["high"] - df["low"]).replace(0, np.nan)
    close_low = (df["close"] - df["low"]).replace(0, np.nan)
    prev_close = grouped["close"].shift(1)
    close_lag5 = grouped["close"].shift(5)

    df["alpha1"] = (df["close"] - df["open"]) / hl_range
    df["alpha2"] = (df["high"] - prev_close) / close_low
    df["alpha3"] = df["volume"] / hl_range
    df["alpha4"] = (df["close"] - df["low"]) - (df["high"] - df["close"])
    df["alpha5"] = grouped["close"].transform(lambda s: s.rolling(5).mean() - s.rolling(20).mean())
    df["alpha6"] = grouped["volume"].transform(
        lambda s: s.rolling(5).std() / s.rolling(20).std().replace(0, np.nan)
    )
    df["alpha7"] = (df["high"] - close_lag5) / close_lag5
    df["alpha8"] = grouped["close"].transform(lambda s: s.pct_change(3) - s.pct_change(10))

    return df


ALPHA_FACTOR_COLUMNS = ("alpha1", "alpha2", "alpha3", "alpha4", "alpha5", "alpha6", "alpha7", "alpha8")

# Catalog copy for the eight factors above. The expressions are written in the
# Alpha158 notation so both sets read alike in a picker, but they only describe
# ``compute_alpha_factors``; nothing evaluates them. ``window`` is the longest
# lookback each one needs, which is what a caller sizing warm-up cares about.
_ALPHA8_META: Tuple[Tuple[str, str, str, Optional[int], str], ...] = (
    ("alpha1", "($close-$open)/($high-$low)", FAMILY_KBAR, None, "实体涨跌幅占当日振幅的比例"),
    ("alpha2", "($high-Ref($close, 1))/($close-$low)", FAMILY_KBAR, 1, "上冲幅度相对下影线长度"),
    ("alpha3", "$volume/($high-$low)", FAMILY_VOLUME_PRICE, None, "单位振幅承载的成交量"),
    ("alpha4", "($close-$low)-($high-$close)", FAMILY_KBAR, None, "收盘价在当日区间内的偏离方向"),
    ("alpha5", "Mean($close, 5)-Mean($close, 20)", FAMILY_TREND, 20, "5 日均线与 20 日均线之差"),
    ("alpha6", "Std($volume, 5)/Std($volume, 20)", FAMILY_VOLUME, 20, "近期成交量波动相对中期的倍数"),
    ("alpha7", "($high-Ref($close, 5))/Ref($close, 5)", FAMILY_MOMENTUM, 5, "最高价相对 5 日前收盘的涨幅"),
    ("alpha8", "$close/Ref($close, 3)-$close/Ref($close, 10)", FAMILY_MOMENTUM, 10, "3 日涨幅与 10 日涨幅之差"),
)

# Alpha158's longest rolling window is 60 trading days; calendar days run about
# 1.45x that, and a short suspension eats into the margin further.
ALPHA158_MIN_WARMUP_DAYS = 150

FACTOR_SETS = ("alpha8", "alpha158")
PANEL_SOURCES = ("panel", "mongo")

# Emitted in order by ``run_factor_backtest`` so a caller can surface a stage
# name without knowing how the pipeline is put together.
STAGE_LOADING_PRICES = "loading_prices"
STAGE_COMPUTING_FACTORS = "computing_factors"
STAGE_BUILDING_DATASET = "building_dataset"
STAGE_SCREENING_IC = "screening_ic"
STAGE_RUNNING_DIAGNOSTICS = "running_diagnostics"
STAGE_COST_ADJUSTMENT = "cost_adjustment"

PROGRESS_STAGES = (
    STAGE_LOADING_PRICES,
    STAGE_COMPUTING_FACTORS,
    STAGE_BUILDING_DATASET,
    STAGE_SCREENING_IC,
    STAGE_RUNNING_DIAGNOSTICS,
    STAGE_COST_ADJUSTMENT,
)

# ``on_progress(stage, message)`` for the orchestrator, ``on_message(text)`` for
# the individual steps, which know their own detail but not the stage they sit in.
ProgressFn = Callable[[str, str], None]
MessageFn = Callable[[str], None]


def _emit(on_message: Optional[MessageFn], text: str) -> None:
    if on_message is not None:
        on_message(text)


class FactorBacktestError(ValueError):
    """A request the pipeline refuses: bad parameters or nothing left to evaluate."""


class FactorBudgetError(FactorBacktestError):
    """The request would need more memory than the caller allowed."""


# A-share calendars run about 244 sessions a year, which is what decides how
# many panel rows a symbol contributes. Annualising returns uses the separate
# 252 convention above; keeping them apart avoids one silently redefining the other.
PANEL_ROWS_PER_SYMBOL_YEAR = 244

# Measured on the CSI1000 x 158-factor run: peak RSS was ~4.2 GB against a raw
# factor matrix of ~2.0 GB, the rest being the forward-return dataset copy and
# the wide matrices the evaluator builds. Estimating from the factor matrix and
# multiplying is close enough to catch a request that cannot possibly fit.
FACTOR_MEMORY_OVERHEAD = 2.5

# Rough member counts, only used to size a request before any prices are read.
APPROX_UNIVERSE_SIZE = {"hs300": 300, "a500": 500, "csi1000": 1000, "csi2000": 2000}


@dataclass(frozen=True)
class FactorBacktestParams:
    """A validated, JSON-round-trippable factor backtest request.

    Validation and normalisation happen in ``__post_init__`` so every entry
    point gets them: the CLI, a queued job document, and a direct call. Values
    are coerced to tuples to keep the object hashable and safe to reuse.
    """

    source: str = "panel"
    panel: Optional[str] = None
    index_code: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    warmup_days: int = 60
    workers: int = 8
    horizons: Tuple[int, ...] = (1, 5, 10, 20)
    quantiles: int = 5
    min_names: int = 30
    top_k: Tuple[int, ...] = (20, 50)
    factor_set: str = "alpha8"
    factors: Optional[Tuple[str, ...]] = None
    screen_top: int = 20
    screen_horizon: Optional[int] = None
    slippage_bps: float = 10.0
    turnover: float = 1.0
    memory_budget_gb: float = 6.0

    def __post_init__(self) -> None:
        set_ = lambda name, value: object.__setattr__(self, name, value)  # noqa: E731
        set_("horizons", tuple(sorted({int(h) for h in self.horizons})))
        set_("top_k", tuple(sorted({int(k) for k in self.top_k})))
        set_("factors", tuple(self.factors) if self.factors else None)
        # Alpha158's 60-bar windows would otherwise start mid-sample and the
        # first months of factor values would be incomparable across names.
        if self.factor_set == "alpha158" and self.warmup_days < ALPHA158_MIN_WARMUP_DAYS:
            set_("warmup_days", ALPHA158_MIN_WARMUP_DAYS)
        self._validate()

    def _validate(self) -> None:
        if self.source not in PANEL_SOURCES:
            raise FactorBacktestError(f"source must be one of {PANEL_SOURCES}, got {self.source!r}")
        if self.factor_set not in FACTOR_SETS:
            raise FactorBacktestError(f"factor_set must be one of {FACTOR_SETS}, got {self.factor_set!r}")

        if self.source == "mongo":
            missing = [n for n in ("index_code", "start_date", "end_date") if not getattr(self, n)]
            if missing:
                raise FactorBacktestError(f"source 'mongo' requires: {', '.join(missing)}")
        elif not self.panel:
            raise FactorBacktestError("source 'panel' requires a panel path")

        for name in ("start_date", "end_date"):
            value = getattr(self, name)
            if value is not None:
                _parse_date(value, name)
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise FactorBacktestError(f"start_date {self.start_date} is after end_date {self.end_date}")

        if not self.horizons:
            raise FactorBacktestError("at least one horizon is required")
        if self.horizons[0] < 1:
            raise FactorBacktestError(f"horizons must be >= 1, got {self.horizons}")
        if self.screen_horizon is not None and self.screen_horizon not in self.horizons:
            raise FactorBacktestError(
                f"screen_horizon {self.screen_horizon} is not among horizons {self.horizons}"
            )
        if self.quantiles < 2:
            raise FactorBacktestError(f"quantiles must be >= 2, got {self.quantiles}")
        if self.min_names < self.quantiles:
            raise FactorBacktestError(
                f"min_names {self.min_names} cannot fill {self.quantiles} quantile buckets"
            )
        if not self.top_k or self.top_k[0] < 1:
            raise FactorBacktestError(f"top_k values must be >= 1, got {self.top_k}")
        if self.screen_top < 1:
            raise FactorBacktestError(f"screen_top must be >= 1, got {self.screen_top}")
        if self.warmup_days < 0:
            raise FactorBacktestError(f"warmup_days must be >= 0, got {self.warmup_days}")
        if self.workers < 1:
            raise FactorBacktestError(f"workers must be >= 1, got {self.workers}")
        if self.slippage_bps < 0:
            raise FactorBacktestError(f"slippage_bps must be >= 0, got {self.slippage_bps}")
        if not 0 < self.turnover <= 1:
            raise FactorBacktestError(f"turnover must be in (0, 1], got {self.turnover}")
        if self.memory_budget_gb <= 0:
            raise FactorBacktestError(f"memory_budget_gb must be > 0, got {self.memory_budget_gb}")

        if self.factors is not None:
            self._validate_factor_names()

    def _validate_factor_names(self) -> None:
        assert self.factors is not None
        if self.factor_set == "alpha8":
            known = set(ALPHA_FACTOR_COLUMNS)
        else:
            from analysis.factor.alpha158_features import alpha158_feature_config

            known = set(alpha158_feature_config()[1])
        unknown = [name for name in self.factors if name not in known]
        if unknown:
            raise FactorBacktestError(
                f"unknown {self.factor_set} factors: {', '.join(sorted(unknown))}"
            )

    @property
    def factor_count(self) -> int:
        """How many factor columns the run will materialise."""
        if self.factors is not None:
            return len(self.factors)
        if self.factor_set == "alpha8":
            return len(ALPHA_FACTOR_COLUMNS)
        from analysis.factor.alpha158_features import alpha158_feature_config

        return len(alpha158_feature_config()[1])

    @property
    def years(self) -> Optional[float]:
        """Sample length in years, including the warm-up that gets loaded."""
        if not self.start_date or not self.end_date:
            return None
        span = _parse_date(self.end_date, "end_date") - _parse_date(self.start_date, "start_date")
        return (span.days + abs(self.warmup_days)) / 365.25

    def estimated_memory_gb(self) -> Optional[float]:
        """Peak memory the factor matrix implies, or ``None`` when unknowable.

        A CSV panel could hold anything, so only the Mongo path with a known
        index is estimated; everything else skips the guard.
        """
        universe = APPROX_UNIVERSE_SIZE.get((self.index_code or "").lower())
        years = self.years
        if self.source != "mongo" or universe is None or years is None:
            return None
        rows = universe * PANEL_ROWS_PER_SYMBOL_YEAR * years
        cells = rows * self.factor_count
        return cells * 8 * FACTOR_MEMORY_OVERHEAD / 1024 ** 3

    def check_budget(self) -> Optional[float]:
        """Reject a request that cannot fit, pointing at the two-stage workflow."""
        estimate = self.estimated_memory_gb()
        if estimate is not None and estimate > self.memory_budget_gb:
            raise FactorBudgetError(
                f"{self.factor_count} factors x {self.years:.1f}y on {self.index_code} needs about "
                f"{estimate:.1f} GB, over the {self.memory_budget_gb:.1f} GB budget. Screen all "
                "factors on a shorter window first, then re-test the shortlist by name over the "
                "long history."
            )
        return estimate

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FactorBacktestParams":
        """Build from a job document, rejecting keys the pipeline would ignore."""
        known = {f.name for f in dataclasses_fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise FactorBacktestError(f"unknown parameters: {', '.join(sorted(unknown))}")
        return cls(**dict(payload))

    def to_dict(self) -> Dict[str, Any]:
        """Normalised parameters, safe to store next to the report."""
        payload: Dict[str, Any] = {}
        for f in dataclasses_fields(self):
            value = getattr(self, f.name)
            payload[f.name] = list(value) if isinstance(value, tuple) else value
        return payload


def _parse_date(value: str, label: str) -> datetime:
    try:
        return datetime.strptime(str(value), "%Y%m%d")
    except ValueError as exc:
        raise FactorBacktestError(f"{label} must be YYYYMMDD, got {value!r}") from exc


def compute_factor_set(
    panel: pd.DataFrame,
    factor_set: str,
    only: Optional[Sequence[str]] = None,
    on_message: Optional[MessageFn] = None,
) -> "tuple[pd.DataFrame, List[str], List[Dict[str, str]]]":
    """Attach the requested factor family, the columns it produced, and its skips.

    A skip carries the reason the panel could not supply the factor, so a run
    that evaluates fewer factors than the set nominally holds can say why.
    """
    if factor_set == "alpha8":
        wanted = [c for c in ALPHA_FACTOR_COLUMNS if only is None or c in set(only)]
        return compute_alpha_factors(panel), wanted, []
    if factor_set == "alpha158":
        from analysis.factor.alpha158_features import compute_alpha158

        ordered = panel.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
        return compute_alpha158(ordered, only=only, on_message=on_message)
    raise ValueError(f"unknown factor set {factor_set!r}")


def normalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases, coerce dtypes, and drop rows with unusable prices."""
    df = panel.rename(columns={k: v for k, v in _PANEL_ALIASES.items() if k in panel.columns})
    missing = [col for col in _REQUIRED_PANEL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"panel is missing required columns: {', '.join(missing)}")

    df = df.copy()
    df["symbol"] = df["symbol"].astype(str)
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "", regex=False).str[:8]
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["close"].notna() & (df["close"] > 0)]
    return df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


# ===================== Mongo source =====================
def resolve_data_db(db: Any = None) -> Any:
    """Return a handle on the ``quant_data`` pipeline DB.

    Connection settings come from ``stock_data_access.mongo_context`` so this
    worker never imports the scoring repo's ``data_manager``.
    """
    if db is not None:
        return db
    from stock_data_access.mongo_context import get_db

    return get_db()


def shift_calendar_days(date_text: str, days: int) -> str:
    return (datetime.strptime(date_text, "%Y%m%d") + timedelta(days=days)).strftime("%Y%m%d")


class PointInTimeUniverse:
    """Index membership as of each trading day, from ``index_constituents``.

    Snapshots are keyed by ``weight_trade_date``. The snapshot in force on a
    given day is the newest one dated on or before it, so a stock only enters a
    cross-section on the days it was actually in the index. Using the union of
    all snapshots instead would quietly add survivorship bias.
    """

    def __init__(self, coll: Any, index_code: str, start_date: str, end_date: str):
        self.index_code = index_code
        self._dates: List[str] = []
        self._members: Dict[str, Set[str]] = {}

        for snapshot_date in self._snapshot_dates(coll, index_code, start_date, end_date):
            symbols = snapshot_symbols_for_keys(
                coll, index_code, weight_trade_date=snapshot_date, update_date=None
            )
            if symbols:
                self._dates.append(snapshot_date)
                self._members[snapshot_date] = set(symbols)

        if not self._dates:
            # Legacy documents without weight_trade_date: fall back to one snapshot.
            fallback = set(asof_snapshot_symbols(coll, index_code, end_date))
            if fallback:
                self._dates = [start_date]
                self._members = {start_date: fallback}

    @staticmethod
    def _snapshot_dates(coll: Any, index_code: str, start_date: str, end_date: str) -> List[str]:
        raw = coll.distinct("weight_trade_date", {"index_code": index_code})
        known = sorted({normalize_yyyymmdd(value) for value in raw} - {""})
        if not known:
            return []
        in_window = [value for value in known if start_date < value <= end_date]
        earlier = [value for value in known if value <= start_date]
        # The snapshot already in force when the window opens has to lead the list.
        return ([earlier[-1]] if earlier else [known[0]]) + in_window

    @property
    def symbols(self) -> Set[str]:
        """Every symbol that belongs to the index at some point in the window."""
        return set().union(*self._members.values()) if self._members else set()

    def members_on(self, trade_date: str) -> Set[str]:
        if not self._dates:
            return set()
        position = max(0, bisect_right(self._dates, trade_date) - 1)
        return self._members[self._dates[position]]


def _adjusted_symbol_frame(
    accessor: AdjustedPriceDataAccess,
    symbol: str,
    start_date: str,
    end_date: str,
) -> Optional[pd.DataFrame]:
    frame = accessor.load_adjusted_ohlc(symbol, start_date, end_date, adjust="hfq")
    if frame.empty:
        return None
    columns = [col for col in ("trade_date", "open", "high", "low", "close", "volume") if col in frame.columns]
    out = frame.reset_index(drop=True)[columns].copy()
    out["symbol"] = symbol
    return out


def load_mongo_panel(
    index_code: str,
    start_date: str,
    end_date: str,
    *,
    db: Any = None,
    warmup_days: int = 60,
    workers: int = 8,
    on_message: Optional[MessageFn] = None,
) -> pd.DataFrame:
    """Backward-adjusted OHLCV panel for an index, flagged with point-in-time membership.

    Prices are pulled from ``warmup_days`` before ``start_date`` so rolling
    factors are warm on the first scored day, and for every symbol that is ever
    a member. The extra rows stay in the frame because forward returns must be
    able to exit a position after the stock leaves the index; ``in_pool`` marks
    the subset that may enter a cross-section.
    """
    db = resolve_data_db(db)
    universe = PointInTimeUniverse(db["index_constituents"], index_code, start_date, end_date)
    symbols = sorted(universe.symbols)
    if not symbols:
        raise ValueError(f"no index_constituents snapshot found for {index_code}")

    fetch_start = shift_calendar_days(start_date, -abs(warmup_days))
    accessor = AdjustedPriceDataAccess(db=db)
    frames: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="alpha-px") as pool:
        futures = {
            pool.submit(_adjusted_symbol_frame, accessor, symbol, fetch_start, end_date): symbol
            for symbol in symbols
        }
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                frame = future.result()
            except Exception as exc:  # pragma: no cover - defensive, per-symbol read
                _emit(on_message, f"price load failed for {futures[future]}: {exc}")
                continue
            if frame is not None:
                frames.append(frame)
            if done % 200 == 0 or done == len(futures):
                _emit(on_message, f"loaded {done}/{len(futures)} symbols")

    if not frames:
        raise ValueError(f"no adjusted prices found for {index_code} in {fetch_start}-{end_date}")

    panel = normalize_panel(pd.concat(frames, ignore_index=True))
    panel["in_pool"] = [
        date >= start_date and symbol in universe.members_on(date)
        for symbol, date in zip(panel["symbol"], panel["trade_date"])
    ]
    return panel


# ===================== ScoreBacktester adapter =====================
def build_price_cache(panel: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """symbol -> date-sorted OHLC rows in the shape ``ScoreBacktester`` expects."""
    cache: Dict[str, List[Dict[str, Any]]] = {}
    columns = ["trade_date", "open", "high", "low", "close"]
    for symbol, group in panel.groupby("symbol", sort=False):
        rows = group[columns].sort_values("trade_date").to_dict("records")
        for row in rows:
            # Entry/exit fills read ``open`` and fall back to ``close``; a NaN open
            # would poison that fallback because ``NaN or close`` yields NaN.
            if row["open"] is None or (isinstance(row["open"], float) and math.isnan(row["open"])):
                row["open"] = row["close"]
        cache[str(symbol)] = rows
    return cache


def panel_to_score_records(
    panel: pd.DataFrame,
    factor_columns: Sequence[str],
) -> List[Dict[str, Any]]:
    """One record per (symbol, date), using the bar's own date as ``score_date``.

    The factor is known at that bar's close and ``ScoreBacktester`` enters at the
    next open, so the panel carries no lookahead. When the panel carries an
    ``in_pool`` flag, warm-up rows and days outside index membership are dropped
    here rather than in the price cache, which still needs them to exit trades.
    """
    frame = panel[panel["in_pool"]] if "in_pool" in panel.columns else panel
    present = [col for col in factor_columns if col in panel.columns]
    frame = frame[["symbol", "trade_date", *present]].dropna(subset=present, how="all")

    records: List[Dict[str, Any]] = []
    for row in frame.to_dict("records"):
        record: Dict[str, Any] = {"symbol": row["symbol"], "score_date": row["trade_date"]}
        for col in factor_columns:
            value = row.get(col)
            record[col] = None if value is None or pd.isna(value) else float(value)
        records.append(record)
    return records


class PanelFactorBacktester(ScoreBacktester):
    """``ScoreBacktester`` fed from an in-memory panel instead of Mongo.

    Two overrides are needed. ``_score_row_from_doc`` normally projects a fixed
    list of production score fields, so arbitrary factor columns have to be
    passed through instead. ``_load_future_prices`` normally rescans a symbol's
    whole history per row, which is quadratic once a panel reaches hundreds of
    thousands of rows, so dates are looked up by binary search here.
    """

    def __init__(self, price_cache: Mapping[str, List[Dict[str, Any]]], config: BacktestConfig):
        super().__init__(db_manager=None, config=config)
        self._price_cache = {symbol: list(rows) for symbol, rows in price_cache.items()}
        self._date_index = {
            symbol: [row["trade_date"] for row in rows] for symbol, rows in self._price_cache.items()
        }

    def _score_row_from_doc(self, doc: Dict[str, Any]) -> Dict[str, Optional[float]]:
        return {col: self._to_float(doc.get(col)) for col in self.config.score_columns}

    def _load_future_prices(self, symbol: str, score_date: str, max_horizon: int) -> List[Dict[str, Any]]:
        history = self._price_cache.get(symbol)
        if not history:
            return []
        start = bisect_left(self._date_index[symbol], score_date)
        return history[start : start + max_horizon + 1]


_PRESENCE_COLUMN = "__scored__"


def build_dataset(
    panel: pd.DataFrame,
    factor_columns: Sequence[str],
    config: BacktestConfig,
) -> pd.DataFrame:
    """Join factors to forward returns via the shared backtester.

    A forward return depends only on the symbol and the date, never on the
    factor value, so it is computed once against a placeholder column and the
    factors are merged on afterwards. Routing 158 factors through the record
    path instead would allocate a dict entry per factor per row, which on a
    million-row panel costs tens of gigabytes for values the backtester only
    copies through.
    """
    present = [col for col in factor_columns if col in panel.columns]
    if not present:
        return pd.DataFrame()

    frame = panel[panel["in_pool"]] if "in_pool" in panel.columns else panel
    frame = (
        frame[["symbol", "trade_date", *present]]
        .dropna(subset=present, how="all")
        .drop_duplicates(subset=["symbol", "trade_date"], keep="last")
        .rename(columns={"trade_date": "score_date"})
    )

    backtester = PanelFactorBacktester(
        build_price_cache(panel), replace(config, score_columns=(_PRESENCE_COLUMN,))
    )
    # A hand-built panel rather than a record list: the placeholder carries no
    # per-document state, so there is nothing for _score_row_from_doc to do.
    # score_date is re-normalized because trade_date's format is the caller's.
    # name/industry are blank because the panel contract carries them through to
    # the dataset and this path merges its own frame on afterwards.
    score_panel = pd.DataFrame(
        {
            "symbol": frame["symbol"].to_numpy(),
            "score_date": [backtester._normalize_date(date) for date in frame["score_date"]],
            "name": "",
            "industry": "",
            _PRESENCE_COLUMN: 1.0,
        }
    )
    dataset = backtester.build_dataset_from_panel(score_panel)
    if dataset.empty:
        return dataset
    return dataset.drop(columns=[_PRESENCE_COLUMN]).merge(frame, on=["symbol", "score_date"], how="left")


def screen_rank_ic(
    dataset: pd.DataFrame,
    factor_columns: Sequence[str],
    ret_col: str,
    min_names: int,
) -> Dict[str, pd.DataFrame]:
    """Per-date rank IC for many factors at once.

    ``per_date_rank_ic`` loops over dates in Python and costs a few hundred
    milliseconds per factor per horizon, which is fine for eight factors and
    not for 158. Here the Spearman correlation is rebuilt from grouped sums of
    the within-date ranks, so every factor is done in one pass.

    Only factors that are complete over the evaluated rows take that path.
    A factor with gaps would need its ranks recomputed on its own subset, so
    those fall back to the reference implementation rather than being ranked
    against rows the correlation will not see.
    """
    results: Dict[str, pd.DataFrame] = {col: pd.DataFrame() for col in factor_columns}
    usable = [col for col in factor_columns if col in dataset.columns]
    if dataset.empty or not usable or ret_col not in dataset.columns:
        return results

    frame = dataset[["score_date", ret_col, *usable]].dropna(subset=[ret_col])
    complete = [col for col in usable if not frame[col].isna().any()]
    for col in usable:
        if col not in complete:
            results[col] = per_date_rank_ic(dataset, col, ret_col, min_names)
    if not complete:
        return results

    by_date = frame.groupby("score_date", sort=True)
    factor_ranks = by_date[complete].rank()
    return_ranks = by_date[ret_col].rank()

    counts = by_date[ret_col].count()
    sum_return = return_ranks.groupby(frame["score_date"]).sum()
    sum_return_sq = (return_ranks**2).groupby(frame["score_date"]).sum()
    sum_factor = factor_ranks.groupby(frame["score_date"]).sum()
    sum_factor_sq = (factor_ranks**2).groupby(frame["score_date"]).sum()
    sum_cross = factor_ranks.mul(return_ranks, axis=0).groupby(frame["score_date"]).sum()

    covariance = sum_cross.sub(sum_factor.mul(sum_return / counts, axis=0))
    factor_variance = sum_factor_sq.sub(sum_factor.pow(2).div(counts, axis=0))
    return_variance = sum_return_sq - sum_return.pow(2) / counts
    # A date where every value ties has zero variance and no defined correlation.
    denominator = np.sqrt(factor_variance.clip(lower=0).mul(return_variance.clip(lower=0), axis=0))
    ic = covariance.div(denominator.replace(0, np.nan))
    ic = ic[counts >= min_names]

    for col in complete:
        series = ic[col].dropna()
        results[col] = pd.DataFrame(
            {
                "score_date": series.index.astype(str),
                "rank_ic": series.to_numpy(dtype=float),
                "count": counts.reindex(series.index).to_numpy(dtype=int),
            }
        ).reset_index(drop=True)
    return results


# ===================== Cost model =====================
@dataclass(frozen=True)
class CostAssumptions:
    """Per-rebalance trading cost for a periodically rebalanced book."""

    fee: TradeFeeModel
    slippage_bps_one_way: float = 10.0
    turnover_per_rebalance: float = 1.0

    def round_trip_rate(self) -> float:
        """Fraction of position value lost per buy-and-sell cycle."""
        slippage = 2.0 * max(0.0, self.slippage_bps_one_way) / 10_000.0
        turnover = max(0.0, self.turnover_per_rebalance)
        return self.fee.period_cost_rate(turnover) + turnover * slippage


def net_return_table(
    topk_summary: pd.DataFrame,
    costs: CostAssumptions,
) -> pd.DataFrame:
    """Subtract trading cost from the gross top-K spreads in ``topk_summary``.

    The long-only column is charged one round trip per rebalance and compared to
    a buy-and-hold universe that is assumed costless. The long-short column is
    charged two, one per leg. Annualised figures scale the per-period net by the
    number of rebalances per year without compounding.
    """
    if topk_summary.empty:
        return pd.DataFrame()

    round_trip = costs.round_trip_rate()
    rows: List[Dict[str, Any]] = []
    for row in topk_summary.to_dict("records"):
        horizon = int(row["horizon"])
        rebalances = TRADING_DAYS_PER_YEAR / horizon
        long_gross = float(row["mean_top_vs_universe"])
        ls_gross = float(row["mean_top_minus_bottom"])
        long_net = long_gross - round_trip
        ls_net = ls_gross - 2.0 * round_trip
        rows.append(
            {
                "score_column": row["score_column"],
                "horizon": horizon,
                "top_k": int(row["top_k"]),
                "cost_per_rebalance": round_trip,
                "long_excess_gross": long_gross,
                "long_excess_net": long_net,
                "long_excess_net_annual": long_net * rebalances,
                "long_short_gross": ls_gross,
                "long_short_net": ls_net,
                "long_short_net_annual": ls_net * rebalances,
                "observations": int(row["observations"]),
            }
        )
    return pd.DataFrame(rows)


# ===================== Report =====================
def evaluate(
    dataset: pd.DataFrame,
    factor_columns: Sequence[str],
    config: BacktestConfig,
) -> Dict[str, Any]:
    """Per-factor, per-horizon IC and quantile diagnostics."""
    if dataset.empty:
        return {"error": "empty dataset: no overlapping factor rows and prices found"}

    available = [col for col in factor_columns if col in dataset.columns]
    report: Dict[str, Any] = {
        "rows": int(len(dataset)),
        "score_date_range": [dataset["score_date"].min(), dataset["score_date"].max()],
        "distinct_dates": int(dataset["score_date"].nunique()),
        "symbols": int(dataset["symbol"].nunique()),
        "evaluated_columns": available,
        "factors": {},
    }

    for factor in available:
        per_horizon: Dict[str, Any] = {}
        for horizon in config.horizons:
            ret_col = return_col(horizon)
            if ret_col not in dataset.columns:
                continue
            per_date_ic = per_date_rank_ic(dataset, factor, ret_col, config.min_names_per_date)
            per_horizon[str(horizon)] = {
                "ic": summarize_ic(per_date_ic),
                "nonoverlap": nonoverlap_summary(per_date_ic, horizon),
                "per_year": per_year_ic(per_date_ic),
                "quantiles": quantile_monotonicity(
                    dataset, factor, ret_col, config.quantiles, config.min_names_per_date
                ),
            }
        report["factors"][factor] = per_horizon

    return report


def _fmt(value: Any, precision: int = 4) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{value:.{precision}f}"


def print_ic_table(report: Dict[str, Any]) -> None:
    if report.get("error"):
        print(f"[alpha] {report['error']}")
        return
    print(
        f"[alpha] rows={report['rows']} dates={report['distinct_dates']} "
        f"symbols={report['symbols']} range={report['score_date_range']}"
    )
    header = (
        f"{'factor':<10}{'h':>4}{'IC':>9}{'IC_IR':>8}{'t':>7}{'t_non':>7}{'MDE80':>8}"
        f"{'pos%':>7}{'spread':>10}{'q_corr':>8}{'days':>6}"
    )
    print("\n" + header)
    print("-" * len(header))
    for factor, horizons in report["factors"].items():
        for horizon, block in horizons.items():
            ic = block["ic"]
            non = block.get("nonoverlap") or {}
            quant = block["quantiles"]
            print(
                f"{factor:<10}{horizon:>4}{_fmt(ic['ic_mean']):>9}{_fmt(ic['ic_ir']):>8}"
                f"{_fmt(ic['t_stat'], 2):>7}{_fmt(non.get('t_stat'), 2):>7}"
                f"{_fmt(non.get('mde80'), 3):>8}{_fmt(ic['positive_ratio'], 2):>7}"
                f"{_fmt(quant['top_minus_bottom_spread']):>10}"
                f"{_fmt(quant['quantile_rank_corr'], 2):>8}{ic['observations']:>6}"
            )


def print_net_table(net: pd.DataFrame, top_k: Optional[int] = None) -> None:
    if net.empty:
        print("\n[alpha] no top-K rows to cost-adjust")
        return
    subset = net if top_k is None else net[net["top_k"] == top_k]
    if subset.empty:
        return
    k = int(subset["top_k"].iloc[0])
    cost = float(subset["cost_per_rebalance"].iloc[0])
    header = (
        f"{'factor':<10}{'h':>4}{'LS_gross':>10}{'LS_net':>10}{'LS_net_yr':>11}"
        f"{'L_gross':>10}{'L_net':>10}{'L_net_yr':>11}"
    )
    print(f"\nCost-adjusted top-{k} (round trip {cost * 100:.2f}% per rebalance)")
    print(header)
    print("-" * len(header))
    for row in subset.sort_values(["score_column", "horizon"]).to_dict("records"):
        print(
            f"{row['score_column']:<10}{row['horizon']:>4}"
            f"{_fmt(row['long_short_gross']):>10}{_fmt(row['long_short_net']):>10}"
            f"{_fmt(row['long_short_net_annual'], 3):>11}"
            f"{_fmt(row['long_excess_gross']):>10}{_fmt(row['long_excess_net']):>10}"
            f"{_fmt(row['long_excess_net_annual'], 3):>11}"
        )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate price-volume factors from an OHLCV panel via ScoreBacktester."
    )
    parser.add_argument("--source", choices=("panel", "mongo"), default="panel")
    parser.add_argument("--panel", help="OHLCV panel CSV (.csv or .csv.gz); required for --source panel")
    parser.add_argument("--index", help="index_constituents code, e.g. hs300/a500/csi1000/csi2000")
    parser.add_argument("--start-date", help="First scored trading day, YYYYMMDD")
    parser.add_argument("--end-date", help="Last trading day to load, YYYYMMDD")
    parser.add_argument("--warmup-days", type=int, default=60, help="Calendar days of history before start")
    parser.add_argument("--workers", type=int, default=8, help="Parallel Mongo price readers")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--min-names", type=int, default=30, help="Min names per cross-section")
    parser.add_argument("--top-k", nargs="+", type=int, default=[20, 50])
    parser.add_argument("--factor-set", choices=("alpha8", "alpha158"), default="alpha8")
    parser.add_argument("--factors", nargs="+", help="Restrict to these factor columns")
    parser.add_argument(
        "--screen-top",
        type=int,
        default=20,
        help="Run the full quantile and cost diagnostics on this many factors, ranked by |t|",
    )
    parser.add_argument(
        "--screen-horizon",
        type=int,
        help="Horizon used to rank factors during screening (default: the longest)",
    )
    parser.add_argument("--slippage-bps", type=float, default=10.0, help="One-way slippage in bps")
    parser.add_argument("--turnover", type=float, default=1.0, help="Turnover per rebalance")
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=6.0,
        help="Reject a factors x years request estimated to need more than this",
    )
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument("--export-net", help="Optional CSV path for the cost-adjusted table")
    return parser.parse_args(argv)


def load_source_panel(
    params: FactorBacktestParams,
    *,
    db: Any = None,
    on_message: Optional[MessageFn] = None,
) -> pd.DataFrame:
    """Read the OHLCV panel named by ``params`` from Mongo or a CSV dump."""
    if params.source == "mongo":
        _emit(on_message, f"loading {params.index_code} hfq prices {params.start_date}-{params.end_date}")
        return load_mongo_panel(
            params.index_code,
            params.start_date,
            params.end_date,
            db=db,
            warmup_days=params.warmup_days,
            workers=params.workers,
            on_message=on_message,
        )

    _emit(on_message, f"loading panel {params.panel}")
    raw = pd.read_csv(params.panel, dtype={"trade_date": str, "ts_code": str, "symbol": str})
    return normalize_panel(raw)


def screen_factors(
    dataset: pd.DataFrame,
    factor_columns: Sequence[str],
    horizon: int,
    min_names: int,
) -> pd.DataFrame:
    """Rank every factor by the significance of its rank IC at one horizon."""
    per_factor = screen_rank_ic(dataset, factor_columns, return_col(horizon), min_names)
    rows: List[Dict[str, Any]] = []
    for factor, per_date_ic in per_factor.items():
        summary = summarize_ic(per_date_ic)
        nonoverlap = nonoverlap_summary(per_date_ic, horizon)
        rows.append(
            {
                "factor": factor,
                "horizon": horizon,
                "ic_mean": summary["ic_mean"],
                "ic_ir": summary["ic_ir"],
                "t_stat": summary["t_stat"],
                "positive_ratio": summary["positive_ratio"],
                "observations": summary["observations"],
                "t_nonoverlap": nonoverlap["t_stat"],
                "nonoverlap_observations": nonoverlap["observations"],
                "mde80": nonoverlap["mde80"],
            }
        )
    screen = pd.DataFrame(rows)
    if screen.empty:
        return screen
    # Screening keeps the overlapping t: more draws and a stabler ranking decide
    # which factors deserve the expensive diagnostics. The non-overlap t and
    # MDE80 travel with each row so a reader can tell screening from a verdict.
    # A factor whose IC never varies has no t stat, which leaves the column as
    # objects holding None; coercing first keeps one dead factor from taking the
    # whole screen down with it, and sorts it to the bottom where it belongs.
    screen["abs_t"] = pd.to_numeric(screen["t_stat"], errors="coerce").abs()
    return screen.sort_values("abs_t", ascending=False, na_position="last").reset_index(drop=True)


def print_screen_table(screen: pd.DataFrame, limit: int) -> None:
    if screen.empty:
        print("\n[alpha] screening produced no usable IC")
        return
    horizon = int(screen["horizon"].iloc[0])
    header = f"{'factor':<10}{'IC':>9}{'IC_IR':>8}{'t':>8}{'t_non':>7}{'MDE80':>8}{'pos%':>7}{'days':>7}"
    print(f"\nRank IC screen at {horizon}d, top {min(limit, len(screen))} of {len(screen)} by |t| (overlapping)")
    print(header)
    print("-" * len(header))
    for row in screen.head(limit).to_dict("records"):
        print(
            f"{row['factor']:<10}{_fmt(row['ic_mean']):>9}{_fmt(row['ic_ir']):>8}"
            f"{_fmt(row['t_stat'], 2):>8}{_fmt(row.get('t_nonoverlap'), 2):>7}"
            f"{_fmt(row.get('mde80'), 3):>8}{_fmt(row['positive_ratio'], 2):>7}{row['observations']:>7}"
        )


def factor_set_catalog(factor_set: str) -> List[Dict[str, Any]]:
    """Describe every factor in one set: name, expression, family, window, blurb.

    This is what a picker needs in order to offer factors by name without
    knowing how either set is built, and it is derived from the same
    definitions the pipeline computes, so a factor cannot appear in a backtest
    without a catalog entry.
    """
    if factor_set == "alpha8":
        return [
            {
                "name": name,
                "expression": expression,
                "family": family,
                "operator": None,
                "window": window,
                "description": description,
            }
            for name, expression, family, window, description in _ALPHA8_META
        ]
    if factor_set == "alpha158":
        from analysis.factor.alpha158_features import factor_catalog

        return factor_catalog()
    raise FactorBacktestError(f"factor_set must be one of {FACTOR_SETS}, got {factor_set!r}")


def factor_metadata(names: Sequence[str]) -> List[Dict[str, Any]]:
    """Catalog entries for the named factors, in the order given.

    Both sets are searched because a report only knows factor names, not which
    set produced them. An unknown name is dropped rather than faked, so a
    caller can tell metadata is genuinely absent.
    """
    by_name: Dict[str, Dict[str, Any]] = {}
    for factor_set in FACTOR_SETS:
        for entry in factor_set_catalog(factor_set):
            by_name.setdefault(entry["name"], entry)
    return [by_name[name] for name in names if name in by_name]


def run_factor_backtest(
    params: FactorBacktestParams,
    on_progress: Optional[ProgressFn] = None,
    *,
    db: Any = None,
) -> Dict[str, Any]:
    """Run the whole pipeline for one request and return a JSON-ready report.

    Stages are announced through ``on_progress(stage, message)`` using the names
    in ``PROGRESS_STAGES``; the diagnostics stage dominates the wall clock, so a
    caller that only records the latest stage still gives a useful ETA. Nothing
    here prints or reads ``sys.argv``, which is what lets a queue worker call it.
    """

    def progress(stage: str, message: str = "") -> None:
        if on_progress is not None:
            on_progress(stage, message)

    def relay(stage: str) -> MessageFn:
        return lambda message: progress(stage, message)

    estimated_memory_gb = params.check_budget()

    progress(STAGE_LOADING_PRICES, "reading prices")
    panel = load_source_panel(params, db=db, on_message=relay(STAGE_LOADING_PRICES))

    progress(STAGE_COMPUTING_FACTORS, f"computing {params.factor_set} factors")
    panel, produced, unavailable = compute_factor_set(
        panel, params.factor_set, params.factors, relay(STAGE_COMPUTING_FACTORS)
    )
    requested = list(params.factors or produced)
    factors = [col for col in requested if col in panel.columns]
    if not factors:
        raise FactorBacktestError(f"none of the requested factors exist in the panel: {requested}")

    config = BacktestConfig(
        horizons=params.horizons,
        quantiles=params.quantiles,
        score_columns=tuple(factors),
        min_names_per_date=params.min_names,
        top_k_sizes=params.top_k,
    )

    progress(
        STAGE_BUILDING_DATASET,
        f"building dataset for {len(factors)} factors, horizons={params.horizons}",
    )
    dataset = build_dataset(panel, factors, config)

    # With 158 factors the quantile and top-K diagnostics cost more than the IC
    # they would be reported next to, so they only run on what survives the screen.
    screen_horizon = params.screen_horizon or params.horizons[-1]
    progress(STAGE_SCREENING_IC, f"rank IC screen at {screen_horizon}d over {len(factors)} factors")
    screen = screen_factors(dataset, factors, screen_horizon, params.min_names)
    if len(factors) > params.screen_top:
        if screen.empty:
            raise FactorBacktestError(
                "rank IC screen produced no usable cross-sections, so there is nothing to "
                f"narrow {len(factors)} factors down to; check min_names and the date range"
            )
        factors = screen["factor"].head(params.screen_top).tolist()
        config = replace(config, score_columns=tuple(factors))

    progress(STAGE_RUNNING_DIAGNOSTICS, f"quantile diagnostics on {len(factors)} factors")
    report = evaluate(dataset, factors, config)
    report["screen"] = {
        "horizon": screen_horizon,
        "ranked": screen.drop(columns=["abs_t"], errors="ignore").to_dict(orient="records"),
    }

    progress(STAGE_COST_ADJUSTMENT, f"cost-adjusting top-{'/'.join(str(k) for k in params.top_k)}")
    backtester = PanelFactorBacktester(build_price_cache(panel), config)
    costs = CostAssumptions(
        fee=TradeFeeModel.from_params(),
        slippage_bps_one_way=params.slippage_bps,
        turnover_per_rebalance=params.turnover,
    )
    net = net_return_table(backtester.calculate_topk_summary(dataset), costs)

    report["source"] = {
        "kind": params.source,
        "factor_set": params.factor_set,
        "index_code": params.index_code,
        "panel": params.panel,
        "start_date": params.start_date,
        "end_date": params.end_date,
        "adjust": "hfq" if params.source == "mongo" else "none",
        "point_in_time_universe": params.source == "mongo",
    }
    report["cost_assumptions"] = {
        "slippage_bps_one_way": params.slippage_bps,
        "turnover_per_rebalance": params.turnover,
        "round_trip_rate": costs.round_trip_rate(),
    }
    report["net_returns"] = net.to_dict(orient="records")
    report["params"] = params.to_dict()
    report["factor_meta"] = factor_metadata(factors)
    # Factors the request covered but the panel could not supply. Names outside
    # the set are already rejected by FactorBacktestParams, so this only ever
    # holds missing base fields: the panel carries no $vwap, which is why a full
    # alpha158 run evaluates 157 and would otherwise look like a bug.
    report["skipped_factors"] = unavailable
    report["estimated_memory_gb"] = estimated_memory_gb
    return report


def _params_from_args(args: argparse.Namespace) -> FactorBacktestParams:
    return FactorBacktestParams(
        source=args.source,
        panel=args.panel,
        index_code=args.index,
        start_date=args.start_date,
        end_date=args.end_date,
        warmup_days=args.warmup_days,
        workers=args.workers,
        horizons=tuple(args.horizons),
        quantiles=args.quantiles,
        min_names=args.min_names,
        top_k=tuple(args.top_k),
        factor_set=args.factor_set,
        factors=tuple(args.factors) if args.factors else None,
        screen_top=args.screen_top,
        screen_horizon=args.screen_horizon,
        slippage_bps=args.slippage_bps,
        turnover=args.turnover,
        memory_budget_gb=args.memory_budget_gb,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        params = _params_from_args(args)
    except FactorBacktestError as exc:
        print(f"[alpha] {exc}")
        return 2

    if params.warmup_days != args.warmup_days:
        print(
            f"[alpha] raising warm-up from {args.warmup_days} to {params.warmup_days} days: "
            "Alpha158's 60-bar windows would otherwise start mid-sample"
        )

    try:
        report = run_factor_backtest(params, lambda stage, message: print(f"[alpha] {stage}: {message}"))
    except FactorBacktestError as exc:
        print(f"[alpha] {exc}")
        return 1

    screen = pd.DataFrame(report["screen"]["ranked"])
    if len(screen) > params.screen_top:
        print_screen_table(screen, limit=max(params.screen_top, 30))
    print_ic_table(report)

    net = pd.DataFrame(report["net_returns"])
    for k in params.top_k:
        print_net_table(net, top_k=int(k))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[alpha] report written to {args.output}")
    if args.export_net and not net.empty:
        Path(args.export_net).parent.mkdir(parents=True, exist_ok=True)
        net.to_csv(args.export_net, index=False)
        print(f"[alpha] cost-adjusted table written to {args.export_net}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
