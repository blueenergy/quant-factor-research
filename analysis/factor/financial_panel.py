"""Point-in-time financial panel for fundamental factors.

The raw collections store one row per ``(ts_code, end_date, report_type)`` and
carry ``ann_date``/``f_ann_date``. Two things are easy to get wrong when turning
that into a daily factor panel:

* Using ``end_date`` as the observation date. A Q1 report is published up to
  four months later, so aligning on ``end_date`` hands the model a number the
  market did not have yet.
* Summing cumulative reports. ``report_type=1`` is year-to-date, so a Q3 TTM
  built by adding four cumulative rows double-counts Q1 and Q2. This module
  reduces everything to single quarters first, then sums the last four.

The daily grid is supplied by the caller (the price panel already knows the
trading dates) so this module does not need its own calendar query.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

INCOME_COLLECTION = "financial_income"
BALANCE_COLLECTION = "financial_balance"

_INCOME_PROJECTION = {
    "_id": 0,
    "ts_code": 1,
    "end_date": 1,
    "ann_date": 1,
    "f_ann_date": 1,
    "report_type": 1,
    "revenue": 1,
    "total_revenue": 1,
    "oper_cost": 1,
    "n_income_attr_p": 1,
}
_BALANCE_PROJECTION = {
    "_id": 0,
    "ts_code": 1,
    "end_date": 1,
    "ann_date": 1,
    "f_ann_date": 1,
    "total_hldr_eqy_exc_min_int": 1,
}

_QUARTER_BY_MONTH = {3: 1, 6: 2, 9: 3, 12: 4}


def _text(value: Any) -> str:
    """Normalize a YYYYMMDD date. Do not use this on ts_codes: it truncates."""
    return str(value or "").replace("-", "").strip()[:8]


def _code(value: Any) -> str:
    """Normalize a ts_code (``600519.SH``) without truncating it."""
    return str(value or "").strip()


def _disclosure_date(row: Dict[str, Any]) -> str:
    return _text(row.get("f_ann_date")) or _text(row.get("ann_date"))


def _quarter(end_date: str) -> Optional[tuple]:
    text = _text(end_date)
    if len(text) != 8 or not text.isdigit():
        return None
    quarter = _QUARTER_BY_MONTH.get(int(text[4:6]))
    if quarter is None:
        return None
    return int(text[:4]), quarter


def _previous_quarter(key: tuple) -> tuple:
    year, quarter = key
    return (year, quarter - 1) if quarter > 1 else (year - 1, 4)


def _asof_effective_index(disclosure: str, dates: pd.Index, lag_days: int) -> Optional[int]:
    """First grid position whose information set includes ``disclosure``.

    ``searchsorted(side="right")`` puts a report published on a trading day
    into the *next* session, which is the conservative choice: an EOD signal
    cannot use a filing that landed after the close of the same day.
    """
    if not disclosure or dates.empty:
        return None
    position = int(dates.searchsorted(disclosure, side="right")) + max(lag_days - 1, 0)
    if position >= len(dates):
        return None
    return position


def _single_quarter_series(rows: Iterable[Dict[str, Any]], field: str) -> Dict[str, float]:
    """Reduce one symbol's rows to single-quarter values keyed by ``end_date``.

    ``report_type=2`` rows are already single-quarter. Missing quarters are
    reconstructed from the year-to-date ``report_type=1`` rows by subtracting
    the prior cumulative report. A quarter whose prior cumulative is absent
    stays NaN rather than being silently approximated by the cumulative value.
    """
    cumulative: Dict[str, float] = {}
    single: Dict[str, float] = {}
    for row in rows:
        end_date = _text(row.get("end_date"))
        value = row.get(field)
        if not end_date or value is None or pd.isna(value):
            continue
        report_type = row.get("report_type")
        if report_type == 2:
            single[end_date] = float(value)
        elif report_type == 1:
            cumulative[end_date] = float(value)

    for end_date, value in cumulative.items():
        if end_date in single:
            continue
        key = _quarter(end_date)
        if key is None:
            continue
        if key[1] == 1:
            single[end_date] = value
            continue
        previous = _previous_quarter(key)
        prior_end = next(
            (text for text in cumulative if _quarter(text) == previous),
            None,
        )
        if prior_end is not None:
            single[end_date] = value - cumulative[prior_end]
    return single


def _latest_on_or_before(mapping: Dict[str, Any], end_date: str) -> Any:
    """Value at the latest key not after ``end_date``.

    Balance-sheet items are point-in-time snapshots. A quarter whose balance
    sheet row is missing should still use the last published equity rather than
    going NaN; that is the number the market had.
    """
    candidates = [text for text in mapping if text <= end_date]
    if not candidates:
        return None
    return mapping[max(candidates)]


def _ttm_at(end_date: str, single: Dict[str, float]) -> Optional[float]:
    """Sum the last four single quarters ending at ``end_date``."""
    key = _quarter(end_date)
    if key is None:
        return None
    wanted = []
    cursor = key
    for _ in range(4):
        wanted.append(cursor)
        cursor = _previous_quarter(cursor)
    values = []
    for quarter_key in wanted:
        matching = next((text for text in single if _quarter(text) == quarter_key), None)
        if matching is None:
            return None
        values.append(single[matching])
    return float(sum(values))


def compute_report_metrics(
    income_rows: Sequence[Dict[str, Any]],
    balance_rows: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    """One row per ``(symbol, end_date)`` with TTM ROE and gross margin.

    Returns ``[symbol, end_date, disclosure_date, roe_ttm, gpm_ttm]``. The
    disclosure date is the announcement of the report that completes the TTM
    window, so a caller can align the metrics to a trading calendar without
    looking at ``end_date`` again.
    """
    income_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for row in income_rows:
        income_by_symbol.setdefault(_code(row.get("ts_code")), []).append(row)

    balance_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for row in balance_rows:
        balance_by_symbol.setdefault(_code(row.get("ts_code")), []).append(row)

    out: List[Dict[str, Any]] = []
    for symbol, rows in income_by_symbol.items():
        revenue = _single_quarter_series(rows, "revenue")
        total_revenue = _single_quarter_series(rows, "total_revenue")
        cost = _single_quarter_series(rows, "oper_cost")
        profit = _single_quarter_series(rows, "n_income_attr_p")

        disclosure_by_end = {
            _text(row.get("end_date")): _disclosure_date(row)
            for row in rows
            if _text(row.get("end_date"))
        }
        equity_by_end = {
            _text(row.get("end_date")): row.get("total_hldr_eqy_exc_min_int")
            for row in balance_by_symbol.get(symbol, [])
            if _text(row.get("end_date"))
        }

        for end_date in sorted(disclosure_by_end):
            key = _quarter(end_date)
            if key is None:
                continue
            # Prefer 营业收入, fall back to 营业总收入 when the narrower field
            # is absent (common for financial-sector reporters).
            revenue_ttm = _ttm_at(end_date, revenue)
            if revenue_ttm is None:
                revenue_ttm = _ttm_at(end_date, total_revenue)
            cost_ttm = _ttm_at(end_date, cost)
            profit_ttm = _ttm_at(end_date, profit)

            # Prior-year TTM for the YoY ratios. Missing quarters keep the
            # ratio NaN rather than falling back to a shorter window.
            prior_key = (key[0] - 1, key[1])
            prior_end = next(
                (text for text in disclosure_by_end if _quarter(text) == prior_key),
                None,
            )
            revenue_ttm_prev = (
                _ttm_at(prior_end, revenue) if prior_end is not None else None
            )
            if revenue_ttm_prev is None and prior_end is not None:
                revenue_ttm_prev = _ttm_at(prior_end, total_revenue)
            profit_ttm_prev = (
                _ttm_at(prior_end, profit) if prior_end is not None else None
            )
            revenue_yoy = None
            if revenue_ttm is not None and revenue_ttm_prev not in (None, 0.0):
                revenue_yoy = revenue_ttm / revenue_ttm_prev - 1.0
            profit_yoy = None
            if profit_ttm is not None and profit_ttm_prev not in (None, 0.0):
                # Loss-making prior quarters are common; divide by |prev| so a
                # turnaround reads positive instead of flipping the sign.
                profit_yoy = (profit_ttm - profit_ttm_prev) / abs(profit_ttm_prev)

            equity = _latest_on_or_before(equity_by_end, end_date)
            roe_ttm = None
            if profit_ttm is not None and equity is not None and not pd.isna(equity) and float(equity) != 0.0:
                roe_ttm = profit_ttm / float(equity)

            gpm_ttm = None
            if (
                revenue_ttm is not None
                and cost_ttm is not None
                and revenue_ttm != 0.0
            ):
                gpm_ttm = (revenue_ttm - cost_ttm) / revenue_ttm

            out.append(
                {
                    "symbol": symbol,
                    "end_date": end_date,
                    "disclosure_date": disclosure_by_end[end_date],
                    "roe_ttm": roe_ttm,
                    "gpm_ttm": gpm_ttm,
                    "revenue_ttm": revenue_ttm,
                    "profit_ttm": profit_ttm,
                    "revenue_yoy": revenue_yoy,
                    "profit_yoy": profit_yoy,
                }
            )

    if not out:
        return pd.DataFrame(
            columns=["symbol", "end_date", "disclosure_date", "roe_ttm", "gpm_ttm"]
        )
    return pd.DataFrame(out).sort_values(["symbol", "end_date"]).reset_index(drop=True)


_METRIC_COLUMNS = (
    "roe_ttm",
    "gpm_ttm",
    "revenue_ttm",
    "profit_ttm",
    "revenue_yoy",
    "profit_yoy",
)


def asof_expand(
    report_metrics: pd.DataFrame,
    trade_dates: Sequence[str],
    *,
    lag_days: int = 1,
) -> pd.DataFrame:
    """Forward-fill report metrics onto the trading grid, using disclosure dates.

    A report becomes visible at the first grid date strictly after its
    announcement (plus ``lag_days - 1`` extra sessions). Between reports the
    previous value is carried forward because that is still the latest
    information the market has.
    """
    dates = pd.Index(sorted({_text(date) for date in trade_dates if _text(date)}))
    empty_columns = ["trade_date", "symbol", *_METRIC_COLUMNS]
    if dates.empty or report_metrics.empty:
        return pd.DataFrame(columns=empty_columns)

    frames: List[pd.DataFrame] = []
    for symbol, group in report_metrics.groupby("symbol", sort=False):
        group = group.sort_values("disclosure_date")
        placements = []
        for _, row in group.iterrows():
            position = _asof_effective_index(row["disclosure_date"], dates, lag_days)
            if position is None:
                continue
            placements.append(
                (position, tuple(row.get(column) for column in _METRIC_COLUMNS))
            )
        if not placements:
            continue
        # Later reports at the same grid position win; a duplicate announcement
        # date is not expected, but sorting by position keeps the behaviour
        # deterministic if one appears.
        placements.sort(key=lambda item: item[0])
        values = pd.DataFrame(
            [metrics for _, metrics in placements],
            index=[position for position, _ in placements],
            columns=list(_METRIC_COLUMNS),
        )
        values = values[~values.index.duplicated(keep="last")]
        values = values.reindex(range(len(dates))).ffill()
        values["trade_date"] = dates.to_numpy()
        values["symbol"] = symbol
        frames.append(values.reset_index(drop=True))

    if not frames:
        return pd.DataFrame(columns=empty_columns)
    out = pd.concat(frames, ignore_index=True)
    return out.loc[:, empty_columns]


def _load_rows(
    collection: Any,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    projection: Dict[str, int],
) -> List[Dict[str, Any]]:
    if not symbols:
        return []
    query = {
        "ts_code": {"$in": list(symbols)},
        "end_date": {"$gte": start_date, "$lte": end_date},
    }
    return list(collection.find(query, projection))


def build_financial_panel(
    symbols: Sequence[str],
    trade_dates: Sequence[str],
    *,
    db: Any = None,
    lag_days: int = 1,
    lookback_years: int = 5,
) -> pd.DataFrame:
    """Daily PIT panel with TTM profitability and YoY growth.

    Columns: ``roe_ttm`` / ``gpm_ttm`` (quality), ``revenue_yoy`` /
    ``profit_yoy`` (growth), plus ``yoy_roe_raw``. ``yoy_roe_raw`` is
    ``roe_ttm_t / roe_ttm_{t-252} - 1``; the cross-sectional rank is applied by
    the caller so the raw quantity stays inspectable.
    """
    dates = sorted({_text(date) for date in trade_dates if _text(date)})
    if not dates:
        return pd.DataFrame(
            columns=["trade_date", "symbol", *_METRIC_COLUMNS, "yoy_roe_raw"]
        )
    start_date = f"{int(dates[0][:4]) - lookback_years}0101"
    end_date = dates[-1]

    if db is None:
        from data_manager import DatabaseManager

        db = DatabaseManager().data_db

    income_rows = _load_rows(
        db[INCOME_COLLECTION], symbols, start_date, end_date, _INCOME_PROJECTION
    )
    balance_rows = _load_rows(
        db[BALANCE_COLLECTION], symbols, start_date, end_date, _BALANCE_PROJECTION
    )
    report_metrics = compute_report_metrics(income_rows, balance_rows)
    daily = asof_expand(report_metrics, dates, lag_days=lag_days)
    if daily.empty:
        return daily.assign(yoy_roe_raw=[])

    daily = daily.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    grouped = daily.groupby("symbol", sort=False)
    for column in _METRIC_COLUMNS:
        daily[column] = grouped[column].ffill()
    shifted = daily.groupby("symbol", sort=False)["roe_ttm"].shift(252)
    daily["yoy_roe_raw"] = daily["roe_ttm"] / shifted.replace(0.0, np.nan) - 1.0
    return daily.loc[:, ["trade_date", "symbol", *_METRIC_COLUMNS, "yoy_roe_raw"]]


__all__ = [
    "asof_expand",
    "build_financial_panel",
    "compute_report_metrics",
]
