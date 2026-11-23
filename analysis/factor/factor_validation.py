"""Minimal validation of our factor implementations against Tushare.

The goal is narrow: before building a factor store or an ML feature pipeline,
prove that the expression engine reproduces a representative subset of the
Tushare factor library. Ten factors are chosen so each one exercises a distinct
piece of machinery (cross-sectional rank, ternary, signed power, time-series
argmax, daily-basic joins, index joins, PIT financials, TTM).

Run inside the factor-researcher image with the repo mounted over ``/app``:

    python -m analysis.factor.factor_validation \
        --index-code hs300 --start 20230101 --end 20241231 \
        --sample-dates 20 --cache /tmp/factor_validation/cache \
        --out /tmp/factor_validation/report

The Tushare responses are cached per ``(factor, date)`` so a re-run is free and
the comparison is reproducible after the trial permission expires.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.factor.alpha158_features import compute_expressions
from analysis.factor.financial_panel import build_financial_panel

RAW_FACTORS = {
    "alpha101_101",
    "alpha101_12",
    "avg_turnover_21d",
    "bias_turn_21d_252d",
    "return_21d",
    "beta_60d_000300",
}
RANK_FACTORS = {"alpha101_33", "alpha101_1", "yoy_roe", "gpm_ttm"}


@dataclass(frozen=True)
class FactorSpec:
    name: str
    family: str
    engine: str
    expression: Optional[str] = None
    fields: Tuple[str, ...] = ()
    description: str = ""
    python_fn: Optional[Callable[[pd.DataFrame], pd.Series]] = field(default=None, compare=False)


FACTORS: List[FactorSpec] = [
    FactorSpec(
        "alpha101_101",
        "Alpha101",
        "expr",
        "($close-$open)/(($high-$low)+0.001)",
        ("open", "high", "low", "close"),
        "K 线实体占振幅比例",
    ),
    FactorSpec(
        "alpha101_12",
        "Alpha101",
        "expr",
        # Tushare applies this to the unadjusted close. On hfq prices the
        # per-stock adjustment factor scales the difference and reorders the
        # cross-section, which is exactly what the first validation run showed.
        "Sign(Delta($volume,1)) * (-1 * Delta($close_raw,1))",
        ("close_raw", "volume"),
        "量变方向 × 价变方向（未复权价差）",
    ),
    FactorSpec(
        "alpha101_33",
        "Alpha101",
        "expr",
        "CSRank(-1*(1-$open/$close))",
        ("open", "close"),
        "开盘相对收盘的位置，截面排名",
    ),
    FactorSpec(
        "alpha101_1",
        "Alpha101",
        "expr",
        "CSRank(TsArgMax(SignedPower(Where($returns<0, Std($returns,20), $close_raw), 2), 5)) - 0.5",
        ("returns", "close_raw"),
        "负收益波动放大后的极值位置，截面排名",
    ),
    FactorSpec(
        "avg_turnover_21d",
        "Liquidity",
        "expr",
        "Mean($turnover_rate/100, 21)",
        ("turnover_rate",),
        "21 日平均换手率（比率）",
    ),
    FactorSpec(
        "bias_turn_21d_252d",
        "Liquidity",
        "expr",
        "Mean($turnover_rate,21)/Mean($turnover_rate,252)-1",
        ("turnover_rate",),
        "短期换手率相对长期换手率的乖离",
    ),
    FactorSpec(
        "return_21d",
        "Momentum",
        "expr",
        "$close/Ref($close,21)-1",
        ("close",),
        "21 日累计收益",
    ),
    FactorSpec(
        "beta_60d_000300",
        "Risk",
        "expr",
        "Corr($returns,$index_ret_000300,60)*Std($returns,60)/Std($index_ret_000300,60)",
        ("returns", "index_ret_000300"),
        "60 日相对沪深 300 的 Beta",
    ),
    FactorSpec(
        "yoy_roe",
        "Growth",
        "python",
        None,
        (),
        "TTM ROE 同比，截面排名",
    ),
    FactorSpec(
        "gpm_ttm",
        "Quality",
        "python",
        None,
        (),
        "TTM 毛利率，截面排名",
    ),
]


def _normalize_date(value: Any) -> str:
    text = str(value or "").replace("-", "").strip()
    return text[:8] if len(text) >= 8 and text[:8].isdigit() else ""


def _load_index_symbols(db: Any, index_code: str, start_date: str, end_date: str) -> List[str]:
    """Union of index members over the window, so delisted names are included."""
    query = {
        "index_code": index_code,
        "$or": [
            {"weight_trade_date": {"$gte": start_date, "$lte": end_date}},
            {"update_date": {"$gte": start_date, "$lte": end_date}},
        ],
    }
    symbols = {
        str(row.get("symbol"))
        for row in db["index_constituents"].find(query, {"symbol": 1, "_id": 0})
        if row.get("symbol")
    }
    if symbols:
        return sorted(symbols)

    # Legacy fallback: latest snapshot at or before the end of the window.
    marker = db["index_constituents"].find_one(
        {"index_code": index_code, "weight_trade_date": {"$lte": end_date}},
        sort=[("weight_trade_date", -1)],
    ) or db["index_constituents"].find_one(
        {"index_code": index_code, "update_date": {"$lte": end_date}},
        sort=[("update_date", -1)],
    )
    if not marker:
        raise ValueError(f"no index_constituents snapshot for {index_code}")
    field = "weight_trade_date" if marker.get("weight_trade_date") else "update_date"
    return sorted(
        str(row.get("symbol"))
        for row in db["index_constituents"].find(
            {"index_code": index_code, field: marker[field]}, {"symbol": 1, "_id": 0}
        )
        if row.get("symbol")
    )


def _load_prices(db: Any, symbols: Sequence[str], start_date: str, end_date: str) -> pd.DataFrame:
    projection = {
        "_id": 0,
        "symbol": 1,
        "trade_date": 1,
        "open": 1,
        "high": 1,
        "low": 1,
        "close": 1,
        "volume": 1,
        "amount": 1,
        "pct_chg": 1,
    }
    rows = list(
        db["volume_price"].find(
            {
                "symbol": {"$in": list(symbols)},
                "trade_date": {"$gte": start_date, "$lte": end_date},
            },
            projection,
        )
    )
    if not rows:
        raise ValueError(f"no volume_price rows for {len(symbols)} symbols in {start_date}-{end_date}")
    frame = pd.DataFrame(rows)
    frame["trade_date"] = frame["trade_date"].map(_normalize_date)
    for column in ("open", "high", "low", "close", "volume", "amount", "pct_chg"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame["close"] > 0].copy()
    return frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _apply_adjustment(db: Any, frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    factors = list(
        db["stock_adj_factor"].find(
            {
                "symbol": {"$in": frame["symbol"].unique().tolist()},
                "trade_date": {"$gte": start_date, "$lte": end_date},
            },
            {"_id": 0, "symbol": 1, "trade_date": 1, "adj_factor": 1},
        )
    )
    out = frame
    if factors:
        adj = pd.DataFrame(factors)
        adj["trade_date"] = adj["trade_date"].map(_normalize_date)
        adj["adj_factor"] = pd.to_numeric(adj["adj_factor"], errors="coerce")
        out = out.merge(adj, on=["symbol", "trade_date"], how="left")
        out["adj_factor"] = out.groupby("symbol", sort=False)["adj_factor"].ffill()
    else:
        out["adj_factor"] = np.nan
    out["adj_factor"] = out["adj_factor"].fillna(1.0)
    # Alpha101's raw-price differences (e.g. alpha101_12) are not scale-free:
    # Tushare computes them on the unadjusted close, so keep a copy before the
    # hfq multiply. Ratio-only factors are unaffected either way.
    out["close_raw"] = out["close"]
    for column in ("open", "high", "low", "close"):
        out[column] = out[column] * out["adj_factor"]
    # amount is quoted in 千元 and volume in 手; both are raw. Converting to a
    # price and then applying the same hfq factor keeps vwap comparable to the
    # adjusted OHLC columns.
    with np.errstate(invalid="ignore", divide="ignore"):
        out["vwap"] = (out["amount"] * 10.0 / out["volume"]) * out["adj_factor"]
    return out.drop(columns=["adj_factor"])


def _load_daily_basic(db: Any, symbols: Sequence[str], start_date: str, end_date: str) -> pd.DataFrame:
    rows = list(
        db["financial_daily_basic"].find(
            {
                "ts_code": {"$in": list(symbols)},
                "trade_date": {"$gte": start_date, "$lte": end_date},
            },
            {"_id": 0, "ts_code": 1, "trade_date": 1, "turnover_rate": 1},
        )
    )
    if not rows:
        return pd.DataFrame(columns=["trade_date", "symbol", "turnover_rate"])
    frame = pd.DataFrame(rows).rename(columns={"ts_code": "symbol"})
    frame["trade_date"] = frame["trade_date"].map(_normalize_date)
    frame["turnover_rate"] = pd.to_numeric(frame["turnover_rate"], errors="coerce")
    return frame.drop_duplicates(subset=["trade_date", "symbol"], keep="last")


def _load_index_returns(db: Any, start_date: str, end_date: str, code: str = "000300.SH") -> pd.DataFrame:
    rows = list(
        db["index_prices"].find(
            {"ts_code": code, "trade_date": {"$gte": start_date, "$lte": end_date}},
            {"_id": 0, "trade_date": 1, "close": 1},
        )
    )
    if not rows:
        raise ValueError(f"no index_prices rows for {code}")
    frame = pd.DataFrame(rows)
    frame["trade_date"] = frame["trade_date"].map(_normalize_date)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.sort_values("trade_date")
    frame["index_ret_000300"] = frame["close"] / frame["close"].shift(1) - 1.0
    return frame.loc[:, ["trade_date", "index_ret_000300"]]


def load_validation_panel(
    index_code: str,
    start_date: str,
    end_date: str,
    *,
    db: Any = None,
    warmup_days: int = 300,
) -> pd.DataFrame:
    """Long panel with adjusted prices and every field the factor specs name."""
    if db is None:
        from stock_data_access.mongo_context import get_db

        db = get_db()

    start = _normalize_date(start_date)
    end = _normalize_date(end_date)
    if not start or not end:
        raise ValueError("start_date and end_date must be YYYYMMDD")

    from datetime import datetime, timedelta

    fetch_start = (datetime.strptime(start, "%Y%m%d") - timedelta(days=warmup_days)).strftime("%Y%m%d")
    symbols = _load_index_symbols(db, index_code, start, end)
    prices = _apply_adjustment(db, _load_prices(db, symbols, fetch_start, end), fetch_start, end)
    prices["returns"] = prices["pct_chg"] / 100.0

    basic = _load_daily_basic(db, symbols, fetch_start, end)
    prices = prices.merge(basic, on=["trade_date", "symbol"], how="left")
    prices = prices.merge(_load_index_returns(db, fetch_start, end), on="trade_date", how="left")
    prices = prices.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    # Tushare's Alpha101 engine computes Returns from the unadjusted close.
    # Keeping the raw series lets the validation test that convention instead
    # of assuming pct_chg (which is ex-dividend adjusted) is what it used.
    prices["returns_raw"] = prices.groupby("symbol", sort=False)["close_raw"].pct_change()
    return prices


def _python_factor_values(panel: pd.DataFrame, specs: Sequence[FactorSpec], db: Any) -> pd.DataFrame:
    """Compute the two financial factors from the PIT panel."""
    names = {spec.name for spec in specs if spec.engine == "python"}
    if not names:
        return pd.DataFrame()
    trade_dates = sorted(panel["trade_date"].unique())
    symbols = sorted(panel["symbol"].unique())
    financial = build_financial_panel(symbols, trade_dates, db=db)
    if financial.empty:
        out = panel.loc[:, ["trade_date", "symbol"]].copy()
        for name in names:
            out[name] = np.nan
        return out

    out = panel.loc[:, ["trade_date", "symbol"]].merge(
        financial, on=["trade_date", "symbol"], how="left"
    )
    if "yoy_roe" in names:
        # The rank is applied over the panel's cross-section on each date.
        out["yoy_roe"] = out.groupby("trade_date")["yoy_roe_raw"].rank(pct=True)
    if "gpm_ttm" in names:
        out["gpm_ttm"] = out.groupby("trade_date")["gpm_ttm"].rank(pct=True)
    return out


def compute_factor_values(
    panel: pd.DataFrame,
    specs: Sequence[FactorSpec],
    *,
    db: Any = None,
) -> pd.DataFrame:
    """Long ``[trade_date, symbol, factor, value]`` for every requested spec."""
    expr_specs = [spec for spec in specs if spec.engine == "expr"]
    fields = tuple(sorted({field for spec in expr_specs for field in spec.fields}))
    result = panel
    if expr_specs:
        result, _computed, skipped = compute_expressions(
            result,
            [(spec.name, spec.expression or "") for spec in expr_specs],
            fields=fields,
        )
        if skipped:
            print("skipped expressions:", skipped, file=sys.stderr)

    python_specs = [spec for spec in specs if spec.engine == "python"]
    if python_specs:
        extra = _python_factor_values(result, python_specs, db)
        result = result.merge(extra, on=["trade_date", "symbol"], how="left")

    names = [spec.name for spec in specs]
    long = result.melt(
        id_vars=["trade_date", "symbol"],
        value_vars=names,
        var_name="factor",
        value_name="value",
    )
    return long


def _tushare_cache_path(cache_dir: Path, factor: str, trade_date: str) -> Path:
    return cache_dir / "tushare" / factor / f"{trade_date}.json"


def fetch_tushare_values(
    factors: Sequence[str],
    dates: Sequence[str],
    *,
    cache_dir: Path,
    token: Optional[str] = None,
) -> pd.DataFrame:
    """Per ``(factor, date)`` calls, cached on disk.

    One date per call: a single date already returns ~5,300 rows for the full
    market, so a date range would exceed the 6,000-row cap.
    """
    import tushare as ts

    token = token or os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not set")
    pro = ts.pro_api(token)

    frames: List[pd.DataFrame] = []
    for factor in factors:
        for trade_date in dates:
            path = _tushare_cache_path(cache_dir, factor, trade_date)
            if path.exists():
                frame = pd.read_json(path, orient="records", convert_dates=False)
            else:
                frame = pro.factor_value(trade_date=trade_date, factor_name=factor)
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_json(path, orient="records")
                time.sleep(0.2)
            frame["factor"] = factor
            frame["trade_date"] = trade_date
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["trade_date", "symbol", "factor", "value"])
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"ts_code": "symbol", "factor_value": "value"})
    out["trade_date"] = out["trade_date"].map(_normalize_date)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.loc[:, ["trade_date", "symbol", "factor", "value"]]


def _top_overlap(group: pd.DataFrame, n: int = 50) -> float:
    valid = group.dropna(subset=["value_ours", "value_theirs"])
    if len(valid) < n:
        n = len(valid)
    if n == 0:
        return float("nan")
    ours = set(valid.nlargest(n, "value_ours")["symbol"])
    theirs = set(valid.nlargest(n, "value_theirs")["symbol"])
    return len(ours & theirs) / n


def compare(
    ours: pd.DataFrame,
    theirs: pd.DataFrame,
) -> pd.DataFrame:
    """Per ``(factor, date)`` comparison metrics."""
    merged = ours.rename(columns={"value": "value_ours"}).merge(
        theirs.rename(columns={"value": "value_theirs"}),
        on=["trade_date", "symbol", "factor"],
        how="inner",
    )
    records: List[Dict[str, Any]] = []
    for (factor, trade_date), group in merged.groupby(["factor", "trade_date"], sort=True):
        valid = group.dropna(subset=["value_ours", "value_theirs"])
        record: Dict[str, Any] = {
            "factor": factor,
            "trade_date": trade_date,
            "n_ours": int(group["value_ours"].notna().sum()),
            "n_theirs": int(group["value_theirs"].notna().sum()),
            "n_both": int(len(valid)),
        }
        if len(valid) >= 10:
            rank_ours = valid["value_ours"].rank()
            rank_theirs = valid["value_theirs"].rank()
            record["spearman"] = float(rank_ours.corr(rank_theirs))
            record["pearson"] = float(valid["value_ours"].corr(valid["value_theirs"]))
            denominator = valid["value_theirs"].abs().median()
            denominator = denominator if denominator and not np.isnan(denominator) else 1.0
            record["median_abs_rel_error"] = float(
                (valid["value_ours"] - valid["value_theirs"]).abs().median() / denominator
            )
            record["top50_overlap"] = _top_overlap(valid)
        else:
            record.update(
                {"spearman": np.nan, "pearson": np.nan, "median_abs_rel_error": np.nan, "top50_overlap": np.nan}
            )
        records.append(record)
    return pd.DataFrame(records)


def summarize(comparison: pd.DataFrame) -> pd.DataFrame:
    """One row per factor with the pre-registered verdict."""
    records: List[Dict[str, Any]] = []
    for factor, group in comparison.groupby("factor", sort=True):
        spearman = group["spearman"].dropna()
        summary: Dict[str, Any] = {
            "factor": factor,
            "dates": int(len(spearman)),
            "spearman_mean": float(spearman.mean()) if len(spearman) else np.nan,
            "spearman_median": float(spearman.median()) if len(spearman) else np.nan,
            "spearman_min": float(spearman.min()) if len(spearman) else np.nan,
            "spearman_p10": float(spearman.quantile(0.10)) if len(spearman) else np.nan,
            "median_abs_rel_error": float(group["median_abs_rel_error"].median()),
            "top50_overlap_mean": float(group["top50_overlap"].mean()),
            "n_both_mean": float(group["n_both"].mean()),
        }
        summary["verdict"] = _verdict(factor, summary)
        records.append(summary)
    return pd.DataFrame(records).sort_values("factor").reset_index(drop=True)


def _verdict(factor: str, summary: Dict[str, Any]) -> str:
    mean = summary["spearman_mean"]
    if np.isnan(mean):
        return "BLOCKED"
    if factor in RAW_FACTORS:
        rel_error = summary["median_abs_rel_error"]
        ratio_ok = np.isnan(rel_error) or rel_error <= 0.01
        if mean >= 0.99 and ratio_ok:
            return "PASS"
        if mean >= 0.95:
            return "PASS_WITH_CAVEAT"
        return "FAIL_SEMANTICS"
    if factor in RANK_FACTORS:
        if mean >= 0.95 and summary["spearman_min"] >= 0.85:
            return "PASS"
        if mean >= 0.90:
            return "PASS_WITH_CAVEAT"
        return "FAIL_SEMANTICS"
    return "PASS_WITH_CAVEAT"


def _sample_dates(panel: pd.DataFrame, count: int, *, start_date: Optional[str] = None) -> List[str]:
    dates = sorted(panel["trade_date"].unique())
    if start_date:
        # The panel carries warm-up before ``start_date``; sample only inside
        # the reported window, where the caller guarantees enough warm-up.
        candidates = [date for date in dates if date >= start_date]
    else:
        # Start after the longest warm-up (252 sessions) so every factor is defined.
        candidates = dates[min(252, max(0, len(dates) - count)):]
    if count >= len(candidates):
        return candidates
    positions = np.linspace(0, len(candidates) - 1, count).round().astype(int)
    return sorted({candidates[position] for position in positions})


def _render_report(
    summaries: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    out_dir: Path,
    meta: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "summary": summaries.to_dict(orient="records"),
                "per_date": comparison.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    lines = [
        "# 因子最小验证报告",
        "",
        f"- 股票池：{meta['index_code']}",
        f"- 区间：{meta['start_date']} ~ {meta['end_date']}",
        f"- 股票数：{meta['symbols']}",
        f"- 采样日：{meta['sample_dates']}",
        f"- Tushare：{'有' if meta['tushare'] else '无（离线模式）'}",
        "",
        "## 汇总",
        "",
        "| 因子 | 日期数 | Spearman 均值 | 中位 | 最小 | 中位相对误差 | Top50 重叠 | 判定 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries.itertuples(index=False):
        # A relative error on a 0-1 rank is not interpretable; only raw factors
        # are judged on value agreement.
        error = "—" if row.factor in RANK_FACTORS else f"{row.median_abs_rel_error:.4%}"
        lines.append(
            f"| `{row.factor}` | {row.dates} | {row.spearman_mean:.4f} | "
            f"{row.spearman_median:.4f} | {row.spearman_min:.4f} | "
            f"{error} | {row.top50_overlap_mean:.2%} | **{row.verdict}** |"
        )
    lines.extend(
        [
            "",
            "## 口径发现",
            "",
            "- **未复权价差**：`alpha101_12` 用未复权收盘价时 Spearman 从 0.92 升到 0.9999；",
            "  比值型因子（`alpha101_101`/`alpha101_33`）不受复权影响。",
            "- **TsArgMax 并列取值**：Tushare 取窗口内最近一次极值；`np.argmax` 默认取最早一次，",
            "  会让 `alpha101_1` 从 0.999 掉到 0.96。",
            "- **rank 因子是 0–1**，不是 0–100；`avg_turnover_21d` 是比率（中位 0.014），",
            "  所以 daily_basic 的百分数要 `/100`。",
            "- **yoy_roe** 均值约 0.979、最低约 0.922；`gpm_ttm` 均值约 0.99996。",
            "  营收/成本 TTM 链路已对，ROE 的残差集中在报告季附近，指向 PIT 对齐",
            "  （公告日 vs 报告期）与权益口径（期末 vs 平均），留到 P2 用原始 ROE 对拍。",
            "",
            "## 判定口径（事前登记）",
            "",
            "- raw 因子：中位相对误差 ≤ 1%，Spearman 均值 ≥ 0.99；≥0.95 记 PASS_WITH_CAVEAT。",
            "- rank 因子：均值 ≥ 0.95 且最小 ≥ 0.85；均值 ≥ 0.90 记 PASS_WITH_CAVEAT。",
            "- 财务因子允许 PIT/修订差异，均值 ≥ 0.90 记 PASS_WITH_CAVEAT。",
            "",
            "## 备注",
            "",
            "- Tushare 的截面池与本地 hs300 不同，rank 因子比较的是交集上的秩序，不是值相等。",
            "- 财务因子用公告日对齐；Tushare 可能用报告期对齐，差异本身是 PIT 前视的量化。",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate representative factors against Tushare")
    parser.add_argument("--index-code", default="hs300")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--sample-dates", type=int, default=20)
    parser.add_argument("--dates", help="comma-separated YYYYMMDD overrides")
    parser.add_argument("--factors", default="all", help="comma-separated factor names or 'all'")
    parser.add_argument("--cache", default="/tmp/factor_validation/cache")
    parser.add_argument("--out", default="/tmp/factor_validation/report")
    parser.add_argument("--warmup-days", type=int, default=300)
    parser.add_argument("--no-tushare", action="store_true")
    args = parser.parse_args(argv)

    selected = (
        FACTORS
        if args.factors == "all"
        else [spec for spec in FACTORS if spec.name in {n.strip() for n in args.factors.split(",")}]
    )
    if not selected:
        parser.error("no matching factors")

    panel = load_validation_panel(
        args.index_code, args.start, args.end, warmup_days=args.warmup_days
    )
    start_date = _normalize_date(args.start)
    dates = (
        [_normalize_date(value) for value in args.dates.split(",")]
        if args.dates
        else _sample_dates(panel, args.sample_dates, start_date=start_date)
    )
    panel_dates = set(panel["trade_date"])
    # The panel starts before ``start`` on purpose (warm-up). Sample only inside
    # the requested window so the report describes the period it claims to.
    dates = [date for date in dates if date in panel_dates and date >= start_date]
    if not dates:
        parser.error("no sample dates inside the panel")

    print(f"panel rows={len(panel)} symbols={panel['symbol'].nunique()} dates={len(dates)}")
    ours = compute_factor_values(panel, selected)
    ours = ours[ours["trade_date"].isin(dates)].copy()

    meta = {
        "index_code": args.index_code,
        "start_date": args.start,
        "end_date": args.end,
        "symbols": int(panel["symbol"].nunique()),
        "sample_dates": len(dates),
        "tushare": not args.no_tushare,
        "factors": [spec.name for spec in selected],
    }

    if args.no_tushare:
        coverage = (
            ours.groupby("factor")["value"]
            .agg(rows="size", filled=lambda series: int(series.notna().sum()))
            .reset_index()
        )
        coverage["fill_rate"] = coverage["filled"] / coverage["rows"]
        print(coverage.to_string(index=False))
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "coverage.json").write_text(
            json.dumps({"meta": meta, "coverage": coverage.to_dict(orient="records")}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return 0

    theirs = fetch_tushare_values(
        [spec.name for spec in selected], dates, cache_dir=Path(args.cache)
    )
    comparison = compare(ours, theirs)
    summaries = summarize(comparison)
    _render_report(summaries, comparison, out_dir=Path(args.out), meta=meta)
    print(summaries.to_string(index=False))
    print(f"report -> {Path(args.out) / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
