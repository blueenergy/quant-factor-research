"""Append factor-backtest results to the factor registry + test ledger.

The worker writes ``factor_backtest_results`` (one blob per job). This module
turns that blob into per-factor, per-test rows in ``factor_registry`` /
``factor_tests`` so the evidence survives the job and is queryable.

Rules
-----
* Screens never overwrite a higher-stage verdict. A factor that is already
  ``candidate`` / ``rejected`` / ``no_increment`` / ``in_use`` / ``confirmed``
  keeps its status; only ``untested``/``screened`` move to ``screened``.
* Test ids are content-hashed, so re-running the same result is a no-op and
  matches the id scheme used by the 2026-11-23 backfill.
* A ledger failure is logged by the caller and never fails the backtest job.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

STAGE_SCREEN = "screen"
VERDICT_SCREEN = "screened"
SCREEN_STATUSES = {"untested", "screened", None}
TEST_PRIORITY = {"long_only": 4, "nonoverlap": 3, "ic_screen": 2, "quantile": 1}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _test_id(factor_id: str, test_type: str, as_of: str, note: str,
             protocol: Dict[str, Any]) -> str:
    key = {"factor_id": factor_id, "test_type": test_type, "as_of": as_of,
           "note": note, "protocol": protocol}
    digest = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return f"auto:{digest}"


def _protocol(pool, window, horizon, cost_bp=None, top_k=None) -> Dict[str, Any]:
    out = {"pool": pool, "window": window, "horizon": horizon, "cost_bp": cost_bp}
    if top_k is not None:
        out["top_k"] = top_k
    return out


def _test_rows(result_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    params = result_doc.get("params") or {}
    report = result_doc.get("report") or {}
    factor_set = params.get("factor_set") or (report.get("source") or {}).get("factor_set")
    if not factor_set:
        return []
    result_id = result_doc.get("result_id")
    created = result_doc.get("created_at")
    as_of = str(created or _now())[:10]
    pool = params.get("index_code")
    window = f"{params.get('start_date')}-{params.get('end_date')}"
    cost_bp = round(float((report.get("cost_assumptions") or {}).get("round_trip_rate", 0.0027)) * 10000)
    note = f"result {result_id}"
    rows: List[Dict[str, Any]] = []

    for factor, horizons in (report.get("factors") or {}).items():
        factor_id = f"{factor_set}.{factor}"
        for horizon, block in horizons.items():
            h = int(horizon)
            protocol = _protocol(pool, window, h)
            ic = block.get("ic") or {}
            if ic:
                rows.append(_row(factor_id, "ic_screen", protocol, as_of, note, result_id, {
                    k: ic.get(k) for k in ("ic_mean", "ic_std", "ic_ir", "t_stat",
                                           "positive_ratio", "avg_names", "observations")}))
            no = block.get("nonoverlap")
            if no:
                rows.append(_row(factor_id, "nonoverlap", protocol, as_of, note, result_id, {
                    k: no.get(k) for k in ("step", "observations", "ic_mean", "ic_std",
                                           "ic_ir", "t_stat", "mde80", "positive_ratio")}))
            q = block.get("quantiles")
            if q:
                rows.append(_row(factor_id, "quantile", protocol, as_of, note, result_id, {
                    "mean_return_by_quantile": q.get("mean_return_by_quantile"),
                    "top_minus_bottom_spread": q.get("top_minus_bottom_spread"),
                    "monotonic_increasing": q.get("monotonic_increasing"),
                    "quantile_rank_corr": q.get("quantile_rank_corr")}))

    for nr in report.get("net_returns") or []:
        factor_id = f"{factor_set}.{nr.get('score_column')}"
        rows.append(_row(
            factor_id, "long_only",
            _protocol(pool, window, int(nr.get("horizon") or 20), cost_bp, nr.get("top_k")),
            as_of, note, result_id,
            {"gross_ann": nr.get("long_excess_gross"), "net_ann": nr.get("long_excess_net_annual"),
             "long_short_gross": nr.get("long_short_gross"),
             "long_short_net": nr.get("long_short_net")}))
    return rows


def _row(factor_id, test_type, protocol, as_of, note, result_id, metrics) -> Dict[str, Any]:
    return {
        "_id": _test_id(factor_id, test_type, as_of, note, protocol),
        "factor_id": factor_id, "test_type": test_type, "stage": STAGE_SCREEN,
        "verdict": VERDICT_SCREEN, "protocol": protocol, "metrics": metrics,
        "as_of": as_of, "source": "worker", "note": note, "result_id": result_id,
        "reason": "automated screen", "created_at": _now(),
    }


def _ensure_registry(collections, factor_id: str, factor_set: str) -> None:
    if collections.factor_registry.find_one({"_id": factor_id}, {"_id": 1}):
        return
    collections.factor_registry.update_one(
        {"_id": factor_id},
        {"$setOnInsert": {
            "name": factor_id.split(".", 1)[-1], "source": factor_set,
            "family": None, "formula": None, "description": None, "direction": None,
            "units": None, "aliases": [], "pit_rule": None, "factor_version": None,
            "canonical": None, "status": "screened", "latest_verdict": None,
            "evidence": [], "used_by_models": [], "created_at": _now(),
        }},
        upsert=True,
    )


def _apply_tests(collections, rows: List[Dict[str, Any]]) -> int:
    by_factor: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        collections.factor_tests.replace_one({"_id": row["_id"]}, row, upsert=True)
        by_factor.setdefault(row["factor_id"], []).append(row)

    for factor_id, factor_rows in by_factor.items():
        doc = collections.factor_registry.find_one({"_id": factor_id},
                                                   {"status": 1, "evidence": 1})
        if not doc:
            continue
        status = doc.get("status")
        update: Dict[str, Any] = {
            "updated_at": _now(),
            "evidence": list(dict.fromkeys(
                [*(doc.get("evidence") or []), *(r["_id"] for r in factor_rows)]))[-20:],
        }
        if status in SCREEN_STATUSES:
            if status != "screened":
                update["status"] = "screened"
            best = sorted(factor_rows, key=lambda r: TEST_PRIORITY.get(r["test_type"], 0),
                          reverse=True)[0]
            update["latest_verdict"] = {
                "test_id": best["_id"], "verdict": best["verdict"],
                "test_type": best["test_type"], "as_of": best["as_of"],
                "reason": best.get("reason"),
            }
        collections.factor_registry.update_one({"_id": factor_id}, {"$set": update})
    return len(rows)


def record_result(collections, result_doc: Dict[str, Any]) -> Dict[str, int]:
    """Turn one ``factor_results`` document into ledger rows. Idempotent."""
    rows = _test_rows(result_doc)
    if not rows:
        return {"tests": 0, "factors": 0}
    factor_set = (result_doc.get("params") or {}).get("factor_set") or "unknown"
    for factor_id in {r["factor_id"] for r in rows}:
        _ensure_registry(collections, factor_id, factor_set)
    written = _apply_tests(collections, rows)
    return {"tests": written, "factors": len({r["factor_id"] for r in rows})}


def register_catalog_factors(collections, factor_set: str, factors: List[Dict[str, Any]]) -> int:
    """Register every catalog factor as ``untested`` if it is not present.

    Existing statuses are preserved: a factor that already has evidence must not
    fall back to ``untested`` just because the worker restarted.
    """
    n = 0
    for factor in factors:
        name = factor.get("name")
        if not name:
            continue
        factor_id = f"{factor_set}.{name}"
        collections.factor_registry.update_one(
            {"_id": factor_id},
            {
                "$set": {
                    "name": name, "source": factor_set, "family": factor.get("family"),
                    "formula": factor.get("expression"), "operator": factor.get("operator"),
                    "window": factor.get("window"), "description": factor.get("description"),
                    "canonical": {"code": "analysis/factor/alpha158_features.py",
                                  "function": "factor_catalog", "set": factor_set},
                    "updated_at": _now(),
                },
                "$setOnInsert": {
                    "status": "untested", "latest_verdict": None, "evidence": [],
                    "used_by_models": [], "created_at": _now(),
                },
            },
            upsert=True,
        )
        n += 1
    return n
