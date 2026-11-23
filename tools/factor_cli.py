#!/usr/bin/env python3
"""Read-only CLI over the factor registry, test ledger and model registry.

    python -m tools.factor_cli factor list --status candidate,in_use
    python -m tools.factor_cli factor show fm1.turnover_20d
    python -m tools.factor_cli model candidates --state live,shadow
    python -m tools.factor_cli model show research-csi1000-clean-raw5

Everything is a query against ``quant_trading`` and ``quant_ml``; this tool
never writes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

from pymongo import MongoClient

STATUS_ORDER = {
    "candidate": 0, "confirmed": 1, "in_use": 2, "inconclusive": 3,
    "screened": 4, "no_increment": 5, "rejected": 6, "untested": 7,
}
STATE_ORDER = {"live": 0, "shadow": 1, "validated": 2, "research": 3,
               "retired": 4, "rejected": 5}


def _dbs():
    client = MongoClient(os.environ["MONGO_URI"])
    trading = client[os.environ.get("TRADING_DB_NAME", "quant_trading")]
    ml = client[os.environ.get("QUANT_ML_DB", "quant_ml")]
    return trading, ml


def _split(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _table(headers: List[str], rows: List[List[Any]], max_width: int = 44) -> str:
    cells = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, cell in enumerate(row):
            widths[i] = min(max_width, max(widths[i], len(cell)))
    def fmt(row):
        return "  ".join(c[:widths[i]].ljust(widths[i]) for i, c in enumerate(row)).rstrip()
    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines.extend(fmt(row) for row in cells)
    return "\n".join(lines)


def _num(value, pattern="{:+.4f}", pct=False):
    if value is None:
        return "-"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{v:+.2%}" if pct else pattern.format(v)


def _metric_cell(test: Dict[str, Any]) -> str:
    m = test.get("metrics") or {}
    t = test.get("test_type")
    if t in ("long_only", "long_only_oos", "long_only_group"):
        return f"net={_num(m.get('net_ann'), pct=True)}"
    if t in ("delta_ic", "delta_ic_group"):
        return f"dIC={_num(m.get('delta_ic'))} t={_num(m.get('delta_t'), '{:+.2f}')} MDE={_num(m.get('mde80'))}"
    if t == "quantile":
        return f"spread={_num(m.get('top_minus_bottom_spread'))}"
    if t == "ic_screen":
        return f"IC={_num(m.get('ic_mean'))} t={_num(m.get('t_stat'), '{:+.2f}')}"
    if t in ("nonoverlap", "nonoverlap_screen"):
        return (f"IC={_num(m.get('ic_mean'))} t_non={_num(m.get('nonoverlap_t'), '{:+.2f}')} "
                f"MDE={_num(m.get('mde80'))}")
    if t == "ic_nonoverlap":
        tval = m.get("nonoverlap_t", m.get("nw_t"))
        return f"IC={_num(m.get('ic_mean'))} t={_num(tval, '{:+.2f}')} MDE={_num(m.get('mde80'))}"
    return ""


def _test_count(trading) -> Dict[str, int]:
    return {d["_id"]: int(d["n"]) for d in trading["factor_tests"].aggregate(
        [{"$group": {"_id": "$factor_id", "n": {"$sum": 1}}}])}


def factor_rows(trading, *, statuses=None, source=None, family=None,
                tested=None, limit=50) -> List[Dict[str, Any]]:
    query: Dict[str, Any] = {}
    if statuses:
        query["status"] = {"$in": statuses}
    if source:
        query["source"] = source
    if family:
        query["family"] = family
    docs = list(trading["factor_registry"].find(query))
    counts = _test_count(trading)
    if tested is True:
        docs = [d for d in docs if counts.get(d["_id"], 0) > 0]
    elif tested is False:
        docs = [d for d in docs if counts.get(d["_id"], 0) == 0]
    docs.sort(key=lambda d: (STATUS_ORDER.get(d.get("status"), 99), d["_id"]))
    out = []
    for d in docs[:limit]:
        lv = d.get("latest_verdict") or {}
        out.append({
            "factor_id": d["_id"], "source": d.get("source"), "family": d.get("family"),
            "status": d.get("status"), "tests": counts.get(d["_id"], 0),
            "latest": lv.get("verdict"), "latest_type": lv.get("test_type"),
            "as_of": lv.get("as_of"),
        })
    return out


def factor_detail(trading, factor_id: str, max_tests: int = 10) -> Optional[Dict[str, Any]]:
    reg = trading["factor_registry"].find_one({"_id": factor_id})
    if not reg:
        return None
    tests = list(trading["factor_tests"].find({"factor_id": factor_id})
                 .sort("as_of", -1).limit(max_tests))
    return {"registry": reg, "tests": tests}


def model_rows(ml, *, states=None, factor=None, model_type=None,
               limit=50) -> List[Dict[str, Any]]:
    query: Dict[str, Any] = {}
    if states:
        query["state"] = {"$in": states}
    if factor:
        query["feature_factors.factor_id"] = factor
    if model_type:
        query["model_type"] = model_type
    docs = list(ml["models"].find(query))
    docs.sort(key=lambda d: (STATE_ORDER.get(d.get("state"), 99), d["_id"]))
    out = []
    for d in docs[:limit]:
        status = d.get("feature_status") or {}
        out.append({
            "model_id": d["_id"], "state": d.get("state"), "accepted": d.get("accepted"),
            "type": d.get("model_type"), "features": len(d.get("feature_factors") or []),
            "source": d.get("feature_source"),
            "factor_status": ", ".join(f"{k.split('.')[-1]}={v}" for k, v in status.items()),
        })
    return out


def model_detail(ml, model_id: str) -> Optional[Dict[str, Any]]:
    doc = ml["models"].find_one({"_id": model_id})
    return doc


def _cmd_factor_list(args) -> int:
    trading, _ = _dbs()
    rows = factor_rows(trading, statuses=_split(args.status), source=args.source,
                       family=args.family, tested=args.tested, limit=args.limit)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1, default=str))
        return 0
    print(_table(
        ["factor_id", "source", "family", "status", "tests", "latest", "as_of"],
        [[r["factor_id"], r["source"], r["family"], r["status"], r["tests"],
          f"{r['latest'] or '-'}({r['latest_type'] or '-'})", r["as_of"]] for r in rows],
    ))
    return 0


def _cmd_factor_show(args) -> int:
    trading, _ = _dbs()
    detail = factor_detail(trading, args.factor_id, args.tests)
    if not detail:
        print(f"factor not found: {args.factor_id}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(detail, ensure_ascii=False, indent=1, default=str))
        return 0
    reg = detail["registry"]
    print(f"# {reg['_id']}")
    print(f"name        : {reg.get('name')}")
    print(f"source      : {reg.get('source')}   family: {reg.get('family')}")
    print(f"status      : {reg.get('status')}")
    print(f"formula     : {reg.get('formula')}")
    print(f"description : {reg.get('description')}")
    print(f"aliases     : {', '.join(reg.get('aliases') or []) or '-'}")
    print(f"used by     : {', '.join(reg.get('used_by_models') or []) or '-'}")
    lv = reg.get("latest_verdict") or {}
    print(f"latest      : {lv.get('verdict')} ({lv.get('test_type')}, {lv.get('as_of')}) {lv.get('reason') or ''}")
    print()
    print(_table(
        ["test_type", "stage", "verdict", "pool", "window", "h", "as_of", "metrics"],
        [[t.get("test_type"), t.get("stage"), t.get("verdict"),
          (t.get("protocol") or {}).get("pool"), (t.get("protocol") or {}).get("window"),
          (t.get("protocol") or {}).get("horizon"), t.get("as_of"), _metric_cell(t)]
         for t in detail["tests"]],
    ))
    return 0


def _cmd_model_list(args) -> int:
    _, ml = _dbs()
    rows = model_rows(ml, states=_split(args.state), factor=args.factor,
                      model_type=args.model_type, limit=args.limit)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1, default=str))
        return 0
    print(_table(
        ["model_id", "state", "acc", "type", "features", "source", "factor_status"],
        [[r["model_id"], r["state"], r["accepted"], r["type"], r["features"],
          r["source"], r["factor_status"]] for r in rows],
    ))
    return 0


def _cmd_model_show(args) -> int:
    _, ml = _dbs()
    doc = model_detail(ml, args.model_id)
    if not doc:
        print(f"model not found: {args.model_id}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(doc, ensure_ascii=False, indent=1, default=str))
        return 0
    print(f"# {doc['_id']}")
    print(f"state       : {doc.get('state')}   accepted: {doc.get('accepted')}   type: {doc.get('model_type')}")
    print(f"feature_src : {doc.get('feature_source')}")
    print(f"latest_val  : {json.dumps(doc.get('latest_validation'), ensure_ascii=False, default=str)}")
    print()
    print(_table(
        ["factor_id", "name", "kind", "status"],
        [[f.get("factor_id") or "-", f.get("name"), f.get("kind"),
          (doc.get("feature_status") or {}).get(f.get("factor_id")) or "-"]
         for f in doc.get("feature_factors") or []],
    ))
    evidence = doc.get("factor_evidence") or []
    if evidence:
        print()
        print(_table(
            ["factor_id", "status", "latest_verdict", "tests"],
            [[e.get("factor_id"), e.get("status"), e.get("latest_verdict"),
              len(e.get("test_ids") or [])] for e in evidence],
        ))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="factor_cli", description=__doc__)
    sub = p.add_subparsers(dest="group", required=True)

    f = sub.add_parser("factor", help="factor registry / test ledger")
    fsub = f.add_subparsers(dest="cmd", required=True)

    fl = fsub.add_parser("list", help="list factors")
    fl.add_argument("--status", help="comma-separated statuses")
    fl.add_argument("--source")
    fl.add_argument("--family")
    fl.add_argument("--tested", dest="tested", action="store_true", default=None)
    fl.add_argument("--untested", dest="tested", action="store_false")
    fl.add_argument("--limit", type=int, default=50)
    fl.add_argument("--json", action="store_true")
    fl.set_defaults(func=_cmd_factor_list)

    fs = fsub.add_parser("show", help="show one factor and its tests")
    fs.add_argument("factor_id")
    fs.add_argument("--tests", type=int, default=10)
    fs.add_argument("--json", action="store_true")
    fs.set_defaults(func=_cmd_factor_show)

    m = sub.add_parser("model", help="model candidates")
    msub = m.add_subparsers(dest="cmd", required=True)

    ml = msub.add_parser("candidates", help="list model candidates")
    ml.add_argument("--state", help="comma-separated states")
    ml.add_argument("--factor", help="only models using this factor_id")
    ml.add_argument("--model-type")
    ml.add_argument("--limit", type=int, default=50)
    ml.add_argument("--json", action="store_true")
    ml.set_defaults(func=_cmd_model_list)

    ms = msub.add_parser("show", help="show one model")
    ms.add_argument("model_id")
    ms.add_argument("--json", action="store_true")
    ms.set_defaults(func=_cmd_model_show)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
