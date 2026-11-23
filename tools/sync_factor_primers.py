#!/usr/bin/env python3
"""Materialise the factor primers into ``quant_trading.factor_registry``.

The source of truth is ``analysis/factor/primers.py``; this script expands it
into a ``primer`` sub-document on every registry document so the dashboard can
render it without a second lookup.

    python -m tools.sync_factor_primers              # dry run, prints the plan
    python -m tools.sync_factor_primers --apply      # write

Dry run is the default on purpose: this writes to the production registry, and
the text is edited by hand in code, not by hand in Mongo.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

from pymongo import MongoClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.factor.primers import primer_for  # noqa: E402

REGISTRY = "factor_registry"
FIELDS = {"_id": 1, "name": 1, "source": 1, "family": 1, "operator": 1, "window": 1}


def _registry(client) -> Any:
    db = client[os.environ.get("TRADING_DB_NAME", "quant_trading")]
    return db[REGISTRY]


def plan(coll) -> List[Dict[str, Any]]:
    """One row per registry document, with the primer that would be written."""
    rows = []
    for doc in coll.find({}, FIELDS):
        factor_id = doc["_id"]
        primer = primer_for(factor_id, doc.get("operator"), doc.get("family"),
                            doc.get("window"))
        rows.append({
            "_id": factor_id,
            "source": doc.get("source"),
            "scope": primer["scope"],
            "primer": primer,
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sync_factor_primers", description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write the primers (default: dry run)")
    args = parser.parse_args(argv)

    uri = os.environ.get("MONGO_URI")
    if not uri:
        print("MONGO_URI is not set", file=sys.stderr)
        return 2
    client = MongoClient(uri)
    coll = _registry(client)
    rows = plan(coll)

    by_scope: Dict[str, List[str]] = {}
    for row in rows:
        by_scope.setdefault(row["scope"], []).append(row["_id"])
    for scope in sorted(by_scope):
        print(f"{scope:>6}: {len(by_scope[scope])}")
    missing = by_scope.get("none", [])
    if missing:
        print("\nno primer at all:")
        for factor_id in missing:
            print(f"  {factor_id}")

    if not args.apply:
        print("\ndry run — pass --apply to write")
        return 0

    written = 0
    for row in rows:
        if row["scope"] == "none":
            continue
        primer = dict(row["primer"])
        coll.update_one({"_id": row["_id"]}, {"$set": {"primer": primer}})
        written += 1
    print(f"\nwritten: {written} / {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
