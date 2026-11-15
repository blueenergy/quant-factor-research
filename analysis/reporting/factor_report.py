#!/usr/bin/env python3
"""Generate readable reports from score backtester JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def load_backtest_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def generate_markdown_report(
    data: Dict[str, Any],
    title: str = "Score Backtest Report",
    horizons: Optional[Sequence[int]] = None,
    score_group: str = "all",
    top_n: int = 8,
) -> str:
    rows = _filtered_summary_rows(data.get("summary") or [], horizons=horizons, score_group=score_group)
    lines: List[str] = [f"# {title}", "", f"- Rows: {int(data.get('rows') or 0)}"]

    if horizons:
        lines.append(f"- Horizons: {', '.join(str(h) + 'd' for h in horizons)}")
    if score_group != "all":
        lines.append(f"- Score group: {score_group}")

    if not rows:
        lines.extend(["", "No summary rows matched the report filters."])
        return "\n".join(lines) + "\n"

    lines.extend(["", "## Key Findings"])
    lines.extend(_key_findings(rows))

    for horizon in sorted({int(r["horizon"]) for r in rows if r.get("horizon") is not None}):
        horizon_rows = [r for r in rows if int(r.get("horizon")) == horizon]
        ranked = sorted(horizon_rows, key=lambda r: _safe_number(r.get("top_bottom_spread")), reverse=True)
        lines.extend(["", f"## {horizon}d Ranking", ""])
        lines.extend(_markdown_table(ranked[:top_n]))

    inverse_rows = _inverse_rows(rows)
    if inverse_rows:
        lines.extend(["", "## Potential Inverse Signals", ""])
        lines.extend(_markdown_table(inverse_rows[:top_n]))

    return "\n".join(lines) + "\n"


def _filtered_summary_rows(
    summary: Sequence[Dict[str, Any]],
    horizons: Optional[Sequence[int]] = None,
    score_group: str = "all",
) -> List[Dict[str, Any]]:
    horizon_set = {int(h) for h in horizons} if horizons else None
    out = []
    for row in summary:
        score_column = str(row.get("score_column") or "")
        horizon = row.get("horizon")
        if horizon_set is not None and int(horizon) not in horizon_set:
            continue
        if score_group == "composite" and not score_column.startswith("composite"):
            continue
        if score_group == "atomic" and score_column.startswith("composite"):
            continue
        out.append(row)
    return out


def _key_findings(rows: Sequence[Dict[str, Any]]) -> List[str]:
    findings = []
    max_horizon = max(int(r["horizon"]) for r in rows if r.get("horizon") is not None)
    max_horizon_rows = [r for r in rows if int(r.get("horizon")) == max_horizon]
    best_horizon = max(max_horizon_rows, key=lambda r: _safe_number(r.get("top_bottom_spread")))
    best_ic = max(rows, key=lambda r: _safe_number(r.get("rank_ic_mean")))
    worst_spread = min(rows, key=lambda r: _safe_number(r.get("top_bottom_spread")))

    findings.append(
        f"- Best {max_horizon}d spread: `{best_horizon.get('score_column')}` "
        f"({_format_pct(best_horizon.get('top_bottom_spread'))}, IC {_format_float(best_horizon.get('rank_ic_mean'))})."
    )
    findings.append(
        f"- Highest Rank IC: `{best_ic.get('score_column')}` at {best_ic.get('horizon')}d "
        f"(IC {_format_float(best_ic.get('rank_ic_mean'))}, spread {_format_pct(best_ic.get('top_bottom_spread'))})."
    )
    findings.append(
        f"- Weakest spread: `{worst_spread.get('score_column')}` at {worst_spread.get('horizon')}d "
        f"({_format_pct(worst_spread.get('top_bottom_spread'))}, IC {_format_float(worst_spread.get('rank_ic_mean'))})."
    )

    inverse_count = len(_inverse_rows(rows))
    if inverse_count:
        findings.append(f"- Potential inverse signals: {inverse_count} rows have both negative IC and negative top-bottom spread.")

    return findings


def _inverse_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = [
        r
        for r in rows
        if _safe_number(r.get("rank_ic_mean")) < 0 and _safe_number(r.get("top_bottom_spread")) < 0
    ]
    return sorted(candidates, key=lambda r: _safe_number(r.get("top_bottom_spread")))


def _markdown_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "| Rank | Score Column | Horizon | Rank IC | ICIR | Top Return | Bottom Return | Top-Bottom | Observations |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    f"`{row.get('score_column')}`",
                    str(row.get("horizon")),
                    _format_float(row.get("rank_ic_mean")),
                    _format_float(row.get("rank_ic_ir")),
                    _format_pct(row.get("top_quantile_return")),
                    _format_pct(row.get("bottom_quantile_return")),
                    _format_pct(row.get("top_bottom_spread")),
                    str(row.get("ic_observations") or 0),
                ]
            )
            + " |"
        )
    return lines


def _format_float(value: Any) -> str:
    number = _to_float(value)
    return "N/A" if number is None else f"{number:+.4f}"


def _format_pct(value: Any) -> str:
    number = _to_float(value)
    return "N/A" if number is None else f"{number:+.2%}"


def _safe_number(value: Any) -> float:
    number = _to_float(value)
    return number if number is not None else float("-inf")


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a readable score backtest report.")
    parser.add_argument("input", help="Path to score_backtester JSON output")
    parser.add_argument("--output", help="Optional Markdown output path")
    parser.add_argument("--title", default="Score Backtest Report", help="Report title")
    parser.add_argument("--horizons", nargs="+", type=int, help="Optional horizons to include")
    parser.add_argument("--score-group", choices=["all", "composite", "atomic"], default="all", help="Filter score columns")
    parser.add_argument("--top-n", type=int, default=8, help="Rows per ranking section")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data = load_backtest_json(args.input)
    report = generate_markdown_report(
        data,
        title=args.title,
        horizons=args.horizons,
        score_group=args.score_group,
        top_n=args.top_n,
    )
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
