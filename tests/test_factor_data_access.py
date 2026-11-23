"""The factor worker must not import the scoring repo's data_manager."""

import pathlib

import analysis.factor.alpha_factor_backtest as ab


def test_resolve_data_db_uses_shared_helper(monkeypatch):
    sentinel = object()
    monkeypatch.setattr("stock_data_access.mongo_context.get_db", lambda *a, **k: sentinel)
    assert ab.resolve_data_db() is sentinel


def test_no_data_manager_imports_in_analysis():
    root = pathlib.Path(__file__).resolve().parents[1] / "analysis"
    hits = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if "from data_manager" in path.read_text(encoding="utf-8")
    ]
    assert not hits, f"analysis/ still imports data_manager: {hits}"
