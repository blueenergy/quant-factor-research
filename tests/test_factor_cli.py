"""Read-only factor/model CLI."""

import json

import mongomock
import pytest

from tools import factor_cli


@pytest.fixture
def dbs(monkeypatch):
    client = mongomock.MongoClient()
    trading = client.quant_trading_test
    ml = client.quant_ml_test
    trading.factor_registry.insert_many([
        {"_id": "fm1.turnover_20d", "name": "turnover_20d", "source": "fm1",
         "family": "panel", "status": "candidate",
         "latest_verdict": {"verdict": "promising", "test_type": "ic_nonoverlap",
                            "as_of": "2026-10-17"},
         "coverage": {"pct": 0.947, "non_null": 100, "total": 106,
                      "factor_version": "fm1", "as_of": "2026-11-23",
                      "by_year": {"2025": 0.98, "2026": 0.99}},
         "used_by_models": ["m1"]},
        {"_id": "alpha158.WVMA30", "name": "WVMA30", "source": "alpha158",
         "family": "波动", "status": "rejected",
         "latest_verdict": {"verdict": "rejected", "test_type": "long_only",
                            "as_of": "2026-11-08"}},
        {"_id": "alpha158.alpha1", "name": "alpha1", "source": "alpha158",
         "family": "K线形态", "status": "untested", "latest_verdict": None},
    ])
    trading.factor_tests.insert_many([
        {"_id": "t1", "factor_id": "fm1.turnover_20d", "test_type": "ic_nonoverlap",
         "stage": "validation", "verdict": "promising", "as_of": "2026-10-17",
         "protocol": {"pool": "hs300", "window": "2020-2024", "horizon": 20},
         "metrics": {"ic_mean": -0.0875, "nonoverlap_t": -3.01}},
        {"_id": "t2", "factor_id": "alpha158.WVMA30", "test_type": "long_only",
         "stage": "portfolio", "verdict": "rejected", "as_of": "2026-11-08",
         "protocol": {"pool": "csi1000", "window": "2023-2026", "horizon": 10},
         "metrics": {"net_ann": 0.1043}},
    ])
    ml.models.insert_many([
        {"_id": "m1", "state": "live", "accepted": True, "model_type": "ml",
         "feature_source": "recorded",
         "feature_factors": [{"factor_id": "fm1.turnover_20d", "name": "turnover_20d",
                              "kind": "factor"}],
         "feature_status": {"fm1.turnover_20d": "candidate"},
         "factor_evidence": [{"factor_id": "fm1.turnover_20d", "status": "candidate",
                              "latest_verdict": "promising", "test_ids": ["t1"]}]},
        {"_id": "m2", "state": "shadow", "accepted": True, "model_type": "sleeve",
         "feature_source": "recorded",
         "feature_factors": [{"factor_id": "alpha158.WVMA30", "name": "WVMA30",
                              "kind": "factor"}],
         "feature_status": {"alpha158.WVMA30": "rejected"},
         "factor_evidence": []},
    ])
    monkeypatch.setattr(factor_cli, "_dbs", lambda: (trading, ml))
    return trading, ml


def test_factor_rows_filters_and_sorts(dbs):
    trading, _ = dbs
    rows = factor_cli.factor_rows(trading, statuses=["candidate", "untested"])
    assert [r["factor_id"] for r in rows] == ["fm1.turnover_20d", "alpha158.alpha1"]
    assert rows[0]["tests"] == 1
    assert rows[0]["coverage"] == 0.947
    assert rows[1]["tests"] == 0


def test_factor_rows_tested_filter(dbs):
    trading, _ = dbs
    tested = factor_cli.factor_rows(trading, tested=True)
    assert {r["factor_id"] for r in tested} == {"fm1.turnover_20d", "alpha158.WVMA30"}
    untested = factor_cli.factor_rows(trading, tested=False)
    assert [r["factor_id"] for r in untested] == ["alpha158.alpha1"]


def test_factor_detail_includes_tests(dbs):
    trading, _ = dbs
    detail = factor_cli.factor_detail(trading, "fm1.turnover_20d")
    assert detail["registry"]["status"] == "candidate"
    assert [t["_id"] for t in detail["tests"]] == ["t1"]


def test_model_rows_filter_by_factor(dbs):
    _, ml = dbs
    rows = factor_cli.model_rows(ml, factor="fm1.turnover_20d")
    assert [r["model_id"] for r in rows] == ["m1"]
    assert rows[0]["factor_status"] == "turnover_20d=candidate"


def test_cli_factor_list_json(dbs, capsys):
    assert factor_cli.main(["factor", "list", "--status", "candidate", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["factor_id"] == "fm1.turnover_20d"


def test_cli_factor_show_table(dbs, capsys):
    assert factor_cli.main(["factor", "show", "fm1.turnover_20d"]) == 0
    out = capsys.readouterr().out
    assert "# fm1.turnover_20d" in out
    assert "coverage    : 94.7%" in out
    assert "ic_nonoverlap" in out


def test_cli_model_candidates(dbs, capsys):
    assert factor_cli.main(["model", "candidates", "--state", "live"]) == 0
    out = capsys.readouterr().out
    assert "m1" in out and "m2" not in out
