"""Factor ledger: result blob -> registry + test rows."""

from datetime import datetime

import mongomock
import pytest

from analysis.factor.factor_ledger import record_result, register_catalog_factors
from factor_db import get_factor_collections


@pytest.fixture
def cols():
    return get_factor_collections(mongomock.MongoClient().quant_trading_test)


def _result(result_id="result-1"):
    return {
        "result_id": result_id,
        "created_at": datetime(2026, 11, 23, 3, 0, 0),
        "params": {
            "factor_set": "alpha8", "index_code": "csi1000",
            "start_date": "20230101", "end_date": "20260728",
        },
        "report": {
            "factors": {"alpha1": {"20": {
                "ic": {"observations": 100, "ic_mean": -0.02, "t_stat": -2.0},
                "nonoverlap": {"observations": 10, "ic_mean": -0.02,
                               "mde80": 0.03, "t_stat": -1.5},
                "quantiles": {"mean_return_by_quantile": {"1": 0.001},
                              "top_minus_bottom_spread": -0.001},
            }}},
            "net_returns": [{
                "score_column": "alpha1", "horizon": 20, "top_k": 20,
                "long_excess_gross": -0.001, "long_excess_net_annual": -0.02,
                "long_short_gross": -0.002, "long_short_net": -0.003,
            }],
            "cost_assumptions": {"round_trip_rate": 0.0027},
        },
    }


def test_record_result_writes_tests_and_registry(cols):
    out = record_result(cols, _result())

    assert out == {"tests": 4, "factors": 1}
    assert cols.factor_tests.count_documents({"factor_id": "alpha8.alpha1"}) == 4
    doc = cols.factor_registry.find_one({"_id": "alpha8.alpha1"})
    assert doc["status"] == "screened"
    assert doc["source"] == "alpha8"
    # long_only is the most informative summary of a screen
    assert doc["latest_verdict"]["test_type"] == "long_only"
    assert len(doc["evidence"]) == 4


def test_record_result_is_idempotent(cols):
    record_result(cols, _result())
    record_result(cols, _result())

    assert cols.factor_tests.count_documents({"factor_id": "alpha8.alpha1"}) == 4
    assert cols.factor_registry.count_documents({}) == 1


def test_screen_does_not_downgrade_a_manual_status(cols):
    cols.factor_registry.insert_one({
        "_id": "alpha8.alpha1", "status": "candidate",
        "latest_verdict": {"verdict": "promising", "test_type": "ic_nonoverlap"},
        "evidence": [],
    })
    record_result(cols, _result())

    doc = cols.factor_registry.find_one({"_id": "alpha8.alpha1"})
    assert doc["status"] == "candidate"
    assert doc["latest_verdict"]["verdict"] == "promising"
    assert len(doc["evidence"]) == 4


def test_register_catalog_factors_preserves_status(cols):
    factor = {"name": "alpha1", "family": "K线形态", "expression": "x",
              "operator": None, "window": None, "description": "d"}
    assert register_catalog_factors(cols, "alpha8", [factor]) == 1
    assert cols.factor_registry.find_one({"_id": "alpha8.alpha1"})["status"] == "untested"

    cols.factor_registry.update_one({"_id": "alpha8.alpha1"}, {"$set": {"status": "rejected"}})
    register_catalog_factors(cols, "alpha8", [factor])
    doc = cols.factor_registry.find_one({"_id": "alpha8.alpha1"})
    assert doc["status"] == "rejected"
    assert doc["formula"] == "x"
