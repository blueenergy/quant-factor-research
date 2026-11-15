"""Factor backtest worker.

``run_factor_backtest`` is stubbed throughout: what needs testing here is the
queue contract around it, not the pipeline it wraps, which has its own suite.
"""

import signal
from datetime import datetime, timedelta

import mongomock
import numpy as np
import pytest

from analysis.factor.alpha_factor_backtest import (
    ALPHA_FACTOR_COLUMNS,
    FACTOR_SETS,
    FactorBacktestError,
    factor_metadata,
    factor_set_catalog,
)
from factor_db import get_factor_collections
from tools import run_factor_backtest_loop as factor_loop
from tools.workers import process_factor_backtest_jobs as worker
from utils import job_queue


@pytest.fixture
def cols():
    return get_factor_collections(mongomock.MongoClient().quant_trading_test)


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
        job_queue.set_active_job(None, None)


def _mongo_params(**overrides):
    params = {
        "source": "mongo",
        "index_code": "hs300",
        "start_date": "20250101",
        "end_date": "20251231",
        "factor_set": "alpha158",
    }
    params.update(overrides)
    return params


def _insert_job(cols, job_id="job-1", params=None, **extra):
    cols.factor_jobs.insert_one(
        {
            "job_id": job_id,
            "user_id": "user-1",
            "status": "pending",
            "params": params if params is not None else _mongo_params(),
            "created_at": datetime.utcnow(),
            **extra,
        }
    )


def _fake_report(**overrides):
    report = {
        "factors": {"MA20": {}, "CORR10": {}},
        "screen": {"horizon": 20, "ranked": [{"factor": "MA20", "t_stat": 4.1}, {"factor": "CORR10", "t_stat": 3.2}]},
        "source": {"factor_set": "alpha158", "index_code": "hs300", "start_date": "20250101", "end_date": "20251231"},
        "estimated_memory_gb": 1.4,
    }
    report.update(overrides)
    return report


# ===================== happy path =====================
def test_process_one_stores_the_report_and_completes_the_job(cols, monkeypatch):
    _insert_job(cols)
    monkeypatch.setattr(
        "analysis.factor.alpha_factor_backtest.run_factor_backtest",
        lambda params, on_progress: _fake_report(),
    )

    out = worker.process_one(cols, "w1")

    assert out["status"] == "completed"
    job = cols.factor_jobs.find_one({"job_id": "job-1"})
    assert job["status"] == "completed"
    assert job["result_id"] == out["result_id"]
    result = cols.factor_results.find_one({"result_id": out["result_id"]})
    assert result["report"]["screen"]["ranked"][0]["factor"] == "MA20"
    assert result["user_id"] == "user-1"


def test_process_one_returns_none_on_an_empty_queue(cols):
    assert worker.process_one(cols, "w1") is None
    status = cols.worker_status.find_one({"worker_type": "factor_backtest", "worker_id": "w1"})
    assert status["status"] == "idle"


def test_completed_job_carries_a_summary_the_list_view_can_render(cols, monkeypatch):
    _insert_job(cols)
    monkeypatch.setattr(
        "analysis.factor.alpha_factor_backtest.run_factor_backtest",
        lambda params, on_progress: _fake_report(),
    )

    worker.process_one(cols, "w1")

    artifact = cols.factor_jobs.find_one({"job_id": "job-1"})["artifact"]
    assert artifact["index_code"] == "hs300"
    assert artifact["screened_factors"] == 2
    assert artifact["reported_factors"] == 2
    assert artifact["top_factor"] == "MA20"


def test_report_is_cleaned_before_it_reaches_mongo(cols, monkeypatch):
    report = _fake_report(factors={"MA20": {"ic": np.float64(0.031), "t_stat": float("nan")}})
    monkeypatch.setattr("analysis.factor.alpha_factor_backtest.run_factor_backtest", lambda params, on_progress: report)
    _insert_job(cols)

    out = worker.process_one(cols, "w1")

    stored = cols.factor_results.find_one({"result_id": out["result_id"]})["report"]
    assert stored["factors"]["MA20"]["ic"] == pytest.approx(0.031)
    assert stored["factors"]["MA20"]["t_stat"] is None


# ===================== progress =====================
def test_progress_from_the_pipeline_lands_on_the_job_document(cols, monkeypatch):
    _insert_job(cols)

    def fake_run(params, on_progress):
        on_progress("loading_prices", "reading prices")
        on_progress("running_diagnostics", "quantile diagnostics on 20 factors")
        return _fake_report()

    monkeypatch.setattr("analysis.factor.alpha_factor_backtest.run_factor_backtest", fake_run)

    worker.process_one(cols, "w1")

    job = cols.factor_jobs.find_one({"job_id": "job-1"})
    assert job["progress_stage"] == "running_diagnostics"
    assert job["progress_message"] == "quantile diagnostics on 20 factors"


# ===================== parameters =====================
def test_job_params_inherit_the_worker_memory_budget(cols):
    params = worker.job_params({"params": _mongo_params()}, memory_budget_gb=12.0)
    assert params.memory_budget_gb == 12.0


def test_job_params_let_a_job_ask_for_a_bigger_budget(cols):
    params = worker.job_params({"params": _mongo_params(memory_budget_gb=24.0)}, memory_budget_gb=6.0)
    assert params.memory_budget_gb == 24.0


def test_job_params_reject_keys_the_pipeline_would_ignore():
    with pytest.raises(FactorBacktestError, match="unknown parameters: factorset"):
        worker.job_params({"params": _mongo_params(factorset="alpha158")})


# ===================== failures =====================
def test_a_rejected_request_is_marked_as_the_caller_s_problem(cols, monkeypatch):
    # 158 factors over 16 years on csi1000 cannot fit, and retrying will not help.
    _insert_job(cols, params=_mongo_params(index_code="csi1000", start_date="20100101", end_date="20260101"))

    out = worker.process_one(cols, "w1")

    assert out["status"] == "failed"
    assert out["error_kind"] == "invalid_request"
    job = cols.factor_jobs.find_one({"job_id": "job-1"})
    assert job["status"] == "failed"
    assert job["error_kind"] == "invalid_request"
    assert "GB budget" in job["error_message"]


def test_the_budget_is_enforced_before_any_prices_are_read(cols, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "analysis.factor.alpha_factor_backtest.load_source_panel",
        lambda *a, **k: calls.append(1),
    )
    _insert_job(cols, params=_mongo_params(index_code="csi1000", start_date="20100101", end_date="20260101"))

    worker.process_one(cols, "w1")

    assert calls == []


def test_a_crash_fails_the_job_without_taking_the_worker_down(cols, monkeypatch):
    _insert_job(cols)

    def boom(params, on_progress):
        raise MemoryError("unable to allocate 2.1 GiB")

    monkeypatch.setattr("analysis.factor.alpha_factor_backtest.run_factor_backtest", boom)

    out = worker.process_one(cols, "w1")

    assert out["status"] == "failed"
    assert out["error_kind"] == "error"
    job = cols.factor_jobs.find_one({"job_id": "job-1"})
    assert "2.1 GiB" in job["error_message"]
    status = cols.worker_status.find_one({"worker_type": "factor_backtest", "worker_id": "w1"})
    assert status["failed_count"] == 1


def test_a_failed_job_is_not_retried_on_the_next_poll(cols, monkeypatch):
    _insert_job(cols)
    monkeypatch.setattr(
        "analysis.factor.alpha_factor_backtest.run_factor_backtest",
        lambda params, on_progress: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert worker.process_one(cols, "w1")["status"] == "failed"
    assert worker.process_one(cols, "w1") is None


# ===================== queue isolation and lifecycle =====================
def test_factor_jobs_and_research_jobs_are_separate_queues(cols, monkeypatch):
    _insert_job(cols, "factor-1")
    cols.research_jobs.insert_one(
        {"job_id": "research-1", "status": "pending", "created_at": datetime.utcnow()}
    )
    monkeypatch.setattr("analysis.factor.alpha_factor_backtest.run_factor_backtest", lambda params, on_progress: _fake_report())

    worker.process_one(cols, "w1")

    # The factor worker must never touch a portfolio research job.
    assert cols.research_jobs.find_one({"job_id": "research-1"})["status"] == "pending"


def test_process_one_clears_the_active_job(cols, monkeypatch):
    _insert_job(cols)
    monkeypatch.setattr("analysis.factor.alpha_factor_backtest.run_factor_backtest", lambda params, on_progress: _fake_report())

    worker.process_one(cols, "w1")

    assert job_queue.active_job_id() is None


def test_shutdown_mid_job_requeues_it_instead_of_orphaning_it(cols, monkeypatch):
    _insert_job(cols)

    def interrupted(params, on_progress):
        with pytest.raises(SystemExit):
            job_queue.handle_shutdown_signal(signal.SIGTERM, None)
        return _fake_report()

    monkeypatch.setattr("analysis.factor.alpha_factor_backtest.run_factor_backtest", interrupted)

    worker.process_one(cols, "w1")

    # The signal handler hands the job back while it is still running.
    assert cols.factor_jobs.find_one({"job_id": "job-1"})["error_message"] is not None


def test_worker_uses_a_stale_threshold_that_outlasts_the_diagnostics_stage():
    # The diagnostics stage alone has been measured near 50 minutes.
    assert worker.DEFAULT_STALE_MINUTES >= 120
    args = worker._parse_args([])
    assert args.stale_running_minutes >= 120
    # Draining several jobs per run would stack their memory.
    assert args.limit == 1


def test_loop_defaults_to_one_job_per_iteration():
    args = factor_loop._parse_args([])
    assert args.limit_per_iteration == 1
    assert args.stale_running_minutes >= 120
    assert args.lock_file != "/tmp/portfolio_research_worker.lock"


# ===================== stale requeue through the loop =====================
def test_loop_requeues_a_job_whose_worker_died(cols):
    old = datetime.utcnow() - timedelta(minutes=180)
    cols.factor_jobs.insert_one({"job_id": "orphan", "status": "running", "heartbeat_at": old})
    queue = worker.build_queue(cols)
    args = factor_loop._parse_args([])

    last_check, requeued = factor_loop._maybe_requeue_stale_jobs(queue, args, None)

    assert requeued == 1
    assert last_check is not None
    assert cols.factor_jobs.find_one({"job_id": "orphan"})["status"] == "pending"


def test_loop_does_not_requeue_a_job_that_is_merely_slow(cols):
    beating = datetime.utcnow() - timedelta(minutes=45)
    cols.factor_jobs.insert_one({"job_id": "slow", "status": "running", "heartbeat_at": beating})
    queue = worker.build_queue(cols)

    factor_loop._maybe_requeue_stale_jobs(queue, factor_loop._parse_args([]), None)

    assert cols.factor_jobs.find_one({"job_id": "slow"})["status"] == "running"


def test_loop_throttles_how_often_it_scans_for_stale_jobs(cols):
    queue = worker.build_queue(cols)
    args = factor_loop._parse_args([])

    last_check, _ = factor_loop._maybe_requeue_stale_jobs(queue, args, None)
    same_check, requeued = factor_loop._maybe_requeue_stale_jobs(queue, args, last_check)

    assert same_check == last_check
    assert requeued == 0


# ===================== end to end =====================
def test_worker_runs_the_real_pipeline_and_stores_a_usable_report(cols, tmp_path):
    """No stub: a real (small) backtest, to prove the wiring end to end."""
    from tests.test_alpha_factor_backtest import _raw_panel

    panel = tmp_path / "panel.csv"
    _raw_panel(symbols=[f"{600000 + i:06d}.SH" for i in range(40)], n_days=60).to_csv(panel, index=False)
    _insert_job(
        cols,
        params={
            "source": "panel",
            "panel": str(panel),
            "factor_set": "alpha8",
            "horizons": [1, 5],
            "min_names": 10,
            "top_k": [5],
        },
    )

    out = worker.process_one(cols, "w1", heartbeat_seconds=0)

    assert out["status"] == "completed", out
    report = cols.factor_results.find_one({"result_id": out["result_id"]})["report"]
    assert report["factors"]
    assert report["params"]["factor_set"] == "alpha8"
    assert report["source"]["point_in_time_universe"] is False
    # The stage the job finished on is the last one the pipeline reports.
    assert cols.factor_jobs.find_one({"job_id": "job-1"})["progress_stage"] == "cost_adjustment"


def test_loop_reclaims_its_own_orphan_before_polling(cols, monkeypatch):
    """An OOM killed pod restarts under the same name and must not wait it out."""
    cols.factor_jobs.insert_one(
        {"job_id": "orphan", "status": "running", "worker_id": "pod-a", "heartbeat_at": datetime.utcnow()}
    )
    queue = worker.build_queue(cols)

    assert queue.requeue_stale(120) == 0
    assert queue.requeue_own_orphans("pod-a") == 1
    assert cols.factor_jobs.find_one({"job_id": "orphan"})["status"] == "pending"


def test_publish_factor_catalog_writes_one_document_per_set(cols):
    published = worker.publish_factor_catalog(cols)

    docs = {doc["factor_set"]: doc for doc in cols.factor_catalog.find({})}
    assert set(docs) == set(FACTOR_SETS)
    assert docs["alpha8"]["factor_count"] == len(ALPHA_FACTOR_COLUMNS)
    assert docs["alpha158"]["factor_count"] == 158
    assert published == docs["alpha8"]["factor_count"] + docs["alpha158"]["factor_count"]
    entry = docs["alpha158"]["factors"][0]
    assert set(entry) == {"name", "expression", "family", "operator", "window", "description"}


def test_publish_factor_catalog_replaces_instead_of_accumulating(cols):
    """Every worker start republishes, so the set must not grow a second copy."""
    worker.publish_factor_catalog(cols)
    cols.factor_catalog.update_one({"factor_set": "alpha8"}, {"$set": {"factors": [], "factor_count": 0}})

    worker.publish_factor_catalog(cols)

    assert cols.factor_catalog.count_documents({}) == len(FACTOR_SETS)
    assert cols.factor_catalog.find_one({"factor_set": "alpha8"})["factor_count"] == len(ALPHA_FACTOR_COLUMNS)


def test_alpha8_catalog_covers_every_computed_column():
    """A factor the pipeline computes must not be missing from the picker."""
    names = [entry["name"] for entry in factor_set_catalog("alpha8")]

    assert names == list(ALPHA_FACTOR_COLUMNS)


def test_factor_set_catalog_rejects_an_unknown_set():
    with pytest.raises(FactorBacktestError):
        factor_set_catalog("alpha42")


def test_factor_metadata_resolves_names_from_either_set():
    names = [entry["name"] for entry in factor_metadata(["alpha5", "KMID", "not-a-factor"])]

    assert names == ["alpha5", "KMID"]


def test_params_accept_the_key_set_the_api_enqueues():
    """Contract with quantFinance, which cannot import this module to check.

    ``from_mapping`` refuses an unknown key, so renaming a field here breaks job
    creation over there. Keep this list in step with
    ``routers/factor_backtest.py::CreateFactorBacktestJobRequest``.
    """
    enqueued = {
        "source": "mongo",
        "index_code": "csi1000",
        "start_date": "20200101",
        "end_date": "20241231",
        "factor_set": "alpha158",
        "factors": None,
        "warmup_days": 60,
        "horizons": [1, 5, 10, 20],
        "quantiles": 5,
        "min_names": 30,
        "top_k": [20, 50],
        "screen_top": 20,
        "screen_horizon": None,
        "slippage_bps": 10.0,
        "turnover": 1.0,
    }

    params = worker.job_params({"params": enqueued}, memory_budget_gb=20.0)

    assert params.source == "mongo"
    assert params.index_code == "csi1000"
    # The API sends no budget, so the worker's own is what applies.
    assert params.memory_budget_gb == 20.0
