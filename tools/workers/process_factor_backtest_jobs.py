#!/usr/bin/env python3
"""Consume factor backtest jobs from quant_trading.

A factor backtest runs far longer than a portfolio research sweep and holds
several gigabytes while it does, so it gets its own queue and its own worker
rather than sharing the research worker's loop: one job there would otherwise
block everything queued behind it for the better part of an hour.

Reports are a couple of hundred kilobytes, well inside Mongo's document limit,
so the result is stored inline instead of on a shared volume.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional

# Run before the first project import: as a script sys.path[0] is tools/, so
# ``utils`` would not resolve yet.
from utils.job_queue import (
    HeartbeatTicker,
    JobQueue,
    WorkerLockHeld,
    clean_for_mongo,
    install_shutdown_handlers,
    set_active_job,
    utcnow,
    worker_lock,
)

WORKER_TYPE = "factor_backtest"
LOCK_LABEL = "factor backtest"

log = logging.getLogger(__name__)

# The diagnostics stage runs for tens of minutes inside a single call, so the
# stale threshold has to sit well above anything a healthy job can look like.
# The background heartbeat is what actually keeps a running job alive; this is
# only the backstop for a worker that died without releasing its job.
DEFAULT_STALE_MINUTES = 120
DEFAULT_HEARTBEAT_SECONDS = 30.0


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process pending factor backtest jobs.")
    parser.add_argument(
        "--worker-id",
        default=os.getenv("FACTOR_BACKTEST_WORKER_ID") or os.getenv("HOSTNAME") or "quant-factor-researcher",
        help="Worker id written to claimed jobs",
    )
    parser.add_argument("--limit", type=int, default=1, help="Maximum jobs to process in this run")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.getenv("FACTOR_BACKTEST_POLL_INTERVAL_SECONDS", "0")),
        help="Seconds to sleep between empty polls; 0 exits when empty",
    )
    parser.add_argument(
        "--stale-running-minutes",
        type=int,
        default=int(os.getenv("FACTOR_BACKTEST_STALE_MINUTES", str(DEFAULT_STALE_MINUTES))),
        help="Requeue running jobs whose heartbeat is older than this many minutes",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=float(os.getenv("FACTOR_BACKTEST_HEARTBEAT_SECONDS", str(DEFAULT_HEARTBEAT_SECONDS))),
        help="Background heartbeat interval so the long diagnostics stage stays live",
    )
    parser.add_argument(
        "--lock-file",
        default=os.getenv("FACTOR_BACKTEST_LOCK_FILE", "/tmp/factor_backtest_worker.lock"),
        help="Exclusive lock file to prevent overlapping worker runs",
    )
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=float(os.getenv("FACTOR_BACKTEST_MEMORY_BUDGET_GB", "6")),
        help="Default budget for jobs that do not name their own",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON payload")
    return parser.parse_args(argv)


def build_queue(collections) -> JobQueue:
    return JobQueue(
        collections.factor_jobs,
        worker_type=WORKER_TYPE,
        worker_status=getattr(collections, "worker_status", None),
    )


def publish_factor_catalog(collections) -> int:
    """Mirror the factor catalog into Mongo for the API to serve.

    The API lives in another repo and cannot import the factor definitions, so
    the worker that computes them publishes their metadata instead. Each set is
    replaced wholesale rather than merged, so a factor dropped from a set
    disappears here too instead of lingering as a name nothing can compute.
    """
    from analysis.factor.alpha_factor_backtest import FACTOR_SETS, factor_set_catalog
    from analysis.factor.factor_ledger import register_catalog_factors

    published = 0
    for factor_set in FACTOR_SETS:
        factors = factor_set_catalog(factor_set)
        register_catalog_factors(collections, factor_set, factors)
        collections.factor_catalog.replace_one(
            {"factor_set": factor_set},
            {
                "factor_set": factor_set,
                "factors": factors,
                "factor_count": len(factors),
                "updated_at": utcnow(),
            },
            upsert=True,
        )
        published += len(factors)
    return published


def job_params(job: Dict[str, Any], memory_budget_gb: Optional[float] = None):
    """Turn a job document into validated parameters.

    The budget is a property of the machine, not of the request, so a job that
    stays silent about it inherits the worker's. One that names a larger budget
    is trusted, which is how a bigger box runs what this one refuses.
    """
    from analysis.factor.alpha_factor_backtest import FactorBacktestParams

    payload = dict(job.get("params") or {})
    if memory_budget_gb is not None:
        payload.setdefault("memory_budget_gb", memory_budget_gb)
    return FactorBacktestParams.from_mapping(payload)


def run_job(job: Dict[str, Any], heartbeat=None, memory_budget_gb: Optional[float] = None) -> Dict[str, Any]:
    """Run one factor backtest and shape the result document.

    ``run_factor_backtest`` already reports (stage, message) pairs, so the
    heartbeat is wired straight to it and no stage bookkeeping is duplicated.
    """
    from analysis.factor.alpha_factor_backtest import run_factor_backtest

    params = job_params(job, memory_budget_gb)
    report = run_factor_backtest(params, heartbeat)

    result_id = f"factor_{uuid.uuid4().hex[:12]}"
    result_doc = {
        "result_id": result_id,
        "job_id": job.get("job_id"),
        "user_id": job.get("user_id"),
        "params": params.to_dict(),
        "report": report,
        "created_at": utcnow(),
    }
    return {
        "result_id": result_id,
        "result_doc": clean_for_mongo(result_doc),
        "artifact": _artifact(report),
    }


def _artifact(report: Dict[str, Any]) -> Dict[str, Any]:
    """A small summary the job list can render without loading the report."""
    source = report.get("source") or {}
    screen = (report.get("screen") or {}).get("ranked") or []
    return {
        "factor_set": source.get("factor_set"),
        "index_code": source.get("index_code"),
        "start_date": source.get("start_date"),
        "end_date": source.get("end_date"),
        "screened_factors": len(screen),
        "reported_factors": len(report.get("factors") or {}),
        "top_factor": screen[0].get("factor") if screen else None,
        "estimated_memory_gb": report.get("estimated_memory_gb"),
    }


def process_one(collections, worker_id: str, heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
                memory_budget_gb: Optional[float] = None):
    """Claim and run at most one job. Returns None when the queue is empty."""
    from analysis.factor.alpha_factor_backtest import FactorBacktestError

    queue = build_queue(collections)
    job = queue.claim(worker_id)
    if not job:
        queue.record_worker_status(worker_id, "idle", {"current_job_id": None, "last_message": "no pending job"})
        return None

    job_id = job["job_id"]
    queue.record_worker_status(
        worker_id,
        "running",
        {"current_job_id": job_id, "last_job_id": job_id, "last_message": None},
    )
    set_active_job(queue, job_id)
    ticker = HeartbeatTicker(queue, job_id, heartbeat_seconds).start()
    try:
        heartbeat = lambda stage, message=None: queue.heartbeat(job_id, stage, message)
        output = run_job(job, heartbeat=heartbeat, memory_budget_gb=memory_budget_gb)
        collections.factor_results.replace_one(
            {"result_id": output["result_id"]},
            output["result_doc"],
            upsert=True,
        )
        # The result blob is the source; the ledger is a derived, queryable
        # view. A ledger failure must not fail the backtest job.
        try:
            from analysis.factor.factor_ledger import record_result

            ledger = record_result(collections, output["result_doc"])
            log.info("factor ledger: %s tests across %s factors", ledger["tests"], ledger["factors"])
        except Exception:
            log.exception("factor ledger write failed for result %s", output["result_id"])
        queue.complete(job_id, result_id=output["result_id"], artifact=output["artifact"])
        queue.record_worker_status(
            worker_id,
            "completed",
            {"current_job_id": None, "last_job_id": job_id, "last_result_id": output["result_id"], "last_message": None},
            {"completed_count": 1, "processed_count": 1},
        )
        return {"job_id": job_id, "status": "completed", "result_id": output["result_id"]}
    except FactorBacktestError as exc:
        # A rejected request will be rejected again identically, so mark it as
        # the caller's problem rather than letting it look like a crash.
        queue.fail(job_id, exc)
        queue.jobs.update_one({"job_id": job_id}, {"$set": {"error_kind": "invalid_request"}})
        queue.record_worker_status(
            worker_id,
            "rejected",
            {"current_job_id": None, "last_job_id": job_id, "last_message": str(exc)},
            {"rejected_count": 1, "processed_count": 1},
        )
        return {"job_id": job_id, "status": "failed", "error_kind": "invalid_request", "error_message": str(exc)}
    except Exception as exc:
        queue.fail(job_id, exc)
        queue.jobs.update_one({"job_id": job_id}, {"$set": {"error_kind": "error"}})
        queue.record_worker_status(
            worker_id,
            "failed",
            {"current_job_id": None, "last_job_id": job_id, "last_message": str(exc)},
            {"failed_count": 1, "processed_count": 1},
        )
        return {"job_id": job_id, "status": "failed", "error_kind": "error", "error_message": str(exc)}
    finally:
        ticker.stop()
        set_active_job(None, None)


@contextmanager
def _worker_lock(lock_path: str):
    with worker_lock(lock_path, LOCK_LABEL):
        yield


def _run_worker(args: argparse.Namespace) -> int:
    from factor_db import ensure_factor_indexes, get_factor_collections

    install_shutdown_handlers()
    collections = get_factor_collections()
    ensure_factor_indexes(collections)
    queue = build_queue(collections)
    requeued_count = queue.requeue_stale(args.stale_running_minutes)
    if requeued_count:
        queue.record_worker_status(
            args.worker_id,
            "requeued_stale_jobs",
            {"current_job_id": None, "last_message": f"requeued {requeued_count} stale running factor jobs"},
        )

    processed = []
    while len(processed) < args.limit:
        result = process_one(collections, args.worker_id, args.heartbeat_seconds, args.memory_budget_gb)
        if result:
            processed.append(result)
            continue
        if args.poll_interval <= 0:
            break
        time.sleep(args.poll_interval)

    if processed:
        print(f"Factor backtest jobs processed: {len(processed)}")
        for item in processed:
            print(
                f"  {item['status']} job_id={item['job_id']} result_id={item.get('result_id')} "
                f"error={item.get('error_message')}"
            )
    if args.json:
        print(json.dumps({"processed": processed, "count": len(processed)}, ensure_ascii=False, indent=2, default=str))
    return 1 if any(item["status"] == "failed" for item in processed) else 0


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        with _worker_lock(args.lock_file):
            return _run_worker(args)
    except WorkerLockHeld as exc:
        print(str(exc), flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
