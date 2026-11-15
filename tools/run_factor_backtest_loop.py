#!/usr/bin/env python3
"""Continuously process factor backtest jobs with low pickup latency."""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Optional, Tuple

log = logging.getLogger("factor_backtest_loop")


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously process factor backtest jobs.")
    parser.add_argument(
        "--worker-id",
        default=os.getenv("FACTOR_BACKTEST_WORKER_ID") or os.getenv("HOSTNAME") or "quant-factor-researcher",
        help="Worker id written to claimed jobs",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=float(os.getenv("FACTOR_BACKTEST_LOOP_INTERVAL_SECONDS", "5")),
        help="Seconds to sleep between polling iterations",
    )
    parser.add_argument(
        "--limit-per-iteration",
        type=int,
        default=int(os.getenv("FACTOR_BACKTEST_LOOP_LIMIT", "1")),
        help="Jobs to drain per iteration; one job can hold several GB, so keep this at 1",
    )
    parser.add_argument(
        "--stale-running-minutes",
        type=int,
        default=int(os.getenv("FACTOR_BACKTEST_STALE_MINUTES", "120")),
        help="Requeue running jobs whose heartbeat is older than this many minutes",
    )
    parser.add_argument(
        "--stale-check-interval-seconds",
        type=float,
        default=float(os.getenv("FACTOR_BACKTEST_STALE_CHECK_INTERVAL_SECONDS", "60")),
        help="How often to look for stale running jobs",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=float(os.getenv("FACTOR_BACKTEST_HEARTBEAT_SECONDS", "30")),
        help="Background heartbeat interval while a job runs",
    )
    parser.add_argument(
        "--memory-budget-gb",
        type=float,
        default=float(os.getenv("FACTOR_BACKTEST_MEMORY_BUDGET_GB", "6")),
        help="Default budget for jobs that do not name their own",
    )
    parser.add_argument(
        "--lock-file",
        default=os.getenv("FACTOR_BACKTEST_LOCK_FILE", "/tmp/factor_backtest_worker.lock"),
        help="Exclusive lock file to prevent overlapping factor backtest workers",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=int(os.getenv("FACTOR_BACKTEST_MAX_CONSECUTIVE_FAILURES", "10")),
        help="Exit non-zero once this many iterations fail in a row",
    )
    return parser.parse_args(argv)


def _log_processed(result) -> None:
    detail = ""
    if result.get("result_id"):
        detail += f" result_id={result['result_id']}"
    if result.get("error_kind"):
        detail += f" error_kind={result['error_kind']}"
    if result.get("error_message"):
        detail += f" error={result['error_message']}"
    message = f"job processed: status={result.get('status')} job_id={result.get('job_id')}{detail}"
    # a failed job is the worker doing its job, but it still needs to stand out
    log.warning(message) if result.get("error_message") else log.info(message)


def _maybe_requeue_stale_jobs(
    queue, args: argparse.Namespace, last_stale_check: Optional[float]
) -> Tuple[Optional[float], int]:
    now = time.monotonic()
    if last_stale_check is not None and now - last_stale_check < max(1.0, float(args.stale_check_interval_seconds)):
        return last_stale_check, 0

    requeued_count = queue.requeue_stale(args.stale_running_minutes)
    if requeued_count:
        message = f"requeued {requeued_count} stale running factor jobs"
        queue.record_worker_status(args.worker_id, "requeued_stale_jobs", {"current_job_id": None, "last_message": message})
        log.warning(message)
    return now, requeued_count


def _process_iteration(
    collections, worker, queue, args: argparse.Namespace, last_stale_check: Optional[float]
) -> Tuple[int, Optional[float]]:
    last_stale_check, _ = _maybe_requeue_stale_jobs(queue, args, last_stale_check)

    processed_count = 0
    while processed_count < args.limit_per_iteration:
        result = worker.process_one(collections, args.worker_id, args.heartbeat_seconds, args.memory_budget_gb)
        if not result:
            break
        processed_count += 1
        _log_processed(result)
    return processed_count, last_stale_check


def main(argv=None) -> int:
    from observability.logging_setup import configure_worker_logging
    from factor_db import ensure_factor_indexes, get_factor_collections
    from tools.workers import process_factor_backtest_jobs as worker
    from utils.job_queue import WorkerLockHeld, run_worker_forever

    configure_worker_logging()
    args = _parse_args(argv)
    try:
        with worker._worker_lock(args.lock_file):
            worker.install_shutdown_handlers()
            collections = get_factor_collections()
            ensure_factor_indexes(collections)
            queue = worker.build_queue(collections)
            # Republished on every start so a new image ships new factors to the
            # API without a separate migration step.
            published = worker.publish_factor_catalog(collections)
            log.info("published catalog for %s factors", published)
            # An OOM killed container restarts in seconds but leaves its job
            # marked running; without this the job waits out the whole stale
            # window before anyone retries it.
            reclaimed = queue.requeue_own_orphans(args.worker_id)
            if reclaimed:
                log.warning("requeued %s job(s) left running by a previous run of %s", reclaimed, args.worker_id)
            log.info(
                "loop started: interval_seconds=%s limit_per_iteration=%s stale_running_minutes=%s memory_budget_gb=%s",
                args.interval_seconds,
                args.limit_per_iteration,
                args.stale_running_minutes,
                args.memory_budget_gb,
            )

            last_stale_check = None

            def _step() -> None:
                nonlocal last_stale_check
                _, last_stale_check = _process_iteration(collections, worker, queue, args, last_stale_check)

            return run_worker_forever(
                _step,
                interval_seconds=float(args.interval_seconds),
                max_consecutive_failures=args.max_consecutive_failures,
                logger=log,
            )
    except WorkerLockHeld as exc:
        log.info("%s; exiting without work", exc)
        return 0
    except KeyboardInterrupt:
        log.info("interrupted; shutting down")
        return 0
    except Exception:
        # previously this returned 0, so a crashed worker looked like a clean exit
        log.exception("loop aborted")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
