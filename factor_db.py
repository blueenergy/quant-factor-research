"""Trading DB accessors for the factor-backtest worker.

Only the collections this worker owns: factor jobs/results/catalog and the
shared worker-status registry. This replaces the old dependency on
``portfolio.db.get_trading_collections()``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from pymongo import ASCENDING, DESCENDING

from utils.mongo_client import create_mongo_client


@dataclass(frozen=True)
class FactorCollections:
    factor_jobs: Any
    factor_results: Any
    factor_catalog: Any
    worker_status: Any
    # Present only so the queue-isolation regression test can prove the worker
    # never touches the research queues; the worker itself does not read them.
    research_jobs: Any
    research_results: Any


def get_trading_db(client=None):
    client = client or create_mongo_client()
    return client[os.getenv("TRADING_DB_NAME", "quant_trading")]


def get_factor_collections(db=None) -> FactorCollections:
    db = db if db is not None else get_trading_db()
    return FactorCollections(
        factor_jobs=db["factor_backtest_jobs"],
        factor_results=db["factor_backtest_results"],
        factor_catalog=db["factor_catalog"],
        worker_status=db["portfolio_worker_status"],
        research_jobs=db["portfolio_research_jobs"],
        research_results=db["portfolio_research_results"],
    )


def ensure_factor_indexes(cols: FactorCollections) -> None:
    """Create the indexes the job queue and factor API expect (idempotent)."""
    cols.worker_status.create_index(
        [("worker_type", ASCENDING), ("worker_id", ASCENDING)], unique=True
    )
    cols.worker_status.create_index([("last_seen_at", DESCENDING)])
    cols.factor_jobs.create_index([("job_id", ASCENDING)], unique=True)
    cols.factor_jobs.create_index([("status", ASCENDING), ("created_at", ASCENDING)])
    cols.factor_jobs.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    cols.factor_results.create_index([("result_id", ASCENDING)], unique=True)
    cols.factor_results.create_index([("job_id", ASCENDING)])
    cols.factor_results.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])


__all__ = [
    "FactorCollections",
    "ensure_factor_indexes",
    "get_factor_collections",
    "get_trading_db",
]
