#!/bin/bash
set -e

echo "[quant-factor-research] Starting factor backtest scheduler..."
echo "[quant-factor-research] Factor backtest loop interval: ${FACTOR_BACKTEST_LOOP_INTERVAL_SECONDS:-5}s"
echo "[quant-factor-research] Factor backtest memory budget: ${FACTOR_BACKTEST_MEMORY_BUDGET_GB:-6}GB"

WORKER_ID="${FACTOR_BACKTEST_WORKER_ID:-${HOSTNAME:-quant-factor-researcher}}"
echo "[quant-factor-research] Factor backtest worker id: ${WORKER_ID}"

# One job per iteration: a CSI1000 x 158-factor run peaks near 5GB, so draining
# several back to back is how the container gets OOM killed mid-report.
exec python /app/tools/run_factor_backtest_loop.py \
  --worker-id "${WORKER_ID}" \
  --interval-seconds "${FACTOR_BACKTEST_LOOP_INTERVAL_SECONDS:-5}" \
  --limit-per-iteration "${FACTOR_BACKTEST_LOOP_LIMIT:-1}" \
  --stale-running-minutes "${FACTOR_BACKTEST_STALE_MINUTES:-120}" \
  --memory-budget-gb "${FACTOR_BACKTEST_MEMORY_BUDGET_GB:-6}"
