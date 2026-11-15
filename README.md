# quant-factor-research

Factor-backtest worker, extracted from `stock-scoring-system`. It consumes
`factor_backtest_jobs` from `quant_trading` and writes `factor_backtest_results`.

## Scope

- `analysis/factor/`: Alpha158 and related factor expressions, IC metrics,
  financial panel, factor validation.
- `analysis/core/`: `score_backtester.py` (the `PanelFactorBacktester` adapter
  uses an in-memory panel), `trade_fee_model.py`, `trailing_stop.py`.
- `tools/run_factor_backtest_loop.py` + `tools/workers/process_factor_backtest_jobs.py`.
- `factor_db.py`: factor jobs/results/catalog + worker-status collections.

## Data boundary

Reads only raw market data through `data-access-lib` (`stock_data_access.prices`,
`stock_data_access.index_constituents`) and the in-memory factor panel. It does
**not** read `finance.stock_scores`, does not import `scorers/`, and has no
dependency on `stock-scoring-system` or `quant-scorer`.

`analysis/core/score_backtester.py` is copied because `PanelFactorBacktester`
subclasses `ScoreBacktester` for the price/ledger/fee machinery. The
`ScoreBacktester` Mongo score-panel path is unused on this worker's paths; the
factor backtest feeds it an in-memory `price_cache`.

## Layout

```
analysis/factor/       factor sets, expressions, IC metrics, financial panel
analysis/core/         score_backtester / trade_fee_model / trailing_stop
analysis/reporting/    factor markdown report
tools/                 factor backtest loop + worker
utils/                 job queue + Mongo client
factor_db.py           factor job/result/catalog collections
start.factor-researcher.sh
tests/
```

## Run

```
docker build -t quant-factor-research:dev .
docker run --rm --env-file ../quant-infrastructure/apps/env/common.env \
  --env-file ../quant-infrastructure/apps/env/quant-scorer.env \
  quant-factor-research:dev
```

The worker claims one job per iteration (`FACTOR_BACKTEST_LOOP_LIMIT=1` by
default); a CSI1000 × 158-factor run peaks near 5GB.

## Tests

```
pip install -e '.[dev]'
pip install -e ./data-access-lib
pytest -q
```

## Cutover

The image replaces the `quant-factor-researcher` compose service, which
currently runs the `quant-scorer` image. Staged cutover:

1. Publish `wukongquant/quant-factor-research:sha-*` and bump
   `QUANT_FACTOR_RESEARCH_IMAGE_TAG` in `quant-infrastructure/apps/versions.env`.
2. Change the compose service image to
   `quant-factor-research:${QUANT_FACTOR_RESEARCH_IMAGE_TAG}`.
3. Deploy `quant-factor-researcher`; rollback is reverting the image to the
   scorer tag.
