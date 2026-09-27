# Funding DB and Backtest

Small utilities for pulling Hyperliquid funding rates history and running simple backtests.

## Download funding data
- Run `python get_funding_history.py`.
- Output defaults to `funding_db_test.json`.
- Optional `config_backtest.json` in this folder can set `db_path`, `coins`, and downloader options like `total_days`, `window_days`, `rate_limit_seconds`, `overlap_days`, `start_time_utc`, `end_time_utc`.
- Uses public funding-history endpoints; no credentials required.
