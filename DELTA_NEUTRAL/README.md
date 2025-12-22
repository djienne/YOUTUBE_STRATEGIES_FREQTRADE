# DELTA_NEUTRAL (Hyperliquid + Freqtrade)

Delta-neutral funding strategy for Hyperliquid: shorts perps, longs spot, and rebalances to stay market-neutral while harvesting positive funding.

Associated Youtube video [https://youtu.be/M5MzD10rc0w] .

## Quick start (Docker)

- `docker compose build`
- `docker compose up` (foreground logs)
- `docker compose up -d` (background)
- `docker compose down`

## Config and secrets

- `user_data/config.json`: runtime settings (pairs, max trades, fees, etc.).
- `user_data/config-private.json`: set `walletAddress`, `privateKey`, `privateKeyEthWallet`.
- Keep private keys out of git. Use a dedicated Hyperliquid hot wallet.
- If you change position slots, update both `max_open_trades` in `user_data/config.json` and `MAX_POSITIONS` in `user_data/strategies/DELTA_NEUTRAL.py`.

## Strategy notes

- Core logic lives in `user_data/strategies/DELTA_NEUTRAL.py`.
- Opens a perp short and spot long per pair; rebalances perp/spot USDC.
- Pair support is driven by `user_data/config.json` and `get_coin_info()` in the strategy.

## Logs and state

- Logs: `user_data/logs/freqtrade.log` and `user_data/strategies/delta_neutral.log`.
- Local state: `user_data/tradesv3.sqlite`.

## Tests and scripts

- `tests/` and `user_data/strategies/tests/` are script-style checks; many hit live APIs and can move funds.
- `funding_db_and_backtest/` contains funding history utilities and small backtests.
