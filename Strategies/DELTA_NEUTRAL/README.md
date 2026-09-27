# DELTA_NEUTRAL (Hyperliquid + Freqtrade)

Short perpetuals and hold the corresponding spot tokens to collect positive
funding. This publication copy was audited in September 2026; it has **not** been
validated with authenticated orders or transfers since the changes.

The main fixes address spot orders being sent before perp confirmation, partial
fills being treated as complete hedges, equity being mistaken for withdrawable
cash, and future funding predictions contaminating historical averages.

## Files

- `user_data/strategies/DELTA_NEUTRAL.py`: funding selection and Freqtrade callbacks.
- `user_data/strategies/delta_hedge.py`: spot/perp reconciliation and USDC transfers.
- `user_data/config.json`: runtime settings; this example retains `dry_run: false`.
- `user_data/config-private.example.json`: copy to the ignored `config-private.json`
  and fill in credentials locally. Do not commit the private copy.
- `tests_offline/test_safety.py`: credential-free checks of the production code.
- `tests/`: legacy manual account experiments, now disabled unless explicitly opted in.
- `funding_db_and_backtest/`: separate research tools; ordinary Freqtrade candle
  backtesting does not simulate this live two-leg execution strategy.

## Supported setup

Use one bot process on a dedicated master account with Standard account mode
(`userAbstraction` must explicitly return `disabled`), isolated 1x USDC perps and
only the configured supported assets. Unified accounts, portfolio margin, DEX
abstraction, subaccounts/vault transfers and manually managed positions are not
supported by this collateral model. The bot does not change account mode.

Keep both strategy Python files together. `max_open_trades` in the config is the
single position-count setting; there is no separate `MAX_POSITIONS` constant.
The spot slippage limit defaults to 0.5%, configurable through
`delta_neutral_spot_slippage` (greater than zero and at most 2%).

Read [INSTRUCTIONS.md](INSTRUCTIONS.md) for operation, recovery and limitations.
Environment setup is the reader's responsibility; no running local bot was
modified during this audit.
This audited publication version now differs from the older operational source;
do not overwrite it with that older copy during a general project mirroring pass.

## Offline checks

With Freqtrade 2025.10 and its Python dependencies installed, from this folder:

```console
python -m unittest discover -s tests_offline -v
```

The suite blocks network sockets, uses fake exchange responses, and exercises
actual Freqtrade trade/order objects. It covers partial fills, delayed balances,
timeouts and restarts, rejected orders/transfers, precision and dust, spot
valuation, incomplete/future funding data, exit approval and dry-run isolation.
These checks validate control flow and arithmetic, not venue execution quality,
profitability, latency or the ability to remain hedged during an outage.

## API references checked

- [Hyperliquid account modes](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/account-abstraction-modes)
- [Spot metadata and balances](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/spot)
- [Order/transfer responses](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/exchange-endpoint)
- [Precision rules](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/tick-and-lot-size)
- [Funding history](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals)
- [Python SDK order implementation](https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/hyperliquid/exchange.py)
- [Freqtrade callbacks](https://www.freqtrade.io/en/stable/strategy-callbacks/)

Public metadata was also checked on 2026-09-27: UBTC, UETH, USOL, HYPE, UPUMP,
UFART and PURR each resolved to a unique USDC spot market with a mid price.
Market IDs are resolved from metadata at runtime, not hardcoded from that check.

Associated video: https://youtu.be/M5MzD10rc0w
