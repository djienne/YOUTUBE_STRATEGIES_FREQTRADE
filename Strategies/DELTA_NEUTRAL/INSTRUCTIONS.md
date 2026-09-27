# Operation and recovery

## Model and scope

Each position consists of a 1x isolated perp short and the same quantity of the
mapped spot token. Spot wrappers, basis, fees and USDC/USDT denomination differences
mean this is not a guaranteed riskless position. Positive funding is revenue,
not net profit; it must cover both legs' trading costs and any price/basis changes.
The original selection thresholds remain 10% annualized funding and 2.5 million
USDC quote volume. The 30-day replacement threshold is a policy choice, not proof
that fees will be recovered. Perp profit/loss limits remain +/-50% and stoploss -90%.

The strategy reads current market information on a 5m strategy timeframe with
`process_only_new_candles=False`; callback cadence comes from
`internals.process_throttle_secs` (60 seconds in the supplied config).
It uses settled hourly funding for selection, with complete 3-hour coverage and
24-hour coverage for the negative-funding exit. Hourly rates are converted to
simple APR percent using `rate * 24 * 365 * 100`; no compounding is implied.

Settled rates are refreshed at startup and hourly in `settled_funding_rates.json`.
Predicted observations are stored separately in `funding_observations.json`.
The old bundled `historical_funding_rates_DB.json` is a historical research
artifact and is not used as current production funding evidence.

## Configuration

Copy `user_data/config-private.example.json` to `user_data/config-private.json`.
Use the actual master account address as `walletAddress`, its API agent key as
`privateKey`, and optionally the master account's signing key as
`privateKeyEthWallet` for automatic spot/perp USDC transfers. The transfer signer
must match the configured account address. Without that key, fund/rebalance the
two wallets yourself; spot order signing still uses the API agent key.
Startup verifies that the signing key is the configured master or its registered
API agent. Setting the SDK client's account address alone does not route signed
orders to that account; a key/account mismatch is rejected.

The example config is live mode. Setting `dry_run: true` disables all custom
account reads, spot orders and transfers; it still requests public market/funding
data. Its simulated perp P&L is not a simulation of the full delta-neutral book.

The strategy sizes each entry from available perp stake divided by the remaining
slots, minus a 0.5 USDC reserve, capped by spot purchasing capacity. Therefore
`stake_amount` is not the effective per-position size. Set the capital allocation
and finite positive `max_open_trades` deliberately. Existing live positions must
be isolated 1x shorts in the configured whitelist.

The build/run commands remain:

```console
docker compose build
docker compose up -d
```

These commands are for the person deploying the project. They were not executed
as part of the audit.

## Order lifecycle

1. Entry confirmation validates eligibility and spot purchasing capacity. It
   does not buy spot or assume the perp order will fill.
2. After Freqtrade updates a filled trade, the callback checks the exchange's
   actual perp quantity and adjusts spot toward that quantity. The main loop
   also reconciles positions after restarts and missed callbacks.
3. Spot quantities are floored to venue precision. Orders use the SDK's IOC
   implementation and actual spot-market prices. Partial fills remain incomplete
   until a subsequent balance snapshot confirms their effect and any residual
   gap is reconciled. Other open orders block new external actions.
4. Exit confirmation never vetoes a perp exit because of a spot/API failure.
   After a partial exit or full close, spot is reduced to the remaining perp size.
5. Free USDC can be balanced 50/50 when slots are available: spot `total - hold`
   and perp `withdrawable` determine the amounts. The transfer must return the
   documented success response and appear in the spot balance before further
   actions are allowed. Account equity is not treated as transferable cash.

The two legs cannot fill atomically. There is exposure between fills and during
outages. A failed or partially filled hedge blocks new entries but does not erase
existing exposure. A retry may wait for the next configured bot loop. Residual
flat-account spot holdings below the venue's 10 USDC order minimum remain dust;
a material active hedge gap that cannot meet the minimum requires intervention.
The normal active-hedge tolerance is the largest of one spot lot, one estimated
spot taker fee in base units, or 0.01 USDC of base quantity.

## Ambiguous actions and restarts

`user_data/strategies/delta_neutral_state.json` is an atomic write-ahead record.
It contains the pending client order ID or transfer intent, never a signing key.
Keep it across restarts and keep it out of Git.

If a request times out or returns an unrecognized response, **do not delete the
state file or blindly repeat the order**. The action may have succeeded. Stop
this bot's new activity, inspect Hyperliquid order status/fills using the recorded
client order ID, and reconcile actual perp quantities, spot holdings, open orders
and transfers. Only after the outcome is conclusively resolved should the operator
clear that file's `pending` value to `null` (preserving its `account` field) and
resume. The next reconciliation uses actual balances rather than replaying intent.
If the file is corrupt or belongs to another account, startup refuses to use it.

Acknowledged fills/transfers with delayed balances are handled automatically once
the expected balance range appears. Ambiguous outcomes require human review;
this is intentionally conservative rather than an automatic retry policy.

## Diagnostics and manual scripts

Strategy/hedge events go to Freqtrade's configured log. The strategy writes an
hourly `equity_track.csv` using actual spot mids plus perp account equity. Missing
valuations are reported, not silently replaced with zero. Freqtrade's normal
profit report covers its perp trades and does not include all spot costs/P&L.

Run only `tests_offline` for automated validation. The old `tests/` scripts are
manual live-account experiments and may send orders or transfer money. They skip
on import and require `DELTA_NEUTRAL_ALLOW_LIVE_TESTS=1` when executed directly.
They are not the production reconciliation path and were not run during the audit.

No authenticated order placement, transfer permissions, real partial fills,
liquidation recovery or exchange latency has been validated after these changes.
