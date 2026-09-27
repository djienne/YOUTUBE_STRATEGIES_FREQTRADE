# Ichimoku

Shared source from the `Ichimoku` project. The active strategy class is
`Ichimoku`; code and parameter exports are copied together from that project.
Runtime and backtest configurations retain the source settings, with credentials
and private identifiers replaced by empty values or placeholders. API and Telegram
enablement flags also match the source; configure them privately before use.

Project code, dependencies and local setup are the reader's responsibility.
Run commands from this directory. Personal overrides can go in the ignored
`user_data/config-local.json`, passed as a second `--config` argument.
No trade databases, logs, caches or downloaded market data are included.

The source configuration uses `RemotePairList`. Provide the external pairlist
files/service referenced by the configuration and Compose mounts, or configure
your own pairlist. Generated pairlist outputs are not bundled.
