# Freqtrade strategies and experiments

Projects and source code presented on [Freqtrade FR](https://www.youtube.com/@FreqtradeFR).

## Projects

Each strategy has its own directory under [Strategies](Strategies/), with its
code, parameter exports, configurations and project files kept together.

| Project | Strategy class |
| --- | --- |
| [BB_RPB_TSL](Strategies/BB_RPB_TSL/) | `BB_RPB_TSL` |
| [BigWill](Strategies/BigWill/) | `BigWill` |
| [claude](Strategies/claude/) | `Claude` |
| [chatgpt](Strategies/chatgpt/) | `chatgpt` |
| [DELTA_NEUTRAL](Strategies/DELTA_NEUTRAL/) | `DELTA_NEUTRAL` |
| [Donchian](Strategies/Donchian/) | `donchian` |
| [Ichimoku](Strategies/Ichimoku/) | `Ichimoku` |
| [INTERMARKET_M2](Strategies/INTERMARKET_M2/) | `INTERMARKET_M2` (M2-only, spot) |
| [Market_Making](Strategies/Market_Making/) | `Market_Making` |
| [Marty_EMA](Strategies/Marty_EMA/) | `MartyEMA` |
| [SIMPLE_RSI](Strategies/SIMPLE_RSI/) | `SimpleRSI` |
| [SuperReversal](Strategies/SuperReversal/) | `SuperReversal_mtf` |
| [TRIX](Strategies/TRIX/) | `TRIX` (spot) |
| [TRIX_LS](Strategies/TRIX_LS/) | `TRIX` (futures long/short) |

The TRIX projects are separate implementations despite sharing a class name.
Project READMEs identify the source folder and active class where names differ.

## Other material

- [Configs](Configs/): generic configuration examples, separate from each project's own configs.
- [Experiences](Experiences/): research scripts, experimental strategies and historical analyses.
- [tutorial](tutorial/): introductory projects extracted from the former ZIP archive.

## Using the source

The mirrored projects retain their source code, parameters and configuration
settings. Credentials and private identifiers are replaced by placeholders;
runtime files are excluded. Run commands from the relevant project directory
and adapt the environment to your machine. Projects using `RemotePairList`
require their referenced external pairlist files/service, or your own pairlist.

Put personal settings in an ignored `user_data/config-local.json` and load it
after the shared config with a second `--config` argument. DELTA_NEUTRAL provides
`config-private.example.json` for its private configuration. Do not commit real
credentials, wallet/account identifiers, logs, trade databases or downloaded data.

Small public research inputs and figures remain alongside their analyses.
