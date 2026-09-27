import sqlite3
import sys
import types
from importlib import import_module
from pathlib import Path

_STUBS_INSTALLED = False
_STUB_MODULE_NAMES = [
    "ccxt",
    "numpy",
    "pandas",
    "freqtrade",
    "freqtrade.strategy",
    "freqtrade.optimize.space",
    "freqtrade.persistence",
    "freqtrade.configuration",
]


def add_strategy_path(test_file: str | Path) -> Path:
    strategies_dir = Path(test_file).resolve().parent.parent
    if str(strategies_dir) not in sys.path:
        sys.path.insert(0, str(strategies_dir))
    return strategies_dir


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def print_info(message: str) -> None:
    print(f"INFO: {message}")


def _install_strategy_stubs() -> None:
    def _decorator(fn=None, **_kwargs):
        if fn is None:
            return lambda f: f
        return fn

    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

    def _stub(name: str):
        module = types.ModuleType(name)
        module.__codex_stub__ = True
        return module

    global _STUBS_INSTALLED
    _STUBS_INSTALLED = True

    if "ccxt" not in sys.modules:
        sys.modules["ccxt"] = _stub("ccxt")
    if "numpy" not in sys.modules:
        sys.modules["numpy"] = _stub("numpy")
    if "pandas" not in sys.modules:
        sys.modules["pandas"] = _stub("pandas")

    if "freqtrade" not in sys.modules:
        sys.modules["freqtrade"] = _stub("freqtrade")

    if "freqtrade.strategy" not in sys.modules:
        ft_strategy = _stub("freqtrade.strategy")
        ft_strategy.BooleanParameter = _Dummy
        ft_strategy.CategoricalParameter = _Dummy
        ft_strategy.DecimalParameter = _Dummy
        ft_strategy.IntParameter = _Dummy
        ft_strategy.IStrategy = object
        ft_strategy.stoploss_from_absolute = lambda *args, **kwargs: None
        ft_strategy.informative = _decorator
        ft_strategy.Order = _Dummy
        sys.modules["freqtrade.strategy"] = ft_strategy

    if "freqtrade.optimize.space" not in sys.modules:
        ft_space = _stub("freqtrade.optimize.space")
        ft_space.Categorical = _Dummy
        ft_space.Dimension = _Dummy
        ft_space.Integer = _Dummy
        ft_space.SKDecimal = _Dummy
        sys.modules["freqtrade.optimize.space"] = ft_space

    if "freqtrade.persistence" not in sys.modules:
        ft_persistence = _stub("freqtrade.persistence")
        ft_persistence.Trade = _Dummy
        sys.modules["freqtrade.persistence"] = ft_persistence

    if "freqtrade.configuration" not in sys.modules:
        ft_configuration = _stub("freqtrade.configuration")
        ft_configuration.Configuration = _Dummy
        sys.modules["freqtrade.configuration"] = ft_configuration


def load_strategy_module(require_real: bool = False):
    if require_real and _STUBS_INSTALLED:
        for name in list(_STUB_MODULE_NAMES):
            module = sys.modules.get(name)
            if module is not None and getattr(module, "__codex_stub__", False):
                sys.modules.pop(name, None)
        stubbed = sys.modules.get("DELTA_NEUTRAL")
        if stubbed is not None and getattr(stubbed, "__codex_stub__", False):
            sys.modules.pop("DELTA_NEUTRAL", None)
    try:
        module = import_module("DELTA_NEUTRAL")
    except ModuleNotFoundError:
        if require_real:
            raise
        _install_strategy_stubs()
        module = import_module("DELTA_NEUTRAL")
        module.__codex_stub__ = True
    return module


def with_strategy_globals(dn, **values):
    original = {}
    for key, value in values.items():
        original[key] = getattr(dn, key, None)
        setattr(dn, key, value)
    return original


def open_trade_count_from_db(db_path: str | Path) -> int | None:
    path = Path(db_path)
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM trades WHERE is_open = 1")
        row = cur.fetchone()
        return int(row[0]) if row else 0
    except sqlite3.Error:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass
