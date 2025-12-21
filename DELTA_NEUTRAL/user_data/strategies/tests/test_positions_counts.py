import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

from _test_helpers import (
    add_strategy_path,
    assert_true,
    load_strategy_module,
    open_trade_count_from_db,
    print_info,
    with_strategy_globals,
)

HERE = Path(__file__).resolve().parent
add_strategy_path(__file__)
dn = load_strategy_module()


@contextmanager
def _patch_modules(modules: dict):
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name in modules:
            if saved[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved[name]


def _make_stub_modules(spot_positions: int, perp_positions: int):
    class StubInfo:
        def __init__(self, *args, **kwargs):
            pass

        def spot_user_state(self, address):
            balances = [{"coin": "USDC", "entryNtl": "0"}]
            for idx in range(spot_positions):
                balances.append(
                    {"coin": f"COIN{idx}", "entryNtl": "11", "total": "1"}
                )
            return {"balances": balances}

    class StubTrade:
        @staticmethod
        def get_open_trade_count():
            return perp_positions

    hyperliquid = types.ModuleType("hyperliquid")
    hyperliquid.__path__ = []
    info_module = types.ModuleType("hyperliquid.info")
    utils_module = types.ModuleType("hyperliquid.utils")
    utils_module.__path__ = []
    constants_module = types.ModuleType("hyperliquid.utils.constants")
    constants_module.MAINNET_API_URL = "https://stub.local"
    info_module.Info = StubInfo
    hyperliquid.info = info_module
    hyperliquid.utils = utils_module
    utils_module.constants = constants_module

    return {
        "hyperliquid": hyperliquid,
        "hyperliquid.info": info_module,
        "hyperliquid.utils": utils_module,
        "hyperliquid.utils.constants": constants_module,
        "freqtrade.persistence": types.ModuleType("freqtrade.persistence"),
    }, StubTrade


def _run_counts(spot_positions: int, perp_positions: int):
    modules, trade_cls = _make_stub_modules(spot_positions, perp_positions)
    modules["freqtrade.persistence"].Trade = trade_cls
    with _patch_modules(modules):
        original = with_strategy_globals(dn, GLOBAL_ADDRESS="0xTEST")
        try:
            spot_count = dn.GET_NUMBER_SPOT_POSITION()
            perp_count = trade_cls.get_open_trade_count()
        finally:
            with_strategy_globals(dn, **original)
    return spot_count, perp_count


def test_positions_count_match() -> None:
    cases = [
        {"spot": 0, "perp": 0, "match": True},
        {"spot": 1, "perp": 1, "match": True},
        {"spot": 2, "perp": 1, "match": False},
        {"spot": 1, "perp": 2, "match": False},
    ]
    for case in cases:
        spot_count, perp_count = _run_counts(case["spot"], case["perp"])
        match = spot_count == perp_count
        assert_true(match == case["match"], f"Expected match={case['match']} got {match}.")
        print_info(
            f"counts spot={spot_count} perp={perp_count} match={match}"
        )


def test_positions_count_live() -> None:
    if os.getenv("SKIP_LIVE_COUNTS") == "1":
        print_info("live count check skipped (SKIP_LIVE_COUNTS=1).")
        return
    try:
        dn_live = load_strategy_module(require_real=True)
    except ModuleNotFoundError as exc:
        raise RuntimeError("Live count requires real freqtrade dependencies.") from exc

    prev_address = dn_live.GLOBAL_ADDRESS
    dn_live.GLOBAL_ADDRESS = None
    try:
        spot_count = dn_live.GET_NUMBER_SPOT_POSITION()
    except ModuleNotFoundError as exc:
        print_info(f"SKIP: hyperliquid not installed ({exc})")
        return
    finally:
        dn_live.GLOBAL_ADDRESS = prev_address

    from freqtrade.persistence import Trade

    perp_count = None
    try:
        perp_count = Trade.get_open_trade_count()
    except AttributeError:
        pass
    if perp_count is None:
        db_path = HERE.parent.parent / "tradesv3.sqlite"
        perp_count = open_trade_count_from_db(db_path)
        if perp_count is None:
            raise RuntimeError("Unable to read perp trades count from Trade session or sqlite DB.")
        print_info("live perp count read from tradesv3.sqlite (no Trade.session).")

    print_info(f"live counts spot={spot_count} perp={perp_count} match={spot_count == perp_count}")


def main() -> None:
    tests = [
        ("test_positions_count_match", test_positions_count_match),
        ("test_positions_count_live", test_positions_count_live),
    ]
    for name, fn in tests:
        print(f"RUN: {name}")
        fn()
    print("OK")


if __name__ == "__main__":
    main()
