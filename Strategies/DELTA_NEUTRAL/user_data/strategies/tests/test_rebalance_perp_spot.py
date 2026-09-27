import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
import re

from _test_helpers import add_strategy_path, assert_true, load_strategy_module, print_info

HERE = Path(__file__).resolve().parent
add_strategy_path(__file__)
dn = load_strategy_module()

BALANCE_RE = re.compile(r"Spot:\s*([0-9.]+),\s*Perp:\s*([0-9.]+),\s*target\s*([0-9.]+)")
TRANSFER_RE = re.compile(r"Transfer result:\s*(.+)")

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


def _make_stub_modules(spot_balance: float, perp_balance: float, transfers: list):
    class StubInfo:
        def __init__(self, *args, **kwargs):
            pass

        def spot_user_state(self, address):
            return {"balances": [{"coin": "USDC", "total": spot_balance}]}

        def user_state(self, address):
            return {
                "crossMarginSummary": {"accountValue": perp_balance},
                "marginSummary": {"accountValue": perp_balance},
            }

    class StubExchange:
        def __init__(self, *args, **kwargs):
            pass

        def usd_class_transfer(self, amount, to_perp):
            transfers.append({"amount": float(amount), "to_perp": bool(to_perp)})
            return {"status": "ok", "amount": float(amount), "to_perp": bool(to_perp)}

    class StubAccount:
        @staticmethod
        def from_key(_key):
            return StubLocalAccount()

    class StubLocalAccount:
        pass

    hyperliquid = types.ModuleType("hyperliquid")
    hyperliquid.__path__ = []
    exchange_module = types.ModuleType("hyperliquid.exchange")
    info_module = types.ModuleType("hyperliquid.info")
    utils_module = types.ModuleType("hyperliquid.utils")
    utils_module.__path__ = []
    constants_module = types.ModuleType("hyperliquid.utils.constants")
    constants_module.MAINNET_API_URL = "https://stub.local"
    exchange_module.Exchange = StubExchange
    info_module.Info = StubInfo
    hyperliquid.exchange = exchange_module
    hyperliquid.info = info_module
    hyperliquid.utils = utils_module
    utils_module.constants = constants_module

    eth_account = types.ModuleType("eth_account")
    eth_account.__path__ = []
    eth_account.Account = StubAccount
    eth_signers = types.ModuleType("eth_account.signers")
    eth_signers.__path__ = []
    eth_signers_local = types.ModuleType("eth_account.signers.local")
    eth_signers_local.LocalAccount = StubLocalAccount
    eth_account.signers = eth_signers
    eth_signers.local = eth_signers_local

    return {
        "hyperliquid": hyperliquid,
        "hyperliquid.exchange": exchange_module,
        "hyperliquid.info": info_module,
        "hyperliquid.utils": utils_module,
        "hyperliquid.utils.constants": constants_module,
        "eth_account": eth_account,
        "eth_account.signers": eth_signers,
        "eth_account.signers.local": eth_signers_local,
    }


def _run_rebalance(spot_balance: float, perp_balance: float):
    transfers = []
    modules = _make_stub_modules(spot_balance, perp_balance, transfers)
    with _patch_modules(modules):
        prev_address = dn.GLOBAL_ADDRESS
        prev_wallet = dn.PRIVATE_EVM_WALLET
        dn.GLOBAL_ADDRESS = "0xTEST"
        dn.PRIVATE_EVM_WALLET = "0xTEST"
        try:
            result = dn.REBALANCE_PERP_SPOT()
        finally:
            dn.GLOBAL_ADDRESS = prev_address
            dn.PRIVATE_EVM_WALLET = prev_wallet
    return result, transfers


def _delta_log_path() -> Path:
    try:
        base = Path(dn.__file__).resolve().parent
    except Exception:
        base = HERE.parent
    return base / "delta_neutral.log"


def _read_appended_lines(log_path: Path, start_size: int):
    if not log_path.exists():
        return []
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(start_size)
            return handle.read().splitlines()
    except OSError:
        return []


def _find_latest_matching_line(lines, regex):
    for line in reversed(lines):
        if regex.search(line):
            return line
    return None


def _print_latest_balances(log_path: Path, start_size: int) -> None:
    appended = _read_appended_lines(log_path, start_size)
    line = _find_latest_matching_line(appended, BALANCE_RE)
    if line is None and log_path.exists():
        line = _find_latest_matching_line(log_path.read_text(errors="ignore").splitlines(), BALANCE_RE)
    if line is None:
        print_info("live balances not found in delta_neutral.log")
        return
    match = BALANCE_RE.search(line)
    if not match:
        print_info("live balances not found in delta_neutral.log")
        return
    spot, perp, target = match.groups()
    print_info(f"live balances spot={spot} perp={perp} target={target}")


def _print_latest_transfer(log_path: Path, start_size: int) -> None:
    appended = _read_appended_lines(log_path, start_size)
    line = _find_latest_matching_line(appended, TRANSFER_RE)
    if line is None and log_path.exists():
        line = _find_latest_matching_line(log_path.read_text(errors="ignore").splitlines(), TRANSFER_RE)
    if line is None:
        print_info("live transfer result not found in delta_neutral.log")
        return
    match = TRANSFER_RE.search(line)
    if not match:
        print_info("live transfer result not found in delta_neutral.log")
        return
    print_info(f"live transfer result {match.group(1)}")


def _assert_transfer(result, transfers, expected_result, expected_transfer):
    assert_true(result is expected_result, f"Expected result {expected_result}, got {result}.")
    if expected_transfer is None:
        assert_true(not transfers, f"Expected no transfer, got {transfers}.")
        return
    assert_true(len(transfers) == 1, f"Expected one transfer, got {transfers}.")
    transfer = transfers[0]
    expected_amount, expected_to_perp = expected_transfer
    assert_true(
        abs(transfer["amount"] - expected_amount) < 1e-9,
        f"Expected amount {expected_amount}, got {transfer['amount']}.",
    )
    assert_true(
        transfer["to_perp"] == expected_to_perp,
        f"Expected to_perp={expected_to_perp}, got {transfer['to_perp']}.",
    )


def test_rebalance_scenarios() -> None:
    cases = [
        {
            "name": "within_threshold",
            "spot": 100.0,
            "perp": 100.0,
            "expected_result": False,
            "expected_transfer": None,
        },
        {
            "name": "spot_to_perp",
            "spot": 150.0,
            "perp": 50.0,
            "expected_result": True,
            "expected_transfer": (50.0, True),
        },
        {
            "name": "perp_to_spot",
            "spot": 50.0,
            "perp": 150.0,
            "expected_result": True,
            "expected_transfer": (50.0, False),
        },
        {
            "name": "low_total_skip",
            "spot": 15.0,
            "perp": 5.0,
            "expected_result": False,
            "expected_transfer": None,
        },
    ]

    for case in cases:
        result, transfers = _run_rebalance(case["spot"], case["perp"])
        _assert_transfer(result, transfers, case["expected_result"], case["expected_transfer"])
        print_info(
            f"rebalance {case['name']} spot={case['spot']} perp={case['perp']} "
            f"result={result} transfers={transfers}"
        )


def test_rebalance_live() -> None:
    if os.getenv("SKIP_LIVE_REBALANCE") == "1":
        print_info("live rebalance skipped (SKIP_LIVE_REBALANCE=1).")
        return

    if not hasattr(dn.Configuration, "from_files"):
        print_info("SKIP: freqtrade not installed; live rebalance requires Configuration.from_files.")
        return

    print_info("live rebalance may transfer funds when imbalance > 0.5% and total >= 22 USDC.")
    log_path = _delta_log_path()
    start_size = log_path.stat().st_size if log_path.exists() else 0
    prev_address = dn.GLOBAL_ADDRESS
    prev_wallet = dn.PRIVATE_EVM_WALLET
    dn.GLOBAL_ADDRESS = None
    dn.PRIVATE_EVM_WALLET = None
    try:
        result = dn.REBALANCE_PERP_SPOT()
    finally:
        dn.GLOBAL_ADDRESS = prev_address
        dn.PRIVATE_EVM_WALLET = prev_wallet
    print_info(f"live rebalance result={result}")
    _print_latest_balances(log_path, start_size)
    _print_latest_transfer(log_path, start_size)


def main() -> None:
    tests = [
        ("test_rebalance_scenarios", test_rebalance_scenarios),
        ("test_rebalance_live", test_rebalance_live),
    ]
    for name, fn in tests:
        print(f"RUN: {name}")
        fn()
    print("OK")


if __name__ == "__main__":
    main()
