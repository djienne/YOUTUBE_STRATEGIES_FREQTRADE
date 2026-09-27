import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from _test_helpers import add_strategy_path, assert_true, load_strategy_module, print_info

HERE = Path(__file__).resolve().parent
add_strategy_path(__file__)
dn = load_strategy_module()

HISTORICAL_RATES = {
    "BTC": ("0.0000125", "0.0000200"),
    "ETH": ("0.0000150", "0.0000250"),
    "SOL": ("0.0000100", "0.0000300"),
}
COINS = list(HISTORICAL_RATES.keys())


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _cleanup(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _apr_from_rate(rate: str | float) -> float:
    return round(float(rate) * 24.0 * 365.0 * 100.0, 2)


def _pair_for(coin: str) -> str:
    return f"{coin}/USDC:USDC"


def _extract_pair_series(db: dict, pair: str):
    series = []
    for ts_str, pairs in db.items():
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        val = pairs.get(pair)
        if isinstance(val, (int, float)):
            series.append((ts, float(val)))
    return sorted(series, key=lambda item: item[0])


def _single_hour_key(db: dict) -> str:
    keys = list(db.keys())
    assert_true(len(keys) == 1, f"Expected 1 hour key, got {len(keys)}.")
    return keys[0]


@contextmanager
def _temp_db_path(prefix: str):
    path = HERE / f"{prefix}_{uuid.uuid4().hex}.json"
    _cleanup(path)
    try:
        yield path
    finally:
        _cleanup(path)


def test_update_funding_db_historical() -> None:
    with _temp_db_path("tmp_funding_db") as tmp:
        dt1 = datetime(2025, 1, 1, 10, 34, 56, tzinfo=timezone.utc)
        dt2 = datetime(2025, 1, 1, 11, 5, 1, tzinfo=timezone.utc)

        expected_by_pair = {}
        for coin, (rate1, rate2) in HISTORICAL_RATES.items():
            pair = _pair_for(coin)
            records = [
                {"coin": coin, "fundingRate": rate1, "time": int(dt1.timestamp() * 1000)},
                {"coin": coin, "fundingRate": rate2, "time": int(dt2.timestamp() * 1000)},
            ]
            dn.update_funding_db(str(tmp), records, pair)
            expected_by_pair[pair] = (_apr_from_rate(rate1), _apr_from_rate(rate2))

        db = _read_json(tmp)
        key1 = dn._iso_ms(dn._to_hour_start(dt1))
        key2 = dn._iso_ms(dn._to_hour_start(dt2))

        for pair, (expected1, expected2) in expected_by_pair.items():
            assert_true(db[key1][pair] == expected1, f"Historical funding value mismatch (key1) for {pair}.")
            assert_true(db[key2][pair] == expected2, f"Historical funding value mismatch (key2) for {pair}.")
            print_info(f"historical {pair} {key1} -> {expected1}")
            print_info(f"historical {pair} {key2} -> {expected2}")

        for coin in COINS:
            dn.update_funding_db(
                str(tmp),
                [{"coin": coin, "fundingRate": "0.999999", "time": int(dt1.timestamp() * 1000)}],
                _pair_for(coin),
            )
        db2 = _read_json(tmp)
        for pair, (expected1, _) in expected_by_pair.items():
            assert_true(
                db2[key1][pair] == expected1,
                f"Historical funding value should not be overwritten for {pair}.",
            )


def test_record_hourly_and_avg() -> None:
    with _temp_db_path("tmp_current_funding_db") as tmp:
        values_by_pair = {}
        for idx, coin in enumerate(COINS):
            pair = _pair_for(coin)
            value = round(12.345678 + idx, 6)
            values_by_pair[pair] = value
            dn.record_hourly_funding_by_pair(
                pair,
                value,
                tz=timezone.utc,
                file_path=tmp,
            )

        db = _read_json(tmp)
        hour_key = _single_hour_key(db)
        hour_entry = db[hour_key]
        for pair, expected in values_by_pair.items():
            assert_true(hour_entry[pair] == expected, f"Current funding value mismatch for {pair}.")
            print_info(f"current {pair} {hour_key} -> {hour_entry[pair]}")
        assert_true("harvested_datetime" in hour_entry, "Missing harvested_datetime entry.")

        for pair, expected in values_by_pair.items():
            avg = dn.avg_funding_last_hours(
                pair,
                nb_hours=2,
                file_path=tmp,
                now_utc=datetime.now(timezone.utc) + timedelta(hours=2),
            )
            assert_true(abs(avg - expected) < 1e-9, f"Average funding mismatch for {pair}.")
            print_info(f"average {pair} last 2h -> {avg}")


def test_get_funding_history_live() -> None:
    print_info("live funding history is read-only (no orders or transfers).")
    with _temp_db_path("tmp_live_history_db") as tmp:
        try:
            for coin in COINS:
                ok = dn.get_funding_history(tmp, coin, days_interval=1)
                assert_true(ok is True, f"get_funding_history returned False for {coin}.")
        except ModuleNotFoundError as exc:
            print(f"SKIP: hyperliquid not installed ({exc})")
            return

        db = _read_json(tmp)
        assert_true(len(db) > 0, "No data returned from get_funding_history.")
        for coin in COINS:
            pair = _pair_for(coin)
            series = _extract_pair_series(db, pair)
            assert_true(series, f"{pair} data missing in funding DB.")
            values = [val for _, val in series]
            print_info(
                f"live {pair} samples={len(series)} "
                f"min={min(values):.3f} max={max(values):.3f}"
            )
            for ts, val in series[-3:]:
                print_info(f"live {pair} {ts.isoformat()} -> {val}")


def main() -> None:
    tests = [
        ("test_update_funding_db_historical", test_update_funding_db_historical),
        ("test_record_hourly_and_avg", test_record_hourly_and_avg),
        ("test_get_funding_history_live", test_get_funding_history_live),
    ]
    for name, fn in tests:
        print(f"RUN: {name}")
        fn()
    print("OK")


if __name__ == "__main__":
    main()
