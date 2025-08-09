from pathlib import Path

def log_equity(
    total_equity: float,
    min_interval_minutes: int = 30,
):
    """
    Logs `total_equity` for `address` to CSV, with abs/rel change vs first recorded value.
    Appends only if last row is older than `min_interval_minutes`.

    CSV columns:
      timestamp, total_equity_usdc, abs_change_from_first, rel_change_from_first
    """

    # --- imports inside for easy copy/paste ---
    from datetime import datetime, timezone
    from decimal import Decimal, InvalidOperation
    import csv, os
    from typing import Optional

    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)

    def _read_last_timestamp(csv_path: str) -> Optional[datetime]:
        if not os.path.exists(csv_path):
            return None
        with open(csv_path, "rb") as f:
            try:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode("utf-8").strip()
        if not last_line or last_line.startswith("timestamp"):
            return None
        try:
            return datetime.fromisoformat(last_line.split(",")[0])
        except ValueError:
            return None

    def _read_first_total(csv_path: str) -> Optional[Decimal]:
        if not os.path.exists(csv_path):
            return None
        with open(csv_path, newline="") as f:
            r = csv.reader(f)
            try:
                _ = next(f)
            except StopIteration:
                return None
            for row in r:
                if not row or row[0].strip().lower() == "timestamp":
                    continue
                if len(row) >= 2 and row[1].strip():
                    try:
                        return Decimal(row[1].strip())
                    except InvalidOperation:
                        return None
                break
        return None

    def _write_sample(csv_path: str, when: datetime, equity: Decimal, first_total: Optional[Decimal]) -> None:
        file_exists = os.path.exists(csv_path)
        abs_change, rel_change = "", ""
        if first_total is not None:
            try:
                abs_change_val = equity - first_total
                abs_change = str(abs_change_val)
                if first_total != 0:
                    rel_change = str((equity / first_total) - Decimal("1"))
            except InvalidOperation:
                pass
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow([
                    "timestamp",
                    "total_equity_usdc",
                    "abs_change_from_first",
                    "rel_change_from_first"
                ])
            w.writerow([when.isoformat(), str(equity), abs_change, rel_change])

    # Build file path
    filename = f"equity_track.csv"
    here = Path(__file__).resolve().parent
    csv_path = os.path.join(here, filename)

    # Decide whether to append
    when = _now_utc()
    last_ts = _read_last_timestamp(csv_path)
    if last_ts is None or (when - last_ts).total_seconds() >= min_interval_minutes * 60:
        equity_decimal = Decimal(str(total_equity))
        first_total = _read_first_total(csv_path)
        _write_sample(csv_path, when, equity_decimal, first_total)

def get_account_total_equity(address):
    from hyperliquid.exchange import Exchange
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    import eth_account
    from eth_account.signers.local import LocalAccount
    import math

    def get_spot_token_price_usdc(token_symbol: str) -> float:
        """
        Get the price of a spot token in USDC using Hyperliquid SDK
        
        Args:
            token_symbol: The token symbol (e.g., 'UETH', 'UFART', 'BTC', 'ETH', etc.)
                        Can also handle numerical IDs like '@1', '@100', etc.
        
        Returns:
            Price in USDC as float
        """
        # Initialize the Info client
        info = Info(constants.MAINNET_API_URL, skip_ws=True)

        if token_symbol=='UFART':
            token_symbol='FARTCOIN'
        elif token_symbol=='UBTC':
            token_symbol='BTC'
        elif token_symbol=='UETH':
            token_symbol='ETH'
        elif token_symbol=='USOL':
            token_symbol='SOL'
        elif token_symbol=='PURR':
            token_symbol='PURR'
        elif token_symbol=='HYPE':
            token_symbol='HYPE'
        elif token_symbol=='UPUMP':
            token_symbol='PUMP'
        
        # Get all mids (market prices) for all actively traded coins
        all_mids = info.all_mids()
        
        # Find the price for the specific token
        if token_symbol in all_mids:
            price = float(all_mids[token_symbol])
            return price
        else:
            return 0.0

    def sum_entry_ntl_and_usdc_total(data):
        from decimal import Decimal, InvalidOperation

        total = 0.0

        balances = data.get("balances", [])
        for b in balances:
            try:
                # Sum all entryNtl values
                # print(b.get("coin", "0"))
                # print(b.get("total", "0"))
                if b.get("coin", "0")!='USDC':
                    total += float(b.get("total", "0"))*float(get_spot_token_price_usdc(b.get("coin", "0")))
                else:
                    total += float(b.get("total", "0"))
            except (InvalidOperation, TypeError):
                pass

        return float(total)

    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    perp_user_state = info.user_state(address)
    spot_user_state = info.spot_user_state(address)

    perp_account = float(perp_user_state['marginSummary']['accountValue'])
    spot_account = sum_entry_ntl_and_usdc_total(spot_user_state)
    return perp_account+spot_account

val = get_account_total_equity('0xa906355beaf1d69a5fe73ce55899c49c6e67916c')
print(val)
log_equity(val, min_interval_minutes=1)
