"""Download every published version ("vintage") of US M2, FRED series WM2NS, from ALFRED.

INTERMARKET_M2 backtests replay these vintages, so that each candle only uses the M2 data
already published at its close (see the strategy docstring). Live trading does not need
this file: the bot reads the current series from FRED.

Usage, from the folder that contains user_data (inside the freqtrade container):
    docker compose run --rm --entrypoint python freqtrade user_data/download_m2_vintages.py
    ... user_data/download_m2_vintages.py 2020-01-01   # only the vintages in force since then
Output: user_data/m2/WM2NS_vintages.csv.gz with columns vintage_date (release day),
observation_date (week ending Monday) and value (billions of dollars, not seasonally adjusted).
One request per release (about 260 since 2017), paced at 2 per second.
"""
import gzip
from io import StringIO
from pathlib import Path
import re
import sys
import time
import urllib.request

import pandas as pd

LIST_URL = "https://alfred.stlouisfed.org/series/downloaddata?seid=WM2NS"
VINTAGE_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=WM2NS&vintage_date={}"
OUT = Path(__file__).resolve().parent / "m2" / "WM2NS_vintages.csv.gz"


def get(url):
    # Python's default User-Agent: FRED/ALFRED left requests with a custom one unanswered (Sep 2026).
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read().decode("utf-8")
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def main():
    since = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2017-08-17")  # Binance BTC/USDT listing
    listed = sorted(set(pd.to_datetime(re.findall(r'<option[^>]*value="(\d{4}-\d{2}-\d{2})"', get(LIST_URL)))))
    first = max(v for v in listed if v <= since)  # the version in force on `since`
    frames = []
    for vintage in [v for v in listed if v >= first]:
        frame = pd.read_csv(StringIO(get(VINTAGE_URL.format(f"{vintage:%Y-%m-%d}"))), dtype=str)
        if list(frame.columns) != ["observation_date", f"WM2NS_{vintage:%Y%m%d}"]:
            raise ValueError(f"unexpected ALFRED columns for {vintage:%Y-%m-%d}: {list(frame.columns)}")
        dates = pd.to_datetime(frame["observation_date"])
        if not (dates.dt.dayofweek == 0).all() or dates.iloc[-1] >= vintage or pd.to_numeric(frame.iloc[:, 1]).le(0).any():
            raise ValueError(f"inconsistent ALFRED data for {vintage:%Y-%m-%d}")
        frame.columns = ["observation_date", "value"]
        frame.insert(0, "vintage_date", f"{vintage:%Y-%m-%d}")
        frames.append(frame)
        print(f"{vintage:%Y-%m-%d}: {len(frame)} weeks, last {frame['observation_date'].iloc[-1]}", flush=True)
        time.sleep(0.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as handle, gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as gz:
        gz.write(pd.concat(frames).to_csv(index=False).encode("utf-8"))
    print(f"{len(frames)} vintages saved to {OUT}")


if __name__ == "__main__":
    main()
