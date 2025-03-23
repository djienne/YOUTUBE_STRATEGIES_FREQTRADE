#!/usr/bin/env python3
"""
Download BTCUSD daily historical data using tvDatafeed.

This script logs in to TradingView, downloads the daily aggregated
historical data for BTCUSD (from the INDEX exchange),
and saves it as a CSV file after removing the 'symbol' column 
and renaming the index (datetime) to 'date'.

To install tvDatafeed:
pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git
"""

from tvDatafeed import TvDatafeed, Interval
import pandas as pd

def download_btc_daily(output_file='BTC_daily_data.csv'):
    try:
        # Initialize the tvDatafeed instance
        tv = TvDatafeed()
        print("Downloading BTCUSD daily data...")

        # Get historical data
        data = tv.get_hist(symbol='BTCUSD',
                           exchange='CRYPTO',
                           interval=Interval.in_daily,
                           n_bars=10000
        )
        
        # Check if data was returned successfully
        if data is None or data.empty:
            print("No data was returned. Please check the symbol/exchange credentials.")
            return

        # Reset index to move 'datetime' from index to column
        data = data.reset_index()

        # Rename the index column from 'datetime' to 'date'
        data = data.rename(columns={'datetime': 'date'})

        # Remove 'symbol' column if it exists
        data = data.drop(columns=['symbol'], errors='ignore')

        # Save the processed data to CSV
        print(data)
        data.to_csv(output_file, index=False)
        print(f"Data saved successfully to {output_file}")

    except Exception as e:
        print("An error occurred while downloading data:", str(e))

if __name__ == '__main__':
    download_btc_daily()
