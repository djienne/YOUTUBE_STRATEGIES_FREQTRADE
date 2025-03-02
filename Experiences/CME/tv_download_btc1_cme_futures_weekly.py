#!/usr/bin/env python3
"""
Download BTC1! weekly historical data using tvDatafeed.

This script logs in to TradingView, downloads the weekly aggregated
historical data for BTC1! (assumed here to be a continuous futures contract on CME),
and saves it as a CSV file.

To run this script on a weekly schedule, consider using your OS's scheduler (e.g., cron on Linux)
or add scheduling logic (see the commented-out section at the bottom).

to install tvDatafeed:
pip install --upgrade --no-cache-dir git+https://github.com/rongardF/tvdatafeed.git

"""

from tvDatafeed import TvDatafeed, Interval
import pandas as pd

def download_btc1_weekly(n_bars=1000, output_file='BTC1_weekly_data.csv'):
    try:
        # Initialize the tvDatafeed instance with your TradingView credentials
        tv = TvDatafeed()
        print("Downloading BTC1! weekly data...")

        # Get historical data:
        # - symbol: 'BTC1!'
        # - exchange: 'CME' (adjust if your symbol is on a different exchange)
        # - interval: Interval.in_weekly downloads weekly bars
        # - n_bars: number of bars to download (e.g., 52 for about 1 year)
        # - fut_contract=1: used for continuous futures contracts (if applicable)
        data = tv.get_hist(symbol='BTC1!',
                           exchange='CME',
                           interval=Interval.in_weekly,
                           n_bars=n_bars)
        
        # Check if data was returned successfully
        if data is None or data.empty:
            print("No data was returned. Please check the symbol/exchange credentials.")
            return

        # Save the data to CSV
        print(data)
        data.to_csv(output_file)
        print(f"Data saved successfully to {output_file}")

    except Exception as e:
        print("An error occurred while downloading data:", str(e))

if __name__ == '__main__':
    # Replace these with your actual TradingView username and password
    
    download_btc1_weekly()

    # If you want to schedule this script to run weekly (for example, every Monday at midnight),
    # you could use the "schedule" library. Uncomment the code below and install schedule (pip install schedule).
    #
    # import schedule
    # import time
    #
    # def job():
    #     download_btc1_weekly(username, password)
    #
    # # Schedule the job every Monday at 00:00
    # schedule.every().monday.at("00:00").do(job)
    #
    # print("Scheduler started. Waiting for scheduled run...")
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)

