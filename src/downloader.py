import os
import json
import glob
import aiohttp
import asyncio
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm
import configs_trader as configs
from aiolimiter import AsyncLimiter
from datetime import datetime, timezone



class Downloader:
    def __init__(self):
        self.base_url = "https://api.bybit.com/v5/market"
        self.category = "linear"
        self.all_intervals = {
            "1 minute": "1",
            "3 minutes": "3",
            "5 minutes": "5",
            "15 minutes": "15",
            "30 minutes": "30",
            "1 hour": "60",
            "2 hours": "120",
            "4 hours": "240",
            "6 hours": "360",
            "12 hours": "720",
            "1 day": "D",
            "1 week": "W",
            "1 month": "M",
        }
        self.rate_limiter = AsyncLimiter(configs.MAX_REQUESTS, configs.REQUEST_INTERVAL)
        self.semaphore = asyncio.Semaphore(configs.SEMAPHORE_SIZE)

#region trading symbols

    async def download_trading_symbols(self):
        info_url = f"{self.base_url}/instruments-info"
        ticker_url = f"{self.base_url}/tickers"
        params = {"category": self.category}
        async with aiohttp.ClientSession() as session:
            async with session.get(info_url, params=params) as response:
                data = await response.json()
                symbols = data.get('result', {}).get('list', [])
            async with session.get(ticker_url, params=params) as response:
                data = await response.json()
                tickers = data.get('result', {}).get('list', [])
        
        symbols_dict = {symbol['symbol']: symbol for symbol in symbols if symbol['quoteCoin'] == 'USDT'}
        tickers_dict = {ticker['symbol']: ticker for ticker in tickers}

        symbols_data = []
        for symbol, info in symbols_dict.items():
            if symbol in tickers_dict:
                launch_time = pd.to_datetime(int(info['launchTime']), unit='ms', utc=True)
                history_length = (datetime.now(timezone.utc) - launch_time).days
                tick_size = float(info['priceFilter']['tickSize'])
                last_price = float(tickers_dict[symbol]['lastPrice'])
                tick_size_per_mille = round((tick_size / last_price) * 1000, 5)
                symbol_data = {
                    'Launch Time': int(info['launchTime']),
                    'Launch Time (Date)': launch_time.strftime("%Y-%m-%d"),
                    'History Length (Days)': history_length,
                    'History Length (Weeks)': history_length // 7,
                    'History Length (Months)': history_length // 30,
                    'History Length (Years)': round(history_length / 365, 1),
                    'Tick Size': tick_size,
                    'Tick Size Per Mille': tick_size_per_mille,
                    'Quantity Step': info['lotSizeFilter']['qtyStep'],
                    'Last Price': last_price,
                    'Symbol': symbol,
                }
                symbols_data.append(symbol_data)

        data_dir = "../data"
        for old_file in glob.glob(os.path.join(data_dir, "trading_symbols*.csv")):
            os.remove(old_file)

        active_df = pd.DataFrame(symbols_data)
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._save_sorted_csv(active_df, f"trading_symbols_{current_date}_by_launch.csv", by_column='Launch Time')
        self._save_sorted_csv(active_df, f"trading_symbols_{current_date}_by_size.csv", by_column='Tick Size Per Mille')
    

    def load_trading_symbols(self, last_launch_time=np.inf, exclude_symbols=[]):
        data_dir = "../data"
        files = [f for f in os.listdir(data_dir) if f.startswith("trading_symbols") and f.endswith("by_launch.csv")]
        latest_file = max(files)
        file_path = os.path.join(data_dir, latest_file)
        df = pd.read_csv(file_path)
        filtered_df = df[df['Launch Time'] <= last_launch_time]
        filtered_df = filtered_df[~filtered_df['Symbol'].isin(exclude_symbols)]
        return filtered_df

#endregion
#region klines

    async def fetch_kline_data_async(self, session, symbol, interval, start_time=0, end_time=1e20, limit=1000):
        async with self.semaphore:
            async with self.rate_limiter:
                url = f"{self.base_url}/kline"
                params = {"category": self.category,
                        "symbol": symbol,
                        "interval": self.all_intervals[interval],
                        "start": start_time,
                        "end": end_time,
                        "limit": limit}
                async with session.get(url, params=params) as response:
                    data = await response.json()
                return data['result']['list'] if 'result' in data and 'list' in data['result'] else []


    async def update_kline_history(self, session, symbol, interval, launch_time):
        current_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        filepath = self._get_kline_filepath(symbol, interval)
        interval_ms = self.interval_to_milliseconds(interval)

        # Calculate all start and end times
        time_ranges = []
        start = launch_time
        while start < current_time:
            end = start + 1000 * interval_ms
            time_ranges.append((start, end))
            start = end + 1

        # Fetch all klines
        tasks = [self.fetch_kline_data_async(session, symbol, interval, start, end) 
                 for start, end in time_ranges]
        results = await tqdm.gather(*tasks, desc=f"Fetching {symbol} {interval}", leave=False)

        # Process results
        all_klines = [kline for result in results for kline in result]
        df = self.klines_to_dataframe(all_klines)

        # Remove the latest (potentially incomplete) kline
        df = df.iloc[:-1]

        # Remove zero-value klines from the beginning (up to 10 klines)
        if len(df) > 10:
            first_10 = df.head(10)
            non_zero_start = first_10.index[((first_10['open'] != 0) & (first_10['high'] != 0) & 
                                             (first_10['low'] != 0) & (first_10['close'] != 0) & 
                                             (first_10['volume'] != 0)).argmax()]
            df = df.loc[non_zero_start:]

        # Check for holes and duplicates
        df = df.drop_duplicates(subset='timestamp')
        
        if interval == "1 month":
            # For monthly interval, generate correct timestamps
            start_date = df['date'].min().replace(day=1)
            end_date = df['date'].max().replace(day=1)
            expected_timestamps = pd.date_range(start=start_date, end=end_date, freq='MS', tz='UTC')
        else:
            expected_timestamps = pd.date_range(start=df['date'].min(), 
                                                end=df['date'].max(), 
                                                freq=pd.Timedelta(milliseconds=interval_ms),
                                                tz='UTC')
        
        holes = expected_timestamps[~expected_timestamps.isin(df['date'])]

        if not holes.empty:
            holes_dict = {interval: holes.astype(str).tolist()}  # Convert Timestamp to string
            holes_filepath = os.path.join("../data/klines", f"kline_holes_{symbol}_{interval}.json")
            with open(holes_filepath, 'w') as f:
                json.dump(holes_dict, f)

        df.to_csv(filepath, index=False)

        return df
    
#endregion
#region recent klines

    async def fetch_recent_klines(self, symbols, intervals, sequence_length, intervals_back=0):
        async with aiohttp.ClientSession() as session:
            current_time = int(datetime.now(timezone.utc).timestamp() * 1000)
            time_shift = intervals_back * self.interval_to_milliseconds(intervals[-1])
            end_time = current_time - time_shift

            tasks = []
            for symbol in symbols:
                for interval in intervals:
                    tasks.append(self.fetch_kline_data_async(session, symbol, interval, end_time=end_time, limit=sequence_length + 20))

            results = await tqdm.gather(*tasks, desc="Fetching recent klines")

        processed_results = {}
        incomplete_klines = {}
        last_klines = {}

        for i, klines in enumerate(results):
            symbol = symbols[i // len(intervals)]
            interval = intervals[i % len(intervals)]
            
            df = self.klines_to_dataframe(klines)

            # Remove zero-value klines from the beginning (up to 10 klines)
            if len(df) > 10:
                first_10 = df.head(10)
                non_zero_start = first_10.index[((first_10['open'] != 0) & (first_10['high'] != 0) & 
                                                (first_10['low'] != 0) & (first_10['close'] != 0) & 
                                                (first_10['volume'] != 0)).argmax()]
                df = df.loc[non_zero_start:]

            # Check if the latest kline is complete
            latest_kline_time = df['timestamp'].iloc[-1]

            if interval == "1 month":
                # For monthly interval, check if we're in a new month
                latest_kline_date = pd.to_datetime(latest_kline_time, unit='ms', utc=True)
                end_date = pd.Timestamp(end_time, unit='ms', tz='UTC')
                is_complete = (end_date.year > latest_kline_date.year or 
                               (end_date.year == latest_kline_date.year and 
                                end_date.month > latest_kline_date.month))
            else:
                interval_ms = self.interval_to_milliseconds(interval)
                time_difference = end_time - latest_kline_time
                is_complete = time_difference >= interval_ms
            
            is_complete = False
            if not is_complete:
                df = df.iloc[:-1]
            else:
                incomplete_klines[(symbol, interval)] = end_time - latest_kline_time

            df = df.tail(sequence_length)
            processed_results[(symbol, interval)] = df

            if interval == intervals[-1]:
                last_klines[symbol] = df.iloc[-1].to_dict()

        klines_complete = len(incomplete_klines) == 0

        return processed_results, klines_complete, incomplete_klines, last_klines

#endregion
#region helper

    def remove_delisted_symbols(self, symbols, intervals):
        klines_dir = os.path.join("..", "data", "klines")
        removed_symbols = []
        for interval in intervals:
            interval_dir = os.path.join(klines_dir, interval)
            if os.path.exists(interval_dir):
                existing_files = set(f.split('.')[0] for f in os.listdir(interval_dir) if f.endswith('.csv'))
                delisted_symbols = existing_files - set(symbols['Symbol'])
                for delisted_symbol in delisted_symbols:
                    file_path = os.path.join(interval_dir, f"{delisted_symbol}.csv")
                    os.remove(file_path)
                    removed_symbols.append((delisted_symbol, interval))
        return removed_symbols

    def _get_kline_filepath(self, symbol, interval):
        base_path = os.path.join("../data", "klines", interval)
        os.makedirs(base_path, exist_ok=True)
        return os.path.join(base_path, f"{symbol}.csv")


    def klines_to_dataframe(self, klines):
        df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df["timestamp"] = df["timestamp"].astype(int)
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.sort_values("timestamp").drop_duplicates(subset='timestamp')
        df = df.reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            df[col] = df[col].astype(float)
        columns_order = ['date'] + [col for col in df.columns if col != 'date']
        return df[columns_order]
    
    
    def interval_to_milliseconds(self, interval):
        units = {
            "minute": 60,
            "hour": 60 * 60,
            "day": 24 * 60 * 60,
            "week": 7 * 24 * 60 * 60,
            "month": 28 * 24 * 60 * 60,
        }
        unit = interval.split()[1].rstrip('s')
        value = int(interval.split()[0])
        return value * units[unit] * 1000


    def _save_sorted_csv(self, df, filename, by_column='Symbol', ascending=True):
        df = df.sort_values(by_column, ascending=ascending)
        columns = [by_column] + [col for col in df.columns if col != by_column]
        df = df[columns]

        output_path = os.path.join("../data", filename)
        df.to_csv(output_path, index=False)
        print(f"Saved {filename}")

#endregion