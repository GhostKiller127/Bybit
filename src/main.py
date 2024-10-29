import yaml
import asyncio
import aiohttp
from tqdm import tqdm
from downloader import Downloader


async def main():
    with open('configs.yaml', 'r') as file:
        configs = yaml.safe_load(file)
    # last_launch_time = configs['last_launch_time']
    last_launch_time = 1696118400000  # 2023-10-01 00:00:00
    # exclude_symbols = configs['exclude_symbols']
    exclude_symbols = ['USDCUSDT']
    # intervals = configs['intervals']
    intervals = ['5 minutes']
    # start_from_symbol = None
    start_from_symbol = 'LUNA2USDT'

    downloader = Downloader()
    await downloader.download_trading_symbols()
    symbols = downloader.load_trading_symbols(last_launch_time, exclude_symbols)
    
    removed_symbols = downloader.remove_delisted_symbols(symbols, intervals)
    if removed_symbols:
        print("Removed delisted symbols:")
        for symbol, interval in removed_symbols:
            print(f"  - {symbol} for interval {interval}")
    else:
        print("No delisted symbols found.")

    start_processing = start_from_symbol is None
    try:
        async with aiohttp.ClientSession() as session:
            for interval in tqdm(intervals, desc="Intervals"):
                for symbol in tqdm(symbols['Symbol'], desc=f"Symbols ({interval})", leave=False):
                    if not start_processing and symbol == start_from_symbol:
                        start_processing = True
                    if start_processing:
                        launch_time = int(symbols[symbols['Symbol'] == symbol]['Launch Time'].iloc[0])
                        await downloader.update_kline_history(session, symbol, interval, launch_time)
    except Exception as e:
        print(f"Error processing symbol {symbol}: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())