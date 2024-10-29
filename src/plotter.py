import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('dark_background')

#region load_data

class Plotter:
    def load_data(self, symbol, interval):
        filepath = self._get_kline_filepath(symbol, interval)
        return pd.read_csv(filepath, parse_dates=['date'])


    def load_processed_data(self, symbol, interval):
        filepath = self._get_processed_kline_filepath(symbol, interval)
        
        with np.load(filepath) as data:
            klines_processed = data['klines_processed']
            close = data['close']
            volume = data['volume']
            timestamps = data['timestamps']
        
        df = pd.DataFrame(klines_processed, columns=['open', 'high', 'low', 'close', 'turnover'])
        df['date'] = pd.to_datetime(timestamps, unit='ms')
        df['close_raw'] = close
        df['volume'] = volume
        
        return df


    def _get_kline_filepath(self, symbol, interval):
        base_path = os.path.join("../data", "klines", interval)
        return os.path.join(base_path, f"{symbol}.csv")
    

    def _get_processed_kline_filepath(self, symbol, interval):
        base_path = os.path.join("../data", "env_data", interval)
        return os.path.join(base_path, f"{symbol}.npz")

#endregion
#region plot_data

    def plot_price_history_and_stats(self, df, symbol):
        def format_number(num):
            if num >= 1e9:
                return f"{num / 1e9:.2f}B"
            elif num >= 1e6:
                return f"{num / 1e6:.2f}M"
            elif num >= 1e3:
                return f"{num / 1e3:.2f}K"
            else:
                return f"{num:.2f}"

        print(f"Statistics for {symbol}:")
        print(f"Total number of data points: {len(df)}")
        print(f"Date range: from {df['date'].min()} to {df['date'].max()}")
        print(f"Lowest close price: {df['close'].min():.8f} on {df['date'][df['close'].idxmin()]}")
        print(f"Highest close price: {df['close'].max():.8f} on {df['date'][df['close'].idxmax()]}")
        print(f"Total turnover: {format_number(df['turnover'].sum())}")
        print(f"Average daily turnover: {format_number(df['turnover'].mean())}")
        print(f"Highest daily turnover: {format_number(df['turnover'].max())} on {df['date'][df['turnover'].idxmax()]}")
        print(f"Total trading volume: {format_number(df['volume'].sum())}")
        print(f"Average daily volume: {format_number(df['volume'].mean())}")
        print(f"Price change: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.2f}%")

        # Plot
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20, 18), sharex='col')

        # Close Price - Linear Scale
        ax1.plot(df['date'], df['close'], color='#00FFFF')  # Cyan
        ax1.set_title(f'{symbol} Close Price (Linear Scale)')
        ax1.set_ylabel('Close Price')
        ax1.grid(True, color='#555555')

        # Turnover - Linear Scale
        ax2.plot(df['date'], df['turnover'], color='#FF1493')  # Deep Pink
        ax2.set_title(f'{symbol} Turnover (Linear Scale)')
        ax2.set_ylabel('Turnover')
        ax2.grid(True, color='#555555')

        # Close Price - Log Scale
        ax3.semilogy(df['date'], df['close'], color='#00FFFF')  # Cyan
        ax3.set_title(f'{symbol} Close Price (Log Scale)')
        ax3.set_ylabel('Close Price (log scale)')
        ax3.grid(True, color='#555555')

        # Turnover - Log Scale
        ax4.semilogy(df['date'], df['turnover'], color='#FF1493')  # Deep Pink
        ax4.set_title(f'{symbol} Turnover (Log Scale)')
        ax4.set_ylabel('Turnover (log scale)')
        ax4.grid(True, color='#555555')

        # Close Price - Log10 Values
        ax5.plot(df['date'], np.log10(df['close']), color='#00FFFF')  # Cyan
        ax5.set_title(f'{symbol} Close Price (Log10 Values)')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Log10(Close Price)')
        ax5.grid(True, color='#555555')

        # Turnover - Log10 Values
        ax6.plot(df['date'], np.log10(df['turnover']), color='#FF1493')  # Deep Pink
        ax6.set_title(f'{symbol} Turnover (Log10 Values)')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Log10(Turnover)')
        ax6.grid(True, color='#555555')

        # Rotate and align the tick labels
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.show()

#endregion
#region plot_raw_klines

    def plot_raw_klines(self, interval):
        klines_path = os.path.join("../data/klines", interval)
        plots_path = os.path.join("../data/plots/raw_klines", interval)
        os.makedirs(plots_path, exist_ok=True)

        all_symbols = [f[:-4] for f in os.listdir(klines_path) if f.endswith('.csv')]

        # Load all data at once
        all_data = {}
        for symbol in tqdm(all_symbols, desc="Loading data"):
            try:
                filepath = os.path.join(klines_path, f"{symbol}.csv")
                df = pd.read_csv(filepath, parse_dates=['date'])
                all_data[symbol] = df
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")

        # Plot data
        for symbol in tqdm(all_data.keys(), desc="Plotting data"):
            df = all_data[symbol]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle(f'{symbol} Raw Klines ({interval})', fontsize=16)

            # Close Price
            ax1.plot(df['date'], df['close'], color='#00FFFF')  # Cyan
            ax1.set_title('Close Price')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Close Price')
            ax1.grid(True, color='#555555')

            # Turnover
            ax2.plot(df['date'], df['turnover'], color='#FF1493')  # Deep Pink
            ax2.set_title('Turnover')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Turnover')
            ax2.grid(True, color='#555555')

            # Rotate and align the tick labels
            fig.autofmt_xdate()

            plt.tight_layout()

            # Save the figure
            output_file = os.path.join(plots_path, f'{symbol}_raw_klines.png')
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close(fig)

        print(f"Raw klines plots for {interval} saved in '{plots_path}'")

#endregion
#region check_raw_klines

    def check_raw_klines(self, interval):
        klines_path = os.path.join("../data/klines", interval)
        issues_path = "../data/plots/raw_klines"
        os.makedirs(issues_path, exist_ok=True)

        all_symbols = [f[:-4] for f in os.listdir(klines_path) if f.endswith('.csv')]

        issues = {}
        for symbol in tqdm(all_symbols, desc="Checking klines"):
            try:
                filepath = os.path.join(klines_path, f"{symbol}.csv")
                df = pd.read_csv(filepath, parse_dates=['date'])
                price_changes = df['close'].pct_change()

                # Find indices where price changes are <= -0.5 or >= 0.5
                issue_indices = np.where((price_changes <= -0.5) | (price_changes >= 0.5))[0]
                
                if len(issue_indices) > 0:
                    issues[symbol] = []
                    for idx in issue_indices:
                        issues[symbol].append({
                            'timestamp': df['date'].iloc[idx].isoformat(),
                            'price_change': f"{price_changes.iloc[idx]:.2%}",
                            'prev_close': df['close'].iloc[idx-1],
                            'curr_close': df['close'].iloc[idx]
                        })

            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")

        output_file = os.path.join(issues_path, f'kline_issues_{interval.replace(" ", "_")}.json')
        with open(output_file, 'w') as f:
            json.dump(issues, f, indent=2)

        print(f"Kline issues for {interval} saved in '{output_file}'")
    
#endregion
#region plot_efficient

    def plot_efficient_symbol_comparison(self, interval, num_symbols=10, use_processed=False, use_all_symbols=False, exclude_symbols=None, show_all_plots=False):
        env_data_path = os.path.join("../data/env_data", interval)
        os.makedirs(env_data_path, exist_ok=True)
        all_symbols = [f[:-4] for f in os.listdir(env_data_path) if f.endswith('.npz')]
        
        if exclude_symbols:
            all_symbols = [symbol for symbol in all_symbols if symbol not in exclude_symbols]
        
        if use_all_symbols:
            selected_symbols = all_symbols
            num_symbols = len(all_symbols)
        else:
            selected_symbols = random.sample(all_symbols, min(num_symbols, len(all_symbols)))

        # Load all data at once
        all_data = {}
        for symbol in tqdm(selected_symbols, desc="Loading data"):
            try:
                if use_processed:
                    df = self.load_processed_data(symbol, interval)
                else:
                    df = self.load_data(symbol, interval)
                    # Apply log transformation for raw data
                    for col in ['open', 'high', 'low', 'close', 'turnover']:
                        df[col] = np.log10(df[col].replace(0, 1e-10))
                
                if df is not None:
                    all_data[symbol] = df
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")

        # Create figure and axes
        if show_all_plots:
            fig, axes = plt.subplots(5, 2, figsize=(24, 40))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(24, 20))
        
        data_type = "Processed" if use_processed else "Raw"
        fig.suptitle(f'Comparison of {len(all_data)} {"All" if use_all_symbols else "Random"} Symbols ({interval}) - {data_type} Data', fontsize=16)

        # Generate a color palette
        colors = plt.cm.rainbow(np.linspace(0, 1, len(all_data)))

        # Plot data
        with tqdm(total=len(all_data), desc="Plotting data") as pbar:
            for symbol, color in zip(all_data.keys(), colors):
                df = all_data[symbol]
                
                if show_all_plots:
                    plot_columns = ['open', 'high', 'low', 'close', 'turnover']
                else:
                    plot_columns = ['close', 'turnover']
                
                for i, col in enumerate(plot_columns):
                    axes[i, 0].plot(df['date'], df[col], color=color, alpha=0.7)
                    axes[i, 1].hist(df[col], bins=50, alpha=0.5, color=color)
                
                pbar.update(1)

        # Set titles and labels
        titles = ['Open Price', 'High Price', 'Low Price', 'Close Price', 'Turnover'] if show_all_plots else ['Close Price', 'Turnover']
        for i, title in enumerate(titles):
            axes[i, 0].set_title(f'{title} Over Time')
            axes[i, 0].set_xlabel('Date')
            axes[i, 0].set_ylabel(f'{title}' if use_processed else f'Log10 {title}')
            axes[i, 0].tick_params(axis='x', rotation=45)

            axes[i, 1].set_title(f'{title} Distribution')
            axes[i, 1].set_xlabel(f'{title}' if use_processed else f'Log10 {title}')
            axes[i, 1].set_ylabel('Frequency')

        # Adjust the layout automatically
        plt.tight_layout()

        # Create the plots folder if it doesn't exist
        plots_folder = f'../data/plots/'
        os.makedirs(plots_folder, exist_ok=True)

        # Save the figure with a higher DPI for better quality
        output_file = os.path.join(plots_folder, f'{"all" if use_all_symbols else "random"}_symbols_{data_type.lower()}_{interval.replace(" ", "_")}.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Symbols comparison plot saved as '{output_file}'")

#endregion
#region recent_klines

    def plot_recent_data(self, processed_klines):
        plt.style.use('dark_background')

        # Group the data by interval
        intervals = set(interval for symbol, interval in processed_klines.keys())

        for interval in intervals:
            # Filter data for the current interval
            interval_data = {symbol: data for (symbol, int_val), data in processed_klines.items() if int_val == interval}

            fig, axes = plt.subplots(2, 2, figsize=(24, 20))
            fig.suptitle(f'Comparison of Recent Data ({interval})', fontsize=16)

            # Generate a color palette
            colors = plt.cm.rainbow(np.linspace(0, 1, len(interval_data)))

            for (symbol, color) in zip(interval_data.keys(), colors):
                data = interval_data[symbol]
                df = data['data']
                timestamps = data['timestamps']

                # Convert timestamps to datetime
                dates = pd.to_datetime(timestamps, unit='ms')

                # Close price
                axes[0, 0].plot(dates, df[:, 3], color=color, alpha=0.7, label=symbol)
                axes[0, 1].hist(df[:, 3], bins=50, alpha=0.5, color=color, label=symbol)

                # Turnover
                axes[1, 0].plot(dates, df[:, 4], color=color, alpha=0.7, label=symbol)
                axes[1, 1].hist(df[:, 4], bins=50, alpha=0.5, color=color, label=symbol)

            # Set titles and labels
            titles = ['Close Price', 'Turnover']
            for i, title in enumerate(titles):
                axes[i, 0].set_title(f'{title} Over Time')
                axes[i, 0].set_xlabel('Date')
                axes[i, 0].set_ylabel(f'{title}')
                axes[i, 0].tick_params(axis='x', rotation=45)
                # axes[i, 0].legend()

                axes[i, 1].set_title(f'{title} Distribution')
                axes[i, 1].set_xlabel(f'{title}')
                axes[i, 1].set_ylabel('Frequency')
                # axes[i, 1].legend()

            # Adjust the layout automatically
            plt.tight_layout()

            # Create the plots/recent folder if it doesn't exist
            plots_folder = '../data/plots/recent'
            os.makedirs(plots_folder, exist_ok=True)

            # Save the figure with a higher DPI for better quality
            output_file = os.path.join(plots_folder, f'recent_data_comparison_{interval.replace(" ", "_")}.png')
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"Recent data comparison plot for {interval} saved as '{output_file}'")

#endregion
#region stablecoins

    def identify_potential_stablecoins(self, threshold=0.01, interval="1 day"):
        env_data_path = os.path.join("../data/env_data", interval)
        os.makedirs(env_data_path, exist_ok=True)
        all_symbols = [f[:-4] for f in os.listdir(env_data_path) if f.endswith('.npz')]
        potential_stablecoins = []

        for symbol in tqdm(all_symbols, desc="Checking for stablecoins"):
            try:
                df = self.load_processed_data(symbol, interval)
                if df is None:
                    continue
                
                # Calculate the range of close prices
                price_range = df['close'].max() - df['close'].min()
                
                # If the range is very small, consider it a potential stablecoin
                if price_range < threshold:
                    potential_stablecoins.append(symbol)
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")

        print(f"Potential stablecoins identified: {potential_stablecoins}")
        return potential_stablecoins
    