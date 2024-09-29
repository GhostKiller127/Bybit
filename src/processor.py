import os
import json
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm

#region create_env_data

class Processor:
    def create_env_data(self, intervals=None, reprocess=False, recalculate_constants=False, batch_size=None):
        for interval in intervals:
            print(f"Processing interval: {interval}")
            raw_data_path = os.path.join("../data/klines", interval)
            env_data_path = os.path.join("../data/env_data", interval)
            os.makedirs(env_data_path, exist_ok=True)
            symbols = [f for f in os.listdir(raw_data_path) if f.endswith('.csv')]

            # Remove processed files for delisted symbols
            existing_processed_files = set(f for f in os.listdir(env_data_path) if f.endswith('.npz'))
            current_symbols = set(f.split('.')[0] + '.npz' for f in symbols)
            delisted_symbols = existing_processed_files - current_symbols
            for delisted_file in delisted_symbols:
                os.remove(os.path.join(env_data_path, delisted_file))
                print(f"Removed processed file for delisted symbol: {delisted_file}")

            # Load or recalculate preprocessing constants
            constants_file = os.path.join("../data", "preprocessing_constants.json")
            if recalculate_constants:
                print("Recalculating preprocessing constants...")
                preprocessing_constants = self.calculate_preprocessing_constants(symbols)
                with open(constants_file, 'w') as f:
                    json.dump(preprocessing_constants, f)
            else:
                with open(constants_file, 'r') as f:
                    preprocessing_constants = json.load(f)

            global_constants = preprocessing_constants['global']

            # Determine batch size
            if batch_size is None or batch_size >= len(symbols):
                symbols_to_process = symbols
            else:
                symbols_to_process = symbols[:batch_size]

            # Step 1: Load raw and processed data to RAM
            raw_data = {}
            processed_data = {}
            for symbol in tqdm(symbols_to_process, desc="Loading data"):
                raw_file = os.path.join(raw_data_path, symbol)
                raw_data[symbol] = pd.read_csv(raw_file)
                
                processed_file = os.path.join(env_data_path, f"{symbol.split('.')[0]}.npz")
                if os.path.exists(processed_file) and not reprocess:
                    with np.load(processed_file) as data:
                        processed_data[symbol] = {
                            'data': data['klines_processed'],
                            'timestamps': data['timestamps']
                        }
                else:
                    processed_data[symbol] = None

            # Step 2: Check for new data
            symbols_to_update = []
            full_processing_count = 0
            new_data_count = 0
            up_to_date_count = 0

            for symbol in symbols_to_process:
                if reprocess or processed_data[symbol] is None:
                    symbols_to_update.append(symbol)
                    full_processing_count += 1
                elif raw_data[symbol]['timestamp'].iloc[-1] > processed_data[symbol]['timestamps'][-1]:
                    symbols_to_update.append(symbol)
                    new_data_count += 1
                else:
                    up_to_date_count += 1

            print(f"Processing summary: Full: {full_processing_count}, New data: {new_data_count}, Up-to-date: {up_to_date_count}")

            # Step 3: Process data
            for symbol in tqdm(symbols_to_update, desc="Processing"):
                scaling_factor = preprocessing_constants[symbol]["scaling_factor"]
                try:
                    processed = self.preprocess_kline_data(raw_data[symbol], scaling_factor, symbol, global_constants)
                    processed_data[symbol] = processed
                except Exception as e:
                    print(f"Error processing symbol {symbol}: {str(e)}")
                    continue

            # Step 4: Compare lengths and check for discrepancies
            for symbol in symbols_to_update:
                raw_length = len(raw_data[symbol])
                processed_length = len(processed_data[symbol]['data'])
                if raw_length != processed_length:
                    print(f"Warning: Length mismatch for {symbol}. Raw: {raw_length}, Processed: {processed_length}")

            # Step 5: Save processed data
            for symbol in tqdm(symbols_to_update, desc="Saving"):
                processed_file = os.path.join(env_data_path, f"{symbol.split('.')[0]}.npz")
                np.savez_compressed(processed_file, 
                                    klines_processed=processed_data[symbol]['data'],
                                    close=raw_data[symbol]['close'].values,
                                    volume=raw_data[symbol]['volume'].values,
                                    timestamps=raw_data[symbol]['timestamp'].values)

#endregion
#region recent_klines

    def process_recent_klines(self, processed_results):
        constants_file = os.path.join("../data", "preprocessing_constants.json")
        with open(constants_file, 'r') as f:
            preprocessing_constants = json.load(f)
        global_constants = preprocessing_constants['global']

        processed_klines = {}
        processing_issues = {}

        for (symbol, interval), df in processed_results.items():
            try:
                symbol_key = f"{symbol}.csv"
                if symbol_key not in preprocessing_constants:
                    raise KeyError(f"Symbol {symbol_key} not found in preprocessing constants")
                
                scaling_factor = preprocessing_constants[symbol_key]["scaling_factor"]
                processed = self.preprocess_kline_data(df, scaling_factor, symbol, global_constants)
                processed_klines[(symbol, interval)] = processed
            except Exception as e:
                error_info = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                }
                processing_issues[(symbol, interval)] = error_info
                print(f"Error processing {symbol} {interval}:")
                print(f"Error type: {error_info['error_type']}")
                print(f"Error message: {error_info['error_message']}")
                print(f"Traceback:\n{error_info['traceback']}")
                print("--------------------")

        return processed_klines, processing_issues

#endregion
#region preprocessing

    def preprocess_kline_data(self, df, scaling_factor, symbol, global_constants):
        # Remove the first row if any OHLC column has a zero value
        if (df[['open', 'high', 'low', 'close']].iloc[0] == 0).any():
            print(f"Warning: First kline for {symbol} contains zero values. Removing it.")
            df = df.iloc[1:].reset_index(drop=True)

        processed_data = np.zeros((len(df), 5), dtype=np.float32)
        timestamps = df['timestamp'].values.astype(np.int64)
        
        for i, col in enumerate(['open', 'high', 'low', 'close']):
            with np.errstate(divide='raise', invalid='raise'):
                try:
                    # Apply log10, then symbol-specific scaling factor, then global normalization
                    processed_data[:, i] = (np.log10(df[col]) - scaling_factor - global_constants['ohlc_shift']) / global_constants['ohlc_scale']
                except FloatingPointError:
                    problematic_indices = np.where(df[col] <= 0)[0]
                    for idx in problematic_indices:
                        print(f"Warning: Invalid value in symbol {symbol}, column '{col}' at timestamp {timestamps[idx]}")
                        print(f"Value: {df[col].iloc[idx]}")
                    
                    # Replace invalid values with NaN or a small positive number
                    df[col] = df[col].replace(0, np.nan)
                    processed_data[:, i] = (np.log10(df[col].fillna(1e-8)) - scaling_factor - global_constants['ohlc_shift']) / global_constants['ohlc_scale']

        # Process turnover with global normalization
        processed_data[:, 4] = (np.log10(df['turnover'].clip(lower=1)) - global_constants['turnover_shift']) / global_constants['turnover_scale']

        return {'data': processed_data, 'timestamps': timestamps}


    def calculate_preprocessing_constants(self, symbols):
        constants = {}
        daily_data_path = os.path.join("../data/klines", "1 day")
        
        all_log_ohlc = []
        all_log_turnover = []
        
        for symbol in tqdm(symbols, desc="Calculating constants"):
            daily_file = os.path.join(daily_data_path, f"{symbol}")
            
            if not os.path.exists(daily_file):
                print(f"Warning: Daily data file not found for {symbol}. Skipping...")
                continue
            
            df = pd.read_csv(daily_file)
            
            if df.empty:
                print(f"Warning: Empty daily data for {symbol}. Skipping...")
                continue
            
            first_close_price = df['close'].iloc[0]
            scaling_factor = np.log10(first_close_price)
            
            constants[symbol] = {"scaling_factor": scaling_factor}
            
            # Collect log10 of OHLC and turnover data for global normalization
            log_ohlc = np.log10(df[['open', 'high', 'low', 'close']].values.flatten())
            log_turnover = np.log10(df['turnover'])
            
            all_log_ohlc.extend(log_ohlc[np.isfinite(log_ohlc)])
            all_log_turnover.extend(log_turnover[np.isfinite(log_turnover)])
        
        # Calculate global normalization constants
        ohlc_mean = np.mean(all_log_ohlc)
        ohlc_std = np.std(all_log_ohlc)
        turnover_mean = np.mean(all_log_turnover)
        turnover_std = np.std(all_log_turnover)
        
        # Add global constants
        constants['global'] = {
            "ohlc_shift": ohlc_mean,
            "ohlc_scale": ohlc_std,
            "turnover_shift": turnover_mean,
            "turnover_scale": turnover_std
        }
        
        return constants

#endregion