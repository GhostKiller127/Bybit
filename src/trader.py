import os
import time
import signal
import asyncio
import logging
import smtplib
import traceback
import pandas as pd
from actor import Actor
import ccxt.async_support as ccxt
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import configs_trader as configs


class TradingState:
    def __init__(self):
        self.balance = 0
        self.positions = {}
        self.leverage = 1
        self.orders = {}
        self.trading_symbols = {}
        self.markets = {}

#region init

class Trader:
    def __init__(self):
        self.exchange = None
        self.running = True
        self.auto_trading = False
        self.markets = None
        self.logger = logging.getLogger(__name__)


    async def initialize(self):
        try:
            print("Initializing exchange connection...")
            time_difference = await self.get_time_difference()
            self.create_exchange(time_difference)
            await self.exchange.load_markets()
            self.markets = self.exchange.markets
        except Exception as e:
            self.logger.error(f"Error initializing trader: {e}")
            raise


    def create_exchange(self, time_difference):
        if self.exchange:
            asyncio.create_task(self.exchange.close())
        
        self.exchange = ccxt.bybit({
            'apiKey': configs.API_KEY,
            'secret': configs.API_SECRET,
            'options': {
                'defaultType': 'linear',
                'adjustForTimeDifference': True,
                'timeDifference': time_difference
            }
        })
        self.exchange.verbose = False
        # self.exchange.timeout = 30000


    async def get_time_difference(self):
        temp_exchange = ccxt.bybit()
        try:
            server_time = await temp_exchange.fetch_time()
            return server_time - int(time.time() * 1000)
        except Exception as e:
            self.logger.error(f"Error getting time difference: {e}")
            raise
        finally:
            await temp_exchange.close()


    async def recreate_exchange(self):
        try:
            time_difference = await self.get_time_difference()
            self.create_exchange(time_difference)
            await self.exchange.load_markets()
            self.markets = self.exchange.markets
        except Exception as e:
            self.logger.error(f"Error updating time difference: {e}")
            raise

#region shortcuts

    async def process_command(self, command):
        try:
            parts = command.split()
            if not parts:
                return
            
            cmd = parts[0].lower()
            if cmd == 'help':
                self.print_help()
            elif cmd == 'refresh':
                await self.refresh_markets()
            elif cmd == 'reload':
                await self.reload_everything()
            elif cmd == 'balance':
                await self.get_balance()
            elif cmd == 'positions':
                await self.get_positions()
            elif cmd == 'leverage':
                if len(parts) > 1:
                    await self.set_account_leverage(int(parts[1]))
                else:
                    print("Usage: leverage <leverage>")
            elif cmd == 'price':
                if len(parts) > 1:
                    await self.get_market_price(parts[1])
                else:
                    print("Please specify a trading pair.")
            elif cmd == 'cancel':
                await self.cancel_all_orders()
            elif cmd == 'actor':
                await self.setup_actor()
            elif cmd == 'actor_run':
                await self.run_actor()
            elif cmd == 'auto':
                self.toggle_auto_trading()
            elif cmd == 'close_all':
                await self.emergency_close_all()
            elif cmd in ['buy', 'sell']:
                if len(parts) == 3:
                    await self.open_position(parts[1], cmd, float(parts[2]))
                else:
                    print(f"Usage: {cmd} <symbol> <amount>")
            elif cmd == 'close':
                if len(parts) == 2:
                    await self.close_position(parts[1])
                else:
                    print("Usage: close <symbol>")
            elif cmd == 'email':
                await self.send_status_email()
            elif cmd == 'exit':
                self.running = False
            else:
                print("Unknown command. Type 'help' for available commands.")
        except ValueError as e:
            print(f"Invalid input: {e}")
        except Exception as e:
            error_message = f"Error processing command '{command}': {e}"
            self.logger.error(error_message)
            self.logger.error(traceback.format_exc())
            print("An error occurred. Check the log file for details.")


    def setup_signal_handlers(self):
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self.signal_handler)

#endregion
#region assets

    async def get_balance(self):
        try:
            balance = await self.exchange.fetch_balance()
            self.total_usd = balance['total']['USDT']
            self.available_usd = balance['free']['USDT']
            self.used_usd = balance['used']['USDT']
            print(f"Total balance: {self.total_usd} USDT")
            print(f"Available balance: {self.available_usd} USDT")
            print(f"Used balance: {self.used_usd} USDT")
            if self.total_usd < configs.MIN_BALANCE_THRESHOLD:
                await self.send_email_alert(f"Low balance alert: {self.total_usd} USDT")
        except ccxt.ExchangeError as e:
            self.logger.error(f"Error fetching balance: {e}")
            print("Failed to fetch balance. Please try again later.")


    async def get_positions(self):
        try:
            positions = await self.exchange.fetch_positions(None, {'type': 'linear'})
            if positions:
                for position in positions:
                    if position['contracts'] > 0:
                        print(f"Symbol: {position['symbol']}, Side: {position['side']}, Contracts: {position['contracts']}, Unrealized PNL: {position['unrealizedPnl']}")
            else:
                print("No open positions.")
        except ccxt.ExchangeError as e:
            self.logger.error(f"Error fetching positions: {e}")

#endregion
#region leverage

    async def set_account_leverage(self, leverage=1, verbose=True):
        try:
            tasks = []
            for symbol, market in self.markets.items():
                if (market['linear'] and  
                    market['quote'] == 'USDT' and 
                    market['active'] and
                    'USDT' in symbol):
                    tasks.append(self.set_leverage_for_symbol(symbol, leverage, verbose))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            processed_count = len(results)
            print(f"Account leverage check completed. Processed {processed_count} active USDT linear futures contracts.")
        except Exception as e:
            self.logger.error(f"Error setting account leverage: {e}")
            print("Failed to set account leverage. Please try again later.")


    async def set_leverage_for_symbol(self, symbol, leverage, verbose):
        try:
            response = await self.exchange.set_leverage(leverage, symbol)
            if verbose:
                print(f"Leverage for {symbol} set to {leverage}. Response: {response}")
        except ccxt.ExchangeError as e:
            error_message = str(e)
            if "leverage not modified" in error_message.lower():
                if verbose:
                    print(f"Leverage for {symbol} is already set to {leverage}")
            else:
                self.logger.error(f"Error setting leverage for {symbol}: {e}")
                print(f"Error setting leverage for {symbol}: {e}")

#endregion
#region orders

    async def cancel_all_orders(self):
        try:
            result = await self.exchange.cancel_all_orders(None, {'type': 'linear'})
            print(f"All orders cancelled: {result}")
        except ccxt.ExchangeError as e:
            self.logger.error(f"Error cancelling all orders: {e}")


    async def emergency_close_all(self):
        try:
            positions = await self.exchange.fetch_positions(None, {'type': 'linear'})
            for position in positions:
                if position['contracts'] > 0:
                    await self.exchange.create_market_order(
                        position['symbol'],
                        'sell' if position['side'] == 'long' else 'buy',
                        position['contracts']
                    )
            print("All positions closed.")
            await self.send_email_alert("Emergency: All positions closed")
        except ccxt.ExchangeError as e:
            self.logger.error(f"Error closing all positions: {e}")


    async def open_position(self, symbol, side, amount):
        try:
            full_symbol = f"{symbol}/USDT:USDT"
            if full_symbol not in self.markets:
                print(f"Invalid symbol: {symbol}")
                return
            
            order = await self.exchange.create_market_order(full_symbol, side, amount)
            print(f"Opened position: {full_symbol}, Side: {side}, Contracts: {amount}")
        except ccxt.ExchangeError as e:
            self.logger.error(f"Error opening position: {e}")
            print(f"Failed to open position for {symbol}. Please check your input and try again.")


    async def close_position(self, symbol):
        try:
            full_symbol = f"{symbol}/USDT:USDT"
            if full_symbol not in self.markets:
                print(f"Invalid symbol: {symbol}")
                return
            
            position = await self.exchange.fetch_position(full_symbol)
            if position and position['contracts'] > 0:
                contracts = position['contracts']
                side = 'sell' if position['side'] == 'long' else 'buy'
                                
                order = await self.exchange.create_market_order(
                    full_symbol,
                    side,
                    contracts
                )
                print(f"Closed position: {full_symbol}, Side: {side}, Contracts: {contracts}")
            else:
                print(f"No open position for {full_symbol}")
        except ccxt.ExchangeError as e:
            self.logger.error(f"Error closing position: {e}")
            print(f"Failed to close position for {symbol}. Please try again later.")
        except Exception as e:
            self.logger.error(f"Unexpected error closing position: {type(e).__name__}: {e}")
            print(f"An unexpected error occurred while closing position for {symbol}.")
            print(f"Position data: {position}")

#endregion
#region market

    async def get_market_price(self, symbol):
        try:
            # Use the cached market data to validate the symbol
            if f"{symbol}/USDT:USDT" not in self.markets:
                print(f"Invalid symbol: {symbol}")
                return
            
            ticker = await self.exchange.fetch_ticker(f"{symbol}/USDT:USDT")
            print(f"Current {symbol} price: {ticker['last']} USDT")
        except ccxt.ExchangeError as e:
            self.logger.error(f"Error fetching price for {symbol}: {e}")
            print(f"Failed to fetch price for {symbol}. Please try again later.")


    async def refresh_markets(self):
        try:
            print("Refreshing exchange connection and market data...")
            await self.recreate_exchange()
            self.markets = await self.exchange.load_markets(True)
            print("Exchange connection updated and market data refreshed")
        except ccxt.ExchangeError as e:
            self.logger.error(f"Error refreshing exchange connection and market data: {e}")
            print("Failed to refresh market data. Please try again later.")
    
    
    async def reload_market(self):
        try:
            await self.actor.downloader.get_tick_size_per_mille()
            
            data_dir = os.path.join('..', 'data')
            csv_files = [f for f in os.listdir(data_dir) if f.startswith('tick_size') and f.endswith('.csv')]
            csv_file = os.path.join(data_dir, csv_files[0])
            df = pd.read_csv(csv_file)
            
            self.current_market = {}
            for _, row in df.iterrows():
                symbol = row['Symbol']
                self.current_market[symbol] = {
                    'last_price': row['Last Price'],
                    'quantity_step': row['Quantity Step']
                }
            
            print(f"Market data reloaded. {len(self.current_market)} symbols processed.")
        except Exception as e:
            self.logger.error(f"Error in reload_market: {e}")

#endregion
#region email

    async def send_status_email(self):
        try:
            balance = await self.exchange.fetch_balance()
            positions = await self.exchange.fetch_positions(None, {'type': 'linear'})
            
            body = f"Balance: {balance['total']['USDT']} USDT\n\nOpen Positions:\n"
            for position in positions:
                if position['contracts'] > 0:
                    body += f"{position['symbol']}: Side: {position['side']}, Contracts: {position['contracts']}, Unrealized PNL: {position['unrealizedPnl']}\n"
            
            await self.send_email("Trader Status Update", body)
            print("Status email sent.")
        except Exception as e:
            self.logger.error(f"Error sending status email: {e}")


    async def send_email_alert(self, message):
        await self.send_email("Trader Alert", message)


    async def send_email(self, subject, body):
        msg = MIMEMultipart()
        msg['From'] = configs.EMAIL_SENDER
        msg['To'] = configs.EMAIL_RECIPIENT
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP(configs.SMTP_SERVER, configs.SMTP_PORT)
            server.starttls()
            server.login(configs.EMAIL_SENDER, configs.EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")
            print("Failed to send email. Please check your email configuration.")

#endregion
#region utils

    def toggle_auto_trading(self):
        self.auto_trading = not self.auto_trading
        print(f"Auto-trading {'enabled' if self.auto_trading else 'disabled'}")


    def print_help(self):
        print("\n".join([f"{cmd}: {desc}" for cmd, desc in configs.SHORTCUTS.items()]))


    def signal_handler(self):
        print("\nShutting down...")
        self.running = False


    async def close(self):
        try:
            await self.exchange.close()
        except Exception as e:
            self.logger.error(f"Error closing exchange: {e}")

#endregion
#region actor

    async def setup_actor(self):
        try:
            print("Setting up actor...")
            self.actor = await Actor().initialize()
        except Exception as e:
            self.logger.error(f"Error in setup_actor: {e}")


    def get_symbol_amount(self, symbol, action):
        total_balance = self.total_usd * configs.MAX_BALANCE_USAGE * abs(action)
        symbol_balance = total_balance / configs.MAX_TRADING_SYMBOLS
        available_balance = min(self.available_usd, symbol_balance)
        
        quantity_step = self.current_market[symbol]['quantity_step']
        last_price = self.current_market[symbol]['last_price']
        
        max_quantity = available_balance / last_price
        max_quantity_steps = int(max_quantity / quantity_step)

        return max_quantity_steps * quantity_step


    async def execute_actions(self, chosen_actions):
        try:
            for symbol, data in chosen_actions.items():
                amount = self.get_symbol_amount(symbol, data['action'])
                if data['action'] > 0:
                    await self.open_position(symbol[:-4], 'buy', amount)
                elif data['action'] < 0:
                    await self.open_position(symbol[:-4], 'sell', amount)
                elif data['action'] == 0:
                    await self.close_position(symbol[:-4])
        except Exception as e:
            self.logger.error(f"Error in execute_actions: {e}")
    
#endregion
#region trading

    async def reload_everything(self):
        try:
            await self.refresh_markets()
            await self.reload_market()
            await self.get_balance()
            await self.actor.downloader.download_usdt_symbols()
            self.actor.reload_symbols()
        except Exception as e:
            self.logger.error(f"Error in reload_everything: {e}")
    

    async def run_actor(self):
        try:
            chosen_actions = await self.actor.run()
            print(chosen_actions)
            await self.execute_actions(chosen_actions)
        except Exception as e:
            self.logger.error(f"Error in run_actor: {e}")


    async def run_auto_trading(self):
        while self.auto_trading:
            try:
                now = datetime.now()
                
                if now.minute == 57 and now.second <= 5:
                    await self.reload_everything()
                
                elif now.minute == 0 and now.second >= 6 and now.second <= 10:
                    await self.run_actor()
                
                if now.minute >= 57 or now.minute < 5:
                    await asyncio.sleep(1)
                elif now.minute >= 5 and now.minute <= 55:
                    await asyncio.sleep(60)
                else:
                    await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in run_auto_trading: {type(e).__name__} - {e}")
                self.logger.error(traceback.format_exc())
                await self.send_email_alert(f"Auto-trading error: {type(e).__name__} - {e}")
                await asyncio.sleep(60)

#endregion
#region main

    async def run_interactive(self):
        self.setup_signal_handlers()
        print("Interactive Trader running. Type 'help' for commands.")
        auto_trading_task = None
        try:
            while self.running:
                try:
                    command = await asyncio.get_event_loop().run_in_executor(None, input, "Enter command: ")
                    await self.process_command(command)
                    
                    if self.auto_trading and auto_trading_task is None:
                        auto_trading_task = asyncio.create_task(self.run_auto_trading())
                    elif not self.auto_trading and auto_trading_task is not None:
                        auto_trading_task.cancel()
                        auto_trading_task = None
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error processing command: {e}")
                    print("An error occurred while processing your command. Please try again.")
        finally:
            if auto_trading_task:
                auto_trading_task.cancel()
            await self.close()


def setup_logging():
    log_file = 'trader.log'
    
    if os.path.exists(log_file):
        os.remove(log_file)
    
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


async def main():
    trader = None
    try:    
        trader = Trader()
        await trader.initialize()
        await trader.run_interactive()
    except Exception as e:
        error_message = f"Error in main loop: {type(e).__name__} occurred - {e}"
        logging.error(error_message)
        if trader:
            await trader.send_email_alert(error_message)
    finally:
        if trader:
            await trader.close()


if __name__ == "__main__":
    setup_logging()
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"An unexpected error occurred outside the main trader loop: {type(e).__name__} - {e}")

#endregion