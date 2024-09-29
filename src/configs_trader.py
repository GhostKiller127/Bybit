import os


# API Configuration
API_KEY = os.environ.get('BYBIT_API_KEY', '8GCFE4qfBCPuPDtZYU')
API_SECRET = os.environ.get('BYBIT_API_SECRET', 'crxPnwd5JuMPpBq44WQfQW5CXkFWIGSpA9Ou')
MAX_REQUESTS = 500
REQUEST_INTERVAL = 5
SEMAPHORE_SIZE = 20


# Email Configuration
EMAIL_SENDER = 'ghostkiller2070@gmail.com'
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'igra hkcw bkgq cqwq')
EMAIL_RECIPIENT = 'ghostkiller2070@gmail.com'
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587


# Actor Configuration
ACTOR = 'S5,rng27,bs64,s10,b10,e0.995,g0.99,y10.6,y21.0,x10.0,test'
PROBABILITY_THRESHOLD = 0.9
Q_THRESHOLD = 0.2


# Trader Configuration
REFRESH_BEFORE_INTERVAL = 300
MAX_TRADING_SYMBOLS = 3
MAX_BALANCE_USAGE = 0.95
MIN_BALANCE_THRESHOLD = 10


SHORTCUTS = {
    "help": "Show this help message",
    "refresh": "Update exchange connection and refresh market data",
    "reload": "Reload market data",
    "balance": "Show account balance",
    "positions": "Show current open positions",
    "leverage <leverage>": "Set the leverage for all symbols",
    "price <symbol>": "Show current market price for a symbol",
    "cancel": "Cancel all open orders",
    "test": "Test run",
    "auto": "Toggle auto-trading on/off",
    "buy <symbol> <amount>": "Open a long position",
    "sell <symbol> <amount>": "Open a short position",
    "close <symbol>": "Close a specific position",
    "close_all": "Emergency close all positions",
    "email": "Send a status email",
    "exit": "Exit the program"
}