import ccxt, os
import pandas as pd
from datetime import datetime

class Loader:
    """
    Loader for fetching historical OHLCV data from Binance Futures.

    Usage:
    ```python
    from datetime import datetime
    loader = Loader(
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframe="1h",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 6, 1),
        save_directory="./data"
    )
    loader.download_data_for_symbols()
    ```
    This will download the data and save each symbol's OHLCV history as a Feather file in the specified directory.

    Note: 
    - u can change the exchange but from my own experience, this config has the best data history and good quality
    """

    def __init__(self,
                 symbols: list,
                 timeframe: str = "1d",
                 start_date: datetime = None,
                 end_date: datetime = None,
                 save_directory: str = "."):        
        
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
            'timeout': 30000})
        
        self.symbols = symbols
        self.timeframe = timeframe
        self.limit = 1000
        self.start_date = start_date
        self.end_date = end_date
        self.save_directory = save_directory

        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

    def fetch_historical_data(self, symbol: str) -> list:
        all_data = []
        since = self.exchange.parse8601(self.start_date.isoformat())
        end_ts = self.exchange.parse8601(self.end_date.isoformat())

        while True:
            data = self.exchange.fetch_ohlcv(symbol, self.timeframe, since=since, limit=self.limit)
            if not data:
                break
            all_data += data
            since = data[-1][0] + 1  
            if data[-1][0] >= end_ts or len(data) < self.limit:
                break

        filtered = [item for item in all_data if item[0] < end_ts]

        return [
            {
                "timestamp": datetime.fromtimestamp(item[0] / 1000),
                "open": item[1],
                "high": item[2],
                "low": item[3],
                "close": item[4],
                "volume": item[5],
            }
            for item in filtered
        ]

    def download_data_for_symbols(self) -> None:
        """
        Download data for all symbols and save each as a Feather file in your save_directory
        """
        for symbol in self.symbols:
            print(f"Loading data for {symbol}...")
            historical_data = self.fetch_historical_data(symbol)
            df = pd.DataFrame(historical_data).dropna()

            if not df.empty and df.iloc[-1]["timestamp"] >= self.end_date:
                df = df.iloc[:-1]

            symbol_clean = symbol.replace("/", "_").replace(":", "_")
            filename = os.path.join(
                self.save_directory, f"{symbol_clean}_{self.timeframe}.feather"
            )
            df.to_feather(filename)
            print(f"Data for {symbol} saved in {filename}")

#

