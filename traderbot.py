# pip install lumibot == 2.9.13
from config import ALPACA_CONFIG
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime, timedelta
from alpaca_trade_api import REST
from models.finbert_model import news_signal
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

class MLTrader(Strategy):
    # will run once
    def initialize(self, symbol:str='SPY', risked_cash:float=0.5):
        self.symbol = symbol
        self.sleeptime = '24H'
        self.last_trade = None
        self.risked_cash = risked_cash
        self.api = REST(base_url=ALPACA_CONFIG["ENDPOINT"], 
                        key_id=ALPACA_CONFIG["API_KEY"],
                        secret_key=ALPACA_CONFIG["API_SECRET"])       
        
    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')
    
    def get_assets():
        assets = {
            'BTH' : ['Bitcoin','BTC', 'BTH'],
            'ETH' : ['Ethereum', 'ETH'], 
            'AAPL': ['Apple Inc.', 'Apple', 'AAPL'],
            'GOOGL': ['Google','Alphabet Inc.', 'GOOGL']
}
        return assets
    
    def normalize_text(self, text:str):
        text = text.lower()
        tokens = word_tokenize(text)
        return tokens

    def asset_search(self, text, assets):
        normalized_text = self.normalize_text(text)
        detected_assets = set()  
        for asset_key, names in assets.items():
            for name in names:
                if name.lower() in normalized_text:
                    detected_assets.add(asset_key)  
        return detected_assets

    def get_sentiment(self):
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, 
                                 start=three_days_prior, end=today)
        news = [event.__dict__['_raw']['headline'] for event in news]
        print(news)    
        probability, sentiment = news_signal(news)
        return probability, sentiment 

        
    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.risked_cash / last_price)
        return cash, last_price, quantity

    # will run everyt ime when we get new data
    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment  = self.get_sentiment()
        if cash > last_price:
            if sentiment == 'positive' and probability > 0.8:
                if self.last_trade == 'sell':
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    'buy',
                    type = 'bracket', # what is bracket?
                    take_profit_price = last_price * 1.20,
                    stop_loss_price = last_price * 0.95 
                )
                self.submit_order(order)
                self.last_trade = 'buy'
            elif sentiment == 'negative' and probability > 0.8:
                if self.last_trade == 'buy':
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    'sell',
                    type = 'bracket',
                    take_profit_price = last_price * 0.80,
                    stop_loss_price = last_price * 1.05 
                )
                self.submit_order(order)
                self.last_trade = 'sell'

start_date = datetime(2023,1,1)
end_date = datetime(2023,12,31)

broker = Alpaca(ALPACA_CONFIG)
strategy = MLTrader(name = 'news-interpreter', broker = broker, 
                    parameters={'symbol' : 'SPY', 
                                'risked_cash' : 0.5}) 
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters = {'symbol' : 'SPY',
                   'risked_cash' : 0.5}
)
