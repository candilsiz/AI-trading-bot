from scrapper import fethNews
from summarizer import summarize
from models.finbert_model import news_signal

def word_count(text):
    return len(text.split(' '))

# monitored_tickers = ['ETH', 'GME', 'TSLA', 'BTC']
monitored_tickers = ['ETH']
exclude_list = ['maps', 'policies', 'preferences', 'accounts', 'support']

articles = fethNews(monitored_tickers,exclude_list )

# will use structured or non-structured databases to store data on daily basis.
for ticker, article_list in articles.items():
    for article in article_list:
        print(article)
        print(f"Articles for {ticker}:")
        print("Article Data type is: ", type(article))
        print("Article Word Count: ", word_count(article))
        article_summary = summarize(article)
        probability, sentiment = news_signal(article_summary)
        print("Article Summary: ", article_summary)
        print("Article Sentiment",sentiment)
        print("Article Sentiment Probability",probability)
        print("Article Summary Word Count: ", word_count(article_summary))
        print("\n---\n")