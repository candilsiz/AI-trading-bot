from bs4 import BeautifulSoup
import requests
import re

def linkSearcher(ticker):
    search_url = f'https://www.google.com/search?q=yahoo+finance+{ticker}&tbm=nws'
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return [a['href'] for a in soup.find_all('a')]

def urlFilter(urls, exclude_list):
    filtered = []
    for url in urls:
        if 'https://finance.yahoo.com/news/' in url and not any(exclude in url for exclude in exclude_list):
            clean_url = re.findall(r'(https?://\S+)', url)[0].split('&')[0]
            filtered.append(clean_url)
    return list(set(filtered))

def scrapper(urls):
    # may add wait statement
    articles = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join(p.text for p in paragraphs)
 
        articles.append(text)
    return articles

def fethNews(tickers, exclude_list):
    raw_urls = {ticker: linkSearcher(ticker) for ticker in tickers}
    cleaned_urls = {ticker: urlFilter(raw_urls[ticker], exclude_list) for ticker in tickers}
    articles = {ticker: scrapper(cleaned_urls[ticker]) for ticker in tickers}
    return articles

