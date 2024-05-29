from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import torch
from typing import Tuple

tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert", use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained("ElKulako/cryptobert", num_labels=3)
labels = ["BEARISH", "NEUTRAL", "BULLISH"]

def news_signal(news: str) -> Tuple[float, str]:
    if news:
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64, truncation=True, padding="max_length")
        results = pipe(news)
        sentiment = results[0]['label'].upper()  # Get label 
        probability = results[0]['score']  # probability score
        return probability, sentiment
    else:
        return 0.0, "NEUTRAL"

if __name__ == "__main__":
    news_text = 'Ethereum topped this week, at $1891, in my previous article. Weekly and monthly charts point to higher prices.'
    probability, sentiment = news_signal(news_text)
    print(f'Probability: {probability:.4f}')
    print(f'News Sentiment: {sentiment}')