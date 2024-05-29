from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple 

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
labels = ["positive", "negative", "neutral"]

def news_signal(news):
    if news:
        tokens = tokenizer(news, return_tensors="pt", padding=True)
        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)
        probability = result[torch.argmax(result)].item()  # Convert to float
        sentiment = labels[torch.argmax(result)].upper()  # Convert to uppercase string
        return probability, sentiment
    else:
        return 0.0, "NEUTRAL"

if __name__ == "__main__":
    probability, sentiment = news_signal('VanEck’s spot Ether ETF application due May 23. This potential regulatory shift has elevated Ether’s price')
    print(f'Probability: {probability:.4f}')
    print(f'News: {sentiment}')