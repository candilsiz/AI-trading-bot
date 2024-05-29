from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# Summarization need shorten the text (model accepts max 512 token sequence)
def load_model():
    model_name = "human-centered-summarization/financial-summarization-pegasus"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def generate_summary(tokenizer, model, text, max_length = 100, num_beams = 5):
    input_ids = tokenizer(text, truncation =True, return_tensors="pt").input_ids
    summary_ids = model.generate(input_ids, max_length=max_length, num_beams=num_beams, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
def summarize(text):
    tokenizer, model = load_model()
    summary = generate_summary(tokenizer, model, text)
    return summary

            

    
