from transformers import pipeline

# Load pretrained multilingual sentiment model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
classifier = pipeline("sentiment-analysis", model=model_name)

# Sample test sentences in all 3 languages
samples = {
    "en": "This movie was amazing!",
    "es": "Esta película fue terrible.",
    "fr": "Ce restaurant était correct."
}

def map_sentiment(result):
    label = result["label"]   # e.g., "4 stars"
    score = result["score"]

    stars = int(label[0])  # get the number

    if stars <= 2:
        sentiment = "Negative"
    elif stars == 3:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"

    return sentiment, score

print("\n=== BASELINE SENTIMENT RESULTS ===\n")

for lang, text in samples.items():
    raw = classifier(text)[0]
    sentiment, confidence = map_sentiment(raw)

    print(f"Language: {lang}")
    print(f"Text: {text}")
    print(f"Prediction: {sentiment}")
    print(f"Confidence: {confidence:.3f}\n")