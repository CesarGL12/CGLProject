from sentiment_model import SentimentModel

model = SentimentModel()

# Sample test sentences in all 3 languages
samples = {
    "en": "This movie was amazing!",
    "es": "Esta película fue terrible.",
    "fr": "Ce restaurant était correct."
}

print("\n=== BASELINE SENTIMENT RESULTS ===\n")

for lang, text in samples.items():
    sentiment, confidence, _ = model.predict(text)

    print(f"Language: {lang}")
    print(f"Text: {text}")
    print(f"Prediction: {sentiment}")
    print(f"Confidence: {confidence:.3f}\n")