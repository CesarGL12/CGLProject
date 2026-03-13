from prediction_service import predict_text

samples = {
    "en": "This movie was amazing!",
    "es": "Esta película fue terrible.",
    "fr": "Ce restaurant était correct."
}

print("\n=== BASELINE SENTIMENT RESULTS ===\n")

for lang, text in samples.items():

    result = predict_text(text, lang)

    print(f"Language: {result['language']}")
    print(f"Text: {result['text']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}\n")