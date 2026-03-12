import argparse
import pandas as pd
import re
import os

# Create results folder
os.makedirs("results", exist_ok=True)

from sentiment_model import SentimentModel

model = SentimentModel()
classifier = model.classifier

STOPWORDS = {
    "en": {"this", "a", "the", "is", "was", "but", "and", "of", "to", "in"},
    "es": {"este", "esta", "es", "fue", "pero", "y", "de", "la", "el", "en"},
    "fr": {"ce", "cette", "est", "était", "mais", "et", "de", "la", "le", "en"}
}


def get_prediction(text):
    outputs = classifier(text)[0]
    scores = {int(item["label"][0]): item["score"] for item in outputs}

    weighted_score = sum(star * prob for star, prob in scores.items())

    if weighted_score <= 2:
        sentiment = "Negative"
    elif weighted_score < 4:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"

    confidence = max(scores.values())

    return sentiment, confidence, scores


def explain_text(text, language="en"):
    words = re.findall(r"\b\w+\b", text)

    base_sentiment, base_conf, base_scores = get_prediction(text)
    base_star = max(base_scores, key=base_scores.get)
    base_prob = base_scores[base_star]

    modified_texts = []
    valid_words = []

    for i, word in enumerate(words):
        if word.lower() in STOPWORDS.get(language, set()):
            continue

        modified = words[:i] + words[i+1:]
        modified_texts.append(" ".join(modified))
        valid_words.append(word)

    if not modified_texts:
        return base_sentiment, base_conf, []

    batch_outputs = classifier(modified_texts)

    word_importance = []

    for word, output in zip(valid_words, batch_outputs):
        scores = {int(item["label"][0]): item["score"] for item in output}
        new_prob = scores.get(base_star, 0)

        importance = base_prob - new_prob

        if abs(importance) > 0.02:
            word_importance.append((word, importance))

    word_importance.sort(key=lambda x: abs(x[1]), reverse=True)

    return base_sentiment, base_conf, word_importance


def explain_file(filepath):
    df = pd.read_csv(filepath)

    results = []

    for _, row in df.iterrows():
        text = row["text"]
        language = row.get("language", "en")

        sentiment, confidence, important_words = explain_text(text, language)

        results.append({
            "text": text,
            "language": language,
            "prediction": sentiment,
            "confidence": round(confidence, 4),
            "important_words": important_words[:5]
        })

    results_df = pd.DataFrame(results)
    output_path = "results/explanations.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved explanations to {output_path}")