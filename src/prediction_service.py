from sentiment_model import SentimentModel
from explain_model import explain_text as explain_algorithm

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

model = SentimentModel()


def predict_text(text: str, language: str = "en"):
    """
    Predict sentiment for a given text.

    Returns a standardized dictionary.
    """

    sentiment, confidence, scores = model.predict(text)

    result = {
        "text": text,
        "language": language,
        "prediction": sentiment,
        "confidence": round(confidence, 4),
        "scores": scores
    }

    return result


def explain_text(text: str, language: str = "en"):
    """
    Generate sentiment explanation for text.

    Returns standardized dictionary.
    """

    sentiment, confidence, important_words = explain_algorithm(text, language)

    result = {
        "text": text,
        "language": language,
        "prediction": sentiment,
        "confidence": round(confidence, 4),
        "important_words": [
            {"word": w, "importance": round(i, 4)}
            for w, i in important_words
        ]
    }

    return result
