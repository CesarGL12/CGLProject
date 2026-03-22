from src.sentiment_model import SentimentModel
from src.explain_model import explain_text as explain_algorithm

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


def predict_label(text: str) -> str:

    return predict_text(text)["prediction"]
    

def baseline_predict(text: str) -> str:
    """
    Very simple baseline using keyword rules.
    """

    text_lower = text.lower()

    positive_words = ["good", "great", "love", "excellent", "amazing"]
    negative_words = ["bad", "terrible", "hate", "awful", "horrible"]

    pos_count = sum(word in text_lower for word in positive_words)
    neg_count = sum(word in text_lower for word in negative_words)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"