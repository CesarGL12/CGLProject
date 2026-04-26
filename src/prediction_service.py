from src.sentiment_model import SentimentModel
from src.explain_model import explain_text as explain_algorithm

import pandas as pd

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

model = SentimentModel()


#def predict_text(text: str, language: str = "en"):
def predict_text(text: str):
    """
    Predict sentiment for a given text.

    Returns a standardized dictionary.
    """

    sentiment, confidence, scores = model.predict(text)

    result = {
        "text": text,
        #"language": language,
        "prediction": sentiment,
        "confidence": round(confidence, 4),
        "scores": scores
    }

    return result


#def explain_text(text: str, language: str = "en"):
def explain_text(text: str):
    """
    Generate sentiment explanation for text.

    Returns standardized dictionary.
    """

    sentiment, confidence, important_words = explain_algorithm(text)

    result = {
        "text": text,
        #"language": language,
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



import unicodedata

def normalize(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def baseline_predict(text: str):
    #text = text.lower()
    text = normalize(text.lower())

    KEYWORDS = {
        "positive": [
            "good", "great", "love", "excellent", "amazing", "happy",
            "bueno", "genial", "excelente", "increíble", "amor", "feliz",
            "bon", "génial", "excellent", "incroyable", "amour", "heureux"
        ],
        "negative": [
            "bad", "terrible", "hate", "awful", "horrible", "worst",
            "malo", "terrible", "horrible", "odio", "triste", "peor",
            "mauvais", "terrible", "horrible", "haine", "triste", "pire"
        ]
    }

    #pos_count = sum(word in text for word in KEYWORDS["positive"])
    #neg_count = sum(word in text for word in KEYWORDS["negative"])
    pos_count = sum(normalize(word) in text for word in KEYWORDS["positive"])
    neg_count = sum(normalize(word) in text for word in KEYWORDS["negative"])

    total = pos_count + neg_count

    if pos_count > neg_count:
        prediction = "positive"
    elif neg_count > pos_count:
        prediction = "negative"
    else:
        prediction = "neutral"

    #confidence = (max(pos_count, neg_count) / total) if total > 0 else 0
    confidence = max(pos_count, neg_count) / (pos_count + neg_count + 2)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "pos_count": pos_count,
        "neg_count": neg_count
    }

# def baseline_predict(text: str) -> str:
#     """
#     Multilingual rule-based baseline using keyword matching.
#     Supports English, Spanish, and French.
#     """
# 
#     text = text.lower()
# 
#     KEYWORDS = {
#         "positive": [
#             # English
#             "good", "great", "love", "excellent", "amazing", "happy",
#             # Spanish
#             "bueno", "genial", "excelente", "increíble", "amor", "feliz",
#             # French
#             "bon", "génial", "excellent", "incroyable", "amour", "heureux"
#         ],
#         "negative": [
#             # English
#             "bad", "terrible", "hate", "awful", "horrible", "worst",
#             # Spanish
#             "malo", "terrible", "horrible", "odio", "triste", "peor",
#             # French
#             "mauvais", "terrible", "horrible", "haine", "triste", "pire"
#         ]
#     }
# 
#     #pos_count = sum(word in text for word in KEYWORDS["positive"])
#     #neg_count = sum(word in text for word in KEYWORDS["negative"])
#     pos_count = sum(f" {word} " in f" {text} " for word in KEYWORDS["positive"])
#     neg_count = sum(f" {word} " in f" {text} " for word in KEYWORDS["negative"])
# 
#     if pos_count > neg_count:
#         return "positive"
#     elif neg_count > pos_count:
#         return "negative"
#     else:
#         return "neutral"

# def baseline_predict(text: str) -> str:
#     """
#     Very simple baseline using keyword rules.
#     """
# 
#     text_lower = text.lower()
# 
#     positive_words = ["good", "great", "love", "excellent", "amazing"]
#     negative_words = ["bad", "terrible", "hate", "awful", "horrible"]
# 
#     pos_count = sum(word in text_lower for word in positive_words)
#     neg_count = sum(word in text_lower for word in negative_words)
# 
#     if pos_count > neg_count:
#         return "Positive"
#     elif neg_count > pos_count:
#         return "Negative"
#     else:
#         return "Neutral"

#def map_label(label):
#    try:
#        label = int(label)
#    except:
#        return None  # invalid label
#
#    if label in [1, 2]:
#        return "negative"
#    elif label == 3:
#        return "neutral"
#    elif label in [4, 5]:
#        return "positive"
#    else:
#        return None

# def evaluate_models():
#     """
#     Compare model vs baseline on sample datasets.
#     Returns accuracy dictionary.
#     """
# 
#     paths = [
#         "data/sample/en_sample.csv",
#         "data/sample/es_sample.csv",
#         "data/sample/fr_sample.csv"
#     ]
# 
#     total = 0
#     correct_model = 0
#     correct_baseline = 0
# 
#     for path in paths:
#         df = pd.read_csv(path)
# 
#         for _, row in df.iterrows():
#             text = row["text"]
#             true_label = row["label"]
#             
#             if true_label is None:
#                 continue
# 
#             model_pred = predict_text(text)["prediction"].strip().lower()
#             baseline_pred = baseline_predict(text).strip().lower()
# 
#             if model_pred == true_label:
#                 correct_model += 1
# 
#             if baseline_pred == true_label:
#                 correct_baseline += 1
# 
#             total += 1
# 
#     return {
#         "model_accuracy": round(correct_model / total, 4),
#         "baseline_accuracy": round(correct_baseline / total, 4)
#     }