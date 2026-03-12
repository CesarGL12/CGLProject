from transformers import pipeline, logging

# Suppress HF warnings
logging.set_verbosity_error()

class SentimentModel:
    """
    Centralized multilingual sentiment model.
    Loads the Hugging Face pipeline once.
    """

    _instance = None

    def __new__(cls, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

            cls._instance.classifier = pipeline(
                "sentiment-analysis",
                model=model_name,
                top_k=None,
                truncation=True,
                max_length=512
            )

        return cls._instance


    def predict(self, text):
        outputs = self.classifier(text)[0]

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