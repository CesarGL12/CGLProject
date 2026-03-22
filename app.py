from flask import Flask, request, render_template
from src.prediction_service import predict_text, explain_text, baseline_predict

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    baseline = None
    important_words = None
    text = ""
    language = "en"
    error = None

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        language = request.form.get("language", "en")

        if not text:
            error = "Please enter some text before analyzing."
        elif len(text) > 500:
            error = "Text is too long (max 500 characters)."
        else:
            result = predict_text(text, language)
            explanation = explain_text(text, language)

            prediction = result["prediction"]
            confidence = result["confidence"]
            baseline = baseline_predict(text)
            important_words = explanation["important_words"]

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        baseline=baseline,
        important_words=important_words,
        text=text,
        language=language,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)