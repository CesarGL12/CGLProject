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

    if request.method == "POST":
        text = request.form["text"]

        result = predict_text(text)
        explanation = explain_text(text)

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
        text=text
    )

if __name__ == "__main__":
    app.run(debug=True)