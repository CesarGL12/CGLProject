from flask import Flask, request, render_template
from src.prediction_service import predict_text, explain_text, baseline_predict

app = Flask(__name__)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     confidence = None
#     baseline = None
#     important_words = None
#     text = ""
#     language = "en"
#     error = None
# 
#     if request.method == "POST":
#         text = request.form.get("text", "").strip()
#         language = request.form.get("language", "en")
# 
#         if not text:
#             error = "Please enter some text before analyzing."
#         elif len(text) > 500:
#             error = "Text is too long (max 500 characters)."
#         else:
#             result = predict_text(text, language)
#             explanation = explain_text(text, language)
# 
#             prediction = result["prediction"]
#             confidence = result["confidence"]
#             baseline = baseline_predict(text)
# 
#             if explanation["prediction"] == prediction:
#                 important_words = explanation["important_words"]
#             else:
#                 important_words = []
# 
#     return render_template(
#         "index.html",
#         prediction=prediction,
#         confidence=confidence,
#         baseline=baseline,
#         important_words=important_words,
#         text=text,
#         language=language,
#         error=error
#     )
# 
# if __name__ == "__main__":
#     app.run(debug=True)

from evaluate import evaluate_dataset  # add this

MODELS = {
    "bert": predict_text,
    "baseline": baseline_predict
}

DATASETS = {
    "english": "data/sample/en_sample.csv",
    "spanish": "data/sample/es_sample.csv",
    "french": "data/sample/fr_sample.csv"
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    baseline = None
    important_words = None
    text = ""
    #language = "en"
    error = None

    eval_result = None  # NEW

    if request.method == "POST":

        # 🔹 CASE 1: Text analysis (your existing feature)
        if "analyze_text" in request.form:
            text = request.form.get("text", "").strip()
            #language = request.form.get("language", "en")

            if not text:
                error = "Please enter some text before analyzing."
            elif len(text) > 500:
                error = "Text is too long (max 500 characters)."
            else:
                #result = predict_text(text, language)
                #explanation = explain_text(text, language)
                result = predict_text(text)
                explanation = explain_text(text)

                prediction = result["prediction"]
                confidence = result["confidence"]
                baseline = baseline_predict(text)

                if explanation["prediction"] == prediction:
                    important_words = explanation["important_words"]
                else:
                    important_words = []

        # 🔹 CASE 2: Dataset evaluation (NEW)
        elif "run_evaluation" in request.form:
            dataset_name = request.form.get("dataset")
            path = DATASETS.get(dataset_name)

            if path:
                #eval_result = evaluate_dataset(path, MODELS)
                #eval_result["dataset"] = dataset_name
                try:
                    eval_result = evaluate_dataset(path, MODELS)
                    eval_result["dataset"] = dataset_name
                except Exception as e:
                    error = f"Evaluation failed: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        baseline=baseline,
        important_words=important_words,
        text=text,
        #language=language,
        error=error,
        eval_result=eval_result,   # NEW
        datasets=DATASETS.keys()  # NEW
    )

if __name__ == "__main__":
    print("Server running at: http://127.0.0.1:5000")
    app.run(debug=True)