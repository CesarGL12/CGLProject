import os
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Load pretrained multilingual sentiment model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
classifier = pipeline("sentiment-analysis", model=model_name)

def map_sentiment(result):
    label = result["label"]   # e.g., "4 stars"
    stars = int(label[0])

    if stars <= 2:
        return 0  # Negative
    elif stars == 3:
        return 1  # Neutral
    else:
        return 2  # Positive


languages = ["en", "es", "fr"]
results = {}

for lang in languages:
    df = pd.read_csv(f"data/sample/{lang}_sample.csv")

    # Optional: use smaller sample for faster evaluation
    #df = df.sample(n=300, random_state=42) if len(df) > 300 else df

    predictions = []
    true_labels = []

    correct_example = None
    wrong_example = None

    for _, row in df.iterrows():
        text = row["text"]
        true_label = row["label"]

        raw = classifier(text, truncation=True, max_length=512)[0]
        pred_label = map_sentiment(raw)

        predictions.append(pred_label)
        true_labels.append(true_label)

        # Save one correct example
        if pred_label == true_label and correct_example is None:
            correct_example = (text, true_label)

        # Save one wrong example
        if pred_label != true_label and wrong_example is None:
            wrong_example = (text, true_label, pred_label)

    accuracy = accuracy_score(true_labels, predictions)

    results[lang] = {
        "accuracy": accuracy,
        "correct": correct_example,
        "wrong": wrong_example
    }

    print(f"{lang.upper()} accuracy: {accuracy:.3f}")

# Write Markdown report
with open("results/baseline_results.md", "w") as f:
    f.write("# Baseline Model Evaluation Results\n\n")

    for lang in languages:
        res = results[lang]

        f.write(f"## Language: {lang.upper()}\n")
        f.write(f"- **Accuracy:** {res['accuracy']:.3f}\n\n")

        if res["correct"]:
            f.write("### Example Correct Prediction\n")
            f.write(f"- Text: \"{res['correct'][0]}\"\n")
            f.write(f"- Label: {res['correct'][1]}\n\n")

        if res["wrong"]:
            f.write("### Example Incorrect Prediction\n")
            f.write(f"- Text: \"{res['wrong'][0]}\"\n")
            f.write(f"- True Label: {res['wrong'][1]}\n")
            f.write(f"- Predicted Label: {res['wrong'][2]}\n\n")

print("\nResults written to results/baseline_results.md")


