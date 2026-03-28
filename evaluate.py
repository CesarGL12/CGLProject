import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import classification_report

from src.prediction_service import predict_text, baseline_predict

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

def map_model_output(prediction: str):
    prediction = prediction.lower()

    if "1" in prediction or "2" in prediction:
        return "negative"
    elif "3" in prediction:
        return "neutral"
    elif "4" in prediction or "5" in prediction:
        return "positive"
    else:
        return None

def evaluate_dataset(file_path):
    from collections import Counter
    model_counter = Counter()
    true_counter = Counter()
    
    y_true = []
    y_pred = []
    
    df = pd.read_csv(file_path)

    total = 0
    correct_model = 0
    correct_baseline = 0
    
    def map_label(label):
        if label == 0:
            return "negative"
        elif label == 1:
            return "neutral"
        elif label == 2:
            return "positive"
        else:
            return None

    for _, row in df.iterrows():
        text = row["text"]
        
        true_label = map_label(row["label"])
        
        if true_label is None:
            continue
            
        true_label = true_label.lower()

        #raw_pred = predict_text(text)["prediction"]
        #model_pred = map_model_output(raw_pred)
        model_pred = predict_text(text)["prediction"].strip().lower()
        baseline_pred = baseline_predict(text).strip().lower()

        y_true.append(true_label)
        y_pred.append(model_pred)
        
        if model_pred == true_label:
            correct_model += 1

        if baseline_pred == true_label:
            correct_baseline += 1

        total += 1
        
        model_counter[model_pred] += 1
        true_counter[true_label] += 1
        
        #print(model_pred, true_label)
        #print(raw_pred, "→", model_pred, "| true:", true_label)

    print("Model distribution:", model_counter)
    print("True distribution:", true_counter)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    return {
        "total": total,
        "model_accuracy": round(correct_model / total, 4),
        "baseline_accuracy": round(correct_baseline / total, 4)
    }


def main():
    datasets = {
        "english": "data/sample/en_sample.csv",
        "spanish": "data/sample/es_sample.csv",
        "french": "data/sample/fr_sample.csv"
    }

    results = {}

    for name, path in datasets.items():
        print(f"Evaluating {name} dataset...")
        results[name] = evaluate_dataset(path)

    # Timestamp for file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/evaluation_{timestamp}.json"

    # Save results
    pd.Series(results).to_json(output_file, indent=4)

    print("\nEvaluation complete!")
    print(results)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
