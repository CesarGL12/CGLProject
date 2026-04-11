import json
import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import classification_report

from src.prediction_service import predict_text, baseline_predict

from logger import setup_logger
logger = setup_logger()

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

def evaluate_dataset(file_path, models):
    from collections import Counter

    df = pd.read_csv(file_path)

    total = 0

    # Store per-model stats
    correct_counts = {model_name: 0 for model_name in models}
    predictions = {model_name: [] for model_name in models}

    y_true = []

    def map_label(label):
        return ["negative", "neutral", "positive"][label] if label in [0,1,2] else None

    if "text" not in df.columns or "label" not in df.columns:
        logger.error("Dataset missing required columns")
        return {"error": "Invalid dataset format"}
    
    for _, row in df.iterrows():
        text = row["text"]
        true_label = map_label(row["label"])

        if true_label is None:
            continue

        true_label = true_label.lower()
        y_true.append(true_label)

        for model_name, model_func in models.items():
            #pred = model_func(text)
            try:
                pred = model_func(text)

                if isinstance(pred, dict):
                    pred = pred.get("prediction", "")

                if not isinstance(pred, str):
                    pred = ""

                pred = pred.strip().lower()

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                pred = ""

            # Handle dict vs string output
            if isinstance(pred, dict):
                pred = pred.get("prediction", "")

            pred = pred.strip().lower()

            predictions[model_name].append(pred)

            if pred == true_label:
                correct_counts[model_name] += 1

        total += 1

    # Build result
    result = {
        "num_samples": total
    }

    for model_name in models:
        acc = correct_counts[model_name] / total if total > 0 else 0
        result[f"{model_name}_accuracy"] = round(acc, 4)

    return result
    
# def evaluate_dataset(file_path):
#     from collections import Counter
#     model_counter = Counter()
#     true_counter = Counter()
#     
#     y_true = []
#     y_pred = []
#     
#     df = pd.read_csv(file_path)
# 
#     total = 0
#     correct_model = 0
#     correct_baseline = 0
#     
#     def map_label(label):
#         if label == 0:
#             return "negative"
#         elif label == 1:
#             return "neutral"
#         elif label == 2:
#             return "positive"
#         else:
#             return None
# 
#     for _, row in df.iterrows():
#         text = row["text"]
#         
#         true_label = map_label(row["label"])
#         
#         if true_label is None:
#             continue
#             
#         true_label = true_label.lower()
# 
#         #raw_pred = predict_text(text)["prediction"]
#         #model_pred = map_model_output(raw_pred)
#         model_pred = predict_text(text)["prediction"].strip().lower()
#         baseline_pred = baseline_predict(text).strip().lower()
# 
#         y_true.append(true_label)
#         y_pred.append(model_pred)
#         
#         if model_pred == true_label:
#             correct_model += 1
# 
#         if baseline_pred == true_label:
#             correct_baseline += 1
# 
#         total += 1
#         
#         model_counter[model_pred] += 1
#         true_counter[true_label] += 1
#         
#         #print(model_pred, true_label)
#         #print(raw_pred, "→", model_pred, "| true:", true_label)
# 
#     print("Model distribution:", model_counter)
#     print("True distribution:", true_counter)
#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred))
# 
#     return {
#         "dataset": os.path.basename(file_path),
#         "model": "bert-multilingual",
#         "num_samples": total,
#         "model_accuracy": round(correct_model / total, 4),
#         "baseline_accuracy": round(correct_baseline / total, 4)
#     }

def save_results(results_list):
    os.makedirs("results", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON file
    json_path = f"results/evaluation_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results_list, f, indent=4)

    # CSV file
    csv_path = f"results/evaluation_{timestamp}.csv"
    df = pd.DataFrame(results_list)
    df = df.sort_values(by="bert_accuracy", ascending=False)
    df.to_csv(csv_path, index=False)

    print(f"Results saved to:\n- {json_path}\n- {csv_path}")

def main():
    MODELS = {
        "bert": predict_text,
        "baseline": baseline_predict
    }

    datasets = {
        "english": "data/sample/en_sample.csv",
        "spanish": "data/sample/es_sample.csv",
        "french": "data/sample/fr_sample.csv"
    }

    results = []

    for name, path in datasets.items():
        #print(f"Evaluating {name} dataset...")
        logger.info(f"Evaluating dataset: {name}")
        
        result = evaluate_dataset(path, MODELS)
        result["dataset"] = name
        
        results.append(result)

    print("\nEvaluation complete!")
    #print(results)
    logger.info(f"Results: {results}")

    save_results(results)

# def main():
#     MODELS = {
#         "bert": predict_text,
#         "baseline": baseline_predict
#     }
# 
#     datasets = {
#         "english": "data/sample/en_sample.csv",
#         "spanish": "data/sample/es_sample.csv",
#         "french": "data/sample/fr_sample.csv"
#     }
# 
#     results = []
# 
#     for name, path in datasets.items():
#         print(f"Evaluating {name} dataset...")
#         
#         result = evaluate_dataset(path)
#         result["dataset"] = name  # cleaner label
#         
#         results.append(result)
# 
#     print("\nEvaluation complete!")
#     print(results)
# 
#     save_results(results)

if __name__ == "__main__":
    main()