from logger import setup_logger
from prediction_service import MODEL_NAME
from datetime import datetime
import json
import os
import argparse
import pandas as pd
from prediction_service import predict_text, explain_text
from sklearn.metrics import accuracy_score


def run_predict(args, logger):

    result = predict_text(args.text, args.lang)

    logger.info(f"Input text: {args.text}")
    logger.info(f"Prediction: {result['prediction']}")
    logger.info(f"Confidence: {result['confidence']}")

    os.makedirs("results", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/predict_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Results saved to {output_file}")

    print("\nPrediction Result")
    print("------------------")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")


def run_explain(args, logger):

    result = explain_text(args.text, args.lang)

    logger.info(f"Input text: {args.text}")
    logger.info(f"Prediction: {result['prediction']}")

    os.makedirs("results", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/explain_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Results saved to {output_file}")


def run_evaluate(args, logger):

    df = pd.read_csv(args.file)

    predictions = []
    true_labels = []

    for _, row in df.iterrows():

        text = row["text"]
        true_label = row["label"]

        result = predict_text(text)

        label_map = {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2
        }

        pred_label = label_map[result["prediction"]]

        predictions.append(pred_label)
        true_labels.append(true_label)

    acc = accuracy_score(true_labels, predictions)

    print("\nEvaluation Result")
    print("------------------")
    print(f"Samples evaluated: {len(df)}")
    print(f"Accuracy: {acc:.3f}")

    logger.info(f"Evaluating file: {args.file}")
    logger.info(f"Accuracy: {acc}")

    os.makedirs("results", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/evaluation_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump({"accuracy": acc, "samples": len(df)}, f, indent=2)

    logger.info(f"Results saved to {output_file}")


def main():

    parser = argparse.ArgumentParser(
        description="Multilingual Sentiment Analysis CLI"
    )

    subparsers = parser.add_subparsers(dest="mode")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--text", required=True)
    predict_parser.add_argument("--lang", default="en")

    explain_parser = subparsers.add_parser("explain")
    explain_parser.add_argument("--text", required=True)
    explain_parser.add_argument("--lang", default="en")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--file", required=True)

    args = parser.parse_args()

    logger, log_file = setup_logger(args.mode, MODEL_NAME)

    if args.mode == "predict":
        run_predict(args, logger)

    elif args.mode == "explain":
        run_explain(args, logger)

    elif args.mode == "evaluate":
        run_evaluate(args, logger)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
