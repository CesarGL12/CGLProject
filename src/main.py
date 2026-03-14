import argparse
import pandas as pd
from prediction_service import predict_text, explain_text
from sklearn.metrics import accuracy_score


def run_predict(args):
    result = predict_text(args.text, args.lang)

    print("\nPrediction Result")
    print("------------------")
    print(f"Text: {result['text']}")
    print(f"Language: {result['language']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")


def run_explain(args):
    result = explain_text(args.text, args.lang)

    print("\nExplanation Result")
    print("-------------------")
    print(f"Text: {result['text']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")

    print("\nImportant Words:")

    if not result["important_words"]:
        print("No strong influential words detected.")
    else:
        for item in result["important_words"][:5]:
            print(f"- {item['word']} (importance {item['importance']:.3f})")


def run_evaluate(args):

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


def main():

    parser = argparse.ArgumentParser(
        description="Multilingual Sentiment Analysis CLI"
    )

    subparsers = parser.add_subparsers(dest="mode")

    # Predict mode
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--text", required=True)
    predict_parser.add_argument("--lang", default="en")

    # Explain mode
    explain_parser = subparsers.add_parser("explain")
    explain_parser.add_argument("--text", required=True)
    explain_parser.add_argument("--lang", default="en")

    # Evaluate mode
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--file", required=True)

    args = parser.parse_args()

    if args.mode == "predict":
        run_predict(args)

    elif args.mode == "explain":
        run_explain(args)

    elif args.mode == "evaluate":
        run_evaluate(args)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
