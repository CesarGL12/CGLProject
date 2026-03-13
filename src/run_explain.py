from prediction_service import explain_text


def main():

    print("\nMultilingual Sentiment Explanation System")
    print("-----------------------------------------")

    text = input("Enter text: ").strip()

    if not text:
        print("No input provided.")
        return

    lang = input("Language (en/es/fr) [default=en]: ").strip().lower()

    if lang == "":
        lang = "en"

    result = explain_text(text, lang)

    print("\nPrediction:")
    print(f"{result['prediction']} ({result['confidence']:.2f})")

    print("\nKey words:")

    if not result["important_words"]:
        print("No strong influential words detected.")
    else:
        for item in result["important_words"][:5]:
            print(f"- {item['word']}")


if __name__ == "__main__":
    main()