from explain_model import explain_text

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

    sentiment, confidence, important_words = explain_text(text, lang)

    print("\nPrediction:")
    print(f"{sentiment} ({confidence:.2f})")

    print("\nKey words:")

    if not important_words:
        print("No strong influential words detected.")
    else:
        for word, score in important_words[:5]:
            print(f"- {word}")

if __name__ == "__main__":
    main()
