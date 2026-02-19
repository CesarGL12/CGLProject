from datasets import load_dataset
import pandas as pd
import os
import re
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42  # reproducibility

os.makedirs("data/sample", exist_ok=True)

# -------------------------
# 1. Load dataset
# -------------------------
print("Loading dataset...")
dataset = load_dataset("clapAI/MultiLingualSentiment")
df = dataset["train"].to_pandas()

# -------------------------
# 2. Keep only target languages
# -------------------------
langs = ["en", "es", "fr"]
df = df[df["language"].isin(langs)].copy()

# -------------------------
# 3. Label mapping
# -------------------------
label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

df["label"] = df["label"].map(label_map)
df = df.dropna(subset=["label"])

# -------------------------
# 4. Strict validation setup
# -------------------------

# Allowed characters (Latin alphabets + accents + common punctuation)
allowed_pattern = re.compile(
    r"^[a-zA-ZÀ-ÖØ-öø-ÿ0-9\s.,!?;:'\"()\-\n]+$"
)

def is_valid_text(text, expected_lang):
    if not isinstance(text, str):
        return False

    text = text.strip()

    # Minimum length (improves langdetect reliability)
    if len(text) < 15:
        return False

    # Block corrupted characters
    bad_chars = ["≈", "ƒ", "√", "�"]
    if any(char in text for char in bad_chars):
        return False

    # Block emojis / weird unicode
    if not allowed_pattern.match(text):
        return False

    # Language verification
    try:
        if detect(text) != expected_lang:
            return False
    except:
        return False

    return True

# -------------------------
# 5. Optimized balanced sampling
# -------------------------
SAMPLE_PER_CLASS = 200
BUFFER_MULTIPLIER = 5  # sample extra in case some fail cleaning

for lang in langs:
    print(f"\nProcessing {lang.upper()}...")

    df_lang = df[df["language"] == lang]

    final_samples = []

    for label in [0, 1, 2]:
        df_class = df_lang[df_lang["label"] == label]

        if len(df_class) < SAMPLE_PER_CLASS:
            raise ValueError(
                f"Not enough raw samples for {lang} label {label}"
            )

        # Pre-sample a buffer pool
        buffer_size = min(len(df_class), SAMPLE_PER_CLASS * BUFFER_MULTIPLIER)

        df_buffer = df_class.sample(
            n=buffer_size,
            random_state=42
        )

        # Apply strict cleaning only on buffer
        df_clean = df_buffer[
            df_buffer["text"].apply(lambda x: is_valid_text(x, lang))
        ]

        if len(df_clean) < SAMPLE_PER_CLASS:
            raise ValueError(
                f"Not enough CLEAN samples for {lang} label {label}. "
                f"Found {len(df_clean)}"
            )

        df_sample = df_clean.sample(
            n=SAMPLE_PER_CLASS,
            random_state=42
        )

        final_samples.append(df_sample[["text", "label"]])

    df_final = pd.concat(final_samples).reset_index(drop=True)

    output_path = f"data/sample/{lang}_sample.csv"
    df_final.to_csv(output_path, index=False)

    print("Distribution:")
    print(df_final["label"].value_counts())
    print("Saved to", output_path)
    print("-" * 40)

print("\nAll languages processed successfully.")

