from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

dataset = load_dataset("clapAI/MultiLingualSentiment")
df = dataset["train"].to_pandas()

# Keep only EN, ES, FR
langs = ["en", "es", "fr"]
df = df[df["language"].isin(langs)]

# Convert textual labels to numeric labels
label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

df["label"] = df["label"].map(label_map)

# Save separate files
for lang in langs:
    df_lang = (
        df[df["language"] == lang]
        [["text", "label"]]
        .dropna()
        .reset_index(drop=True)
    )

    output_path = f"data/{lang}.csv"
    df_lang.to_csv(output_path, index=False)
    print(f"Saved {len(df_lang)} rows to {output_path}")