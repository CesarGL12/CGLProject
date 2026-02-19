from datasets import load_dataset
import pandas as pd
import os
import re

os.makedirs("data", exist_ok=True)

dataset = load_dataset("clapAI/MultiLingualSentiment")
df = dataset["train"].to_pandas()

# Keep only EN, ES, FR
langs = ["en", "es", "fr"]
df = df[df["language"].isin(langs)]

def is_valid_en(text):
    # English: basic ASCII letters, numbers, punctuation
    return bool(re.fullmatch(r"[A-Za-z0-9\s.,!?;:'\"()\-]+", text))

def is_valid_es(text):
    # Spanish: allow ñ, á, é, í, ó, ú, ü
    return bool(re.fullmatch(r"[A-Za-z0-9\s.,!?;:'\"()\-áéíóúüñÁÉÍÓÚÜÑ]+", text))

def is_valid_fr(text):
    # French: allow ç, à, â, ê, î, ô, û, ë, ï, ü, é, è
    return bool(re.fullmatch(r"[A-Za-z0-9\s.,!?;:'\"()\-àâçéèêëîïôûùüÿœæÀÂÇÉÈÊËÎÏÔÛÙÜŸŒÆ]+", text))

def language_filter(row):
    text = row["text"]
    lang = row["language"]

    if lang == "en":
        return is_valid_en(text)
    elif lang == "es":
        return is_valid_es(text)
    elif lang == "fr":
        return is_valid_fr(text)
    return False

df = df[df.apply(language_filter, axis=1)]

MAX_SAMPLES = 20000  # choose 10000 or 20000

df = (
    df.groupby("language", group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), MAX_SAMPLES), random_state=42))
)

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