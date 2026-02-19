from datasets import load_dataset
import pandas as pd

dataset = load_dataset("clapAI/MultiLingualSentiment")
df = dataset["train"].to_pandas()

print("Overall label distribution:")
print(df["label"].value_counts())

print("\nLabel distribution per language:")
print(df.groupby("language")["label"].value_counts())

