import ast
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = PROJECT_DIR / "raw_news_data.csv"
OUTPUT_PATH = PROJECT_DIR / "cleaned_news_data.csv"


def parse_source(value):
    if pd.isna(value):
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return ast.literal_eval(value)
    return None


df = pd.read_csv(SOURCE_PATH)

df["source"] = df["source"].apply(parse_source)
df["source_name"] = df["source"].apply(lambda item: item.get("name") if item else None)
df.drop(columns=["source"], inplace=True)

df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True)
df["author"] = df["author"].fillna("Unknown Author")
df["content"] = df["content"].fillna("")
df.drop_duplicates(subset=["title"], inplace=True)

df.to_csv(OUTPUT_PATH, index=False)

print(df.head())
