import os
from pathlib import Path

import pandas as pd
import requests


PROJECT_DIR = Path(__file__).resolve().parent
API_KEY = os.getenv("NEWS_API_KEY")


def fetch_news(query):
    if not API_KEY:
        raise RuntimeError("Set NEWS_API_KEY before running this script.")

    url = (
        f"https://newsapi.org/v2/everything?q={query}"
        f"&from=2026-04-18"
        f"&sortBy=publishedAt"
        f"&apiKey={API_KEY}"
    )

    response = requests.get(url, timeout=30)
    data = response.json()

    if response.status_code != 200:
        raise ValueError(f"Error {response.status_code}: {data}")

    return data.get("articles", [])


query = "Artificial Intelligence"
articles = fetch_news(query)

df = pd.DataFrame(articles)
output_path = PROJECT_DIR / "raw_news_data.csv"
df.to_csv(output_path, index=False)

print(df.head())
