from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_PATH = PROJECT_DIR / "cleaned_news_data.csv"


df = pd.read_csv(INPUT_PATH)
df["publishedAt"] = pd.to_datetime(df["publishedAt"], utc=True)
df["publishedAt"] = df["publishedAt"].dt.tz_convert("Asia/Kolkata")

df["hour"] = df["publishedAt"].dt.hour
plt.figure()
sns.countplot(x="hour", data=df)
plt.title("Number of Articles by Hour (IST)")
plt.xlabel("Hour of Day")
plt.ylabel("Article Count")
plt.savefig(PROJECT_DIR / "articles_by_hour.png")
plt.close()

df["source_name"] = df["source_name"].replace({"Times of India": "The Times of India"})
top_sources = df["source_name"].value_counts().head(10)

plt.figure()
sns.barplot(x=top_sources.values, y=top_sources.index, palette="viridis")
plt.title("Top 10 News Sources Covering AI")
plt.xlabel("Number of Articles")
plt.ylabel("Source")
plt.savefig(PROJECT_DIR / "top_sources.png")
plt.close()

df["content_length"] = df["content"].apply(len)
top5 = df["source_name"].value_counts().head(5).index
df_top5 = df[df["source_name"].isin(top5)]

plt.figure()
sns.boxplot(x="source_name", y="content_length", data=df_top5)
plt.xticks(rotation=30)
plt.title("Content Length Distribution by Top 5 Sources")
plt.xlabel("Source")
plt.ylabel("Content Length")
plt.savefig(PROJECT_DIR / "content_length_boxplot.png")
plt.close()

print("All plots saved successfully.")
