# DataScience-and-MachineLearning

This repository is a home for data science and machine learning projects.

## Layout

- `Projects/` stores individual project folders.
- `Projects/NewsEDAProject/` is the first project in this repo.

## NewsEDAProject

This project fetches news articles, cleans the collected data, and creates a few exploratory charts.

### Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your NewsAPI key:

```bash
set NEWS_API_KEY=your_key_here
```

### Run

```bash
python Projects/NewsEDAProject/NewsAPI2.py
python Projects/NewsEDAProject/clean_news_data.py
python Projects/NewsEDAProject/analyze_news.py
```

### Notes

- Generated CSV and PNG outputs are ignored by git.
- Keep secrets out of source files and use environment variables for API keys.
