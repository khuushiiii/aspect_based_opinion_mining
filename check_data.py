
import pandas as pd

try:
    df = pd.read_csv("data/review.csv")
except FileNotFoundError:
    df = pd.read_csv("review.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSentiment Distribution:")
if "rating" in df.columns:
    def get_sentiment(rating):
        if rating >= 4:
            return "positive"
        elif rating == 3:
            return "neutral"
        else:
            return "negative"
    df["sentiment"] = df["rating"].apply(get_sentiment)
    print(df["sentiment"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())