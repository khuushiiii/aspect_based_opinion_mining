
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
try:
    df = pd.read_csv("data/review.csv")
except:
    df = pd.read_csv("review.csv")

df = df.dropna(subset=["review", "rating"])

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

df["cleaned_review"] = df["review"].apply(clean_text)

def get_sentiment(r):
    if r >= 4:
        return "positive"
    elif r == 3:
        return "neutral"
    return "negative"

df["sentiment"] = df["rating"].apply(get_sentiment)

df.to_csv("data/cleaned_review.csv", index=False)
print("✅ Cleaned data saved")