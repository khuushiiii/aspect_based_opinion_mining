
import pickle
import os

base_path = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(base_path, "sentiment_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(base_path, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("❌ Model files not found. Please run train_model.py first.")
    exit()

NEGATIVE_WORDS = [
    "slow", "bad", "worst", "terrible", "poor", "horrible", "damaged",
    "broken", "waste", "useless", "fake", "delay", "late", "disappoint",
    "lag", "issue", "problem", "defective", "return", "refund", "awful"
]

POSITIVE_WORDS = [
    "fast", "good", "great", "excellent", "amazing", "awesome", "love",
    "perfect", "best", "happy", "satisfied", "recommend", "superb",
    "fantastic", "quick", "smooth", "durable", "worth", "nice", "beautiful"
]

review = input("Enter review: ")

sentence_lower = review.lower()
neg_found = any(word in sentence_lower for word in NEGATIVE_WORDS)
pos_found = any(word in sentence_lower for word in POSITIVE_WORDS)

if neg_found and not pos_found:
    print("Sentiment: negative")
elif pos_found and not neg_found:
    print("Sentiment: positive")
else:
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)
    print("Sentiment:", prediction[0])