
import pandas as pd
import pickle
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ================= LOAD BASE DATA =================
df = pd.read_csv("data/cleaned_review.csv")

# ================= META LEARNING =================
# Load feedback data if it exists and merge with base training data
feedback_count = 0
if os.path.exists("feedback.csv"):
    feedback_df = pd.read_csv("feedback.csv")
    feedback_df = feedback_df.dropna(subset=["cleaned_review", "sentiment"])

    if len(feedback_df) > 0:
        feedback_count = len(feedback_df)

        # Give feedback samples extra weight by duplicating them
        # This ensures the model learns more from human corrections
        weighted_feedback = pd.concat([feedback_df] * 3, ignore_index=True)
        df = pd.concat([df, weighted_feedback], ignore_index=True)
        print(f"[Meta Learning] Included {feedback_count} feedback samples (weighted x3)")
    else:
        print("ℹ️  Feedback file found but empty. Training on base data only.")
else:
    print("ℹ️  No feedback data found. Training on base data only.")

# ================= FEATURE EXTRACTION =================
X = df["cleaned_review"]
y = df["sentiment"]

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

# ================= TRAIN/TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# ================= MODEL TRAINING =================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {acc * 100:.2f}%")

# ================= SAVE MODEL & VECTORIZER =================
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# ================= SAVE ACCURACY =================
with open("accuracy.txt", "w") as f:
    f.write(str(round(acc * 100, 2)))

# ================= SAVE META LEARNING LOG =================
# Log each training run so you can track how the model improves over time
import datetime

log_entry = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "base_samples": len(pd.read_csv("data/cleaned_review.csv")),
    "feedback_samples": feedback_count,
    "total_samples_trained": len(df),
    "accuracy": round(acc * 100, 2)
}

log_df = pd.DataFrame([log_entry])

if os.path.exists("meta_learning_log.csv"):
    log_df.to_csv("meta_learning_log.csv", mode='a', header=False, index=False)
else:
    log_df.to_csv("meta_learning_log.csv", index=False)

print(f"📝 Training log saved to meta_learning_log.csv")
print("✅ Model saved successfully!")