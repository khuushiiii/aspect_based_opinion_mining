import streamlit as st
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import re
import subprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Aspex - Review Intelligence", layout="wide")

# ================= CUSTOM CSS =================
# ================= HEADER =================
st.markdown("""
<style>
.main-title {
    font-size: 50px !important;
    font-weight: 900 !important;
    text-align: center !important;
    color: #ffffff !important;
    margin-bottom: 10px !important;
}
.sub-title {
    font-size: 20px !important;
    text-align: center !important;
    color: #ffffff !important;
    margin-bottom: 30px !important;
}
</style>


""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = pickle.load(open("sentiment_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ================= LOAD ACCURACY =================
if os.path.exists("accuracy.txt"):
    acc = open("accuracy.txt").read()
else:
    acc = "Not Available"

# ================= LOAD FEEDBACK COUNT =================
def get_feedback_count():
    if os.path.exists("feedback.csv"):
        try:
            fb_df = pd.read_csv("feedback.csv")
            return len(fb_df)
        except:
            return 0
    return 0

# ================= NLP SETUP =================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# ================= SIDEBAR =================
st.sidebar.title("Model Info")
st.sidebar.metric("Accuracy", f"{acc}%")
st.sidebar.metric("Feedback Samples", get_feedback_count())

st.sidebar.markdown("---")
st.sidebar.write("**Aspects Detected:**")
st.sidebar.write("Delivery, Price, Quality, Performance, Packaging, Battery, Service, Food")

# ================= META LEARNING LOG IN SIDEBAR =================
st.sidebar.markdown("---")
st.sidebar.write("**Meta Learning History:**")
if os.path.exists("meta_learning_log.csv"):
    try:
        log_df = pd.read_csv("meta_learning_log.csv")
        if len(log_df) > 1:
            fig_log, ax_log = plt.subplots(figsize=(3, 2))
            ax_log.plot(
                range(1, len(log_df) + 1),
                log_df["accuracy"],
                marker='o',
                color='green',
                linewidth=2
            )
            ax_log.set_xlabel("Training Run", fontsize=7)
            ax_log.set_ylabel("Accuracy (%)", fontsize=7)
            ax_log.set_title("Accuracy Over Retrains", fontsize=8)
            ax_log.tick_params(labelsize=6)
            st.sidebar.pyplot(fig_log)
            plt.close(fig_log)
            st.sidebar.caption(f"Total retrains: {len(log_df)}")
        else:
            st.sidebar.info("Retrain at least twice to see progress chart.")
    except:
        st.sidebar.info("Log not readable yet.")
else:
    st.sidebar.info("No training history yet.")

# ================= HEADER =================
st.markdown("""
<p class="main-title">Aspex - Smart Review Analyzer</p>
<p class="sub-title">
Analyze product reviews with AI-powered sentiment, aspect detection, and meta learning.
</p>
""", unsafe_allow_html=True)

review = st.text_area("Enter your review:", height=150)

# ================= FAKE DETECTION =================
def detect_fake(text):
    if text.count("very") > 3 or len(text.split()) < 5:
        return "Possibly Fake"
    return "Likely Genuine"

# ================= ASPECT SENTIMENT WITH REASON =================
def aspect_sentiment(review):
    parts = re.split(r'\bbut\b|\bhowever\b|\band\b|\balthough\b|\bthough\b|\byet\b', review.lower())

    aspect_keywords = {
        "delivery": ["delivery", "shipping", "arrived", "late", "dispatch"],
        "price": ["price", "cost", "expensive", "cheap", "affordable", "worth"],
        "quality": ["quality", "material", "durable", "build"],
        "performance": ["performance", "slow", "fast", "lag", "smooth", "speed"],
        "packaging": ["packaging", "packed", "box", "wrap"],
        "battery": ["battery", "charge", "drain", "backup"],
        "service": ["service", "support", "staff", "response", "team"],
        "food": ["food", "taste", "delicious", "flavour", "dish"]
    }

    NEGATIVE_WORDS = ["slow", "bad", "worst", "terrible", "poor", "horrible",
                      "damaged", "broken", "waste", "useless", "delay", "late",
                      "disappoint", "lag", "defective", "awful", "rude", "weak",
                      "disappointing", "pathetic", "disgusting", "dirty", "cold"]
    POSITIVE_WORDS = ["fast", "good", "great", "excellent", "amazing", "awesome",
                      "love", "perfect", "best", "happy", "satisfied", "superb",
                      "fantastic", "quick", "smooth", "durable", "nice", "delicious",
                      "fresh", "hot", "tasty", "friendly", "helpful", "clean"]

    results = {}

    for part in parts:
        part = part.strip()

        matched_aspect = None
        for aspect, keywords in aspect_keywords.items():
            if any(kw in part for kw in keywords):
                matched_aspect = aspect
                break

        if not matched_aspect:
            continue

        neg_word = next((w for w in NEGATIVE_WORDS if w in part), None)
        pos_word = next((w for w in POSITIVE_WORDS if w in part), None)

        if neg_word and not pos_word:
            sentiment = "negative"
            reason = neg_word
        elif pos_word and not neg_word:
            sentiment = "positive"
            reason = pos_word
        elif pos_word and neg_word:
            sentiment = "negative"
            reason = neg_word
        else:
            vec = vectorizer.transform([preprocess(part)])
            sentiment = model.predict(vec)[0]
            reason = "model prediction"

        results[matched_aspect] = {"sentiment": sentiment, "reason": reason}

    return results

# ================= ANALYZE BUTTON =================
if st.button("Analyze Review"):

    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        aspect_results = aspect_sentiment(review)

        pos = sum(1 for v in aspect_results.values() if v["sentiment"] == "positive")
        neg = sum(1 for v in aspect_results.values() if v["sentiment"] == "negative")
        neu = sum(1 for v in aspect_results.values() if v["sentiment"] == "neutral")

        if neg > pos:
            overall = "negative"
        elif pos > neg:
            overall = "positive"
        else:
            vec = vectorizer.transform([preprocess(review)])
            overall = model.predict(vec)[0]

        st.session_state["last_review"] = review
        st.session_state["last_prediction"] = overall

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Positive", pos)
        with col2:
            st.metric("Negative", neg)
        with col3:
            st.metric("Neutral", neu)

        if pos > 0 or neg > 0 or neu > 0:
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.bar(
                ["Positive", "Negative", "Neutral"],
                [pos, neg, neu],
                color=["#4CAF50", "#f44336", "#FF9800"]
            )
            ax.set_title("Sentiment Distribution")
            ax.set_ylabel("Count")
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10
                    )
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No aspects detected in the review to plot.")

        st.markdown("### Overall Sentiment")
        if overall == "positive":
            st.success("Positive Review")
        elif overall == "negative":
            st.error("Negative Review")
        else:
            st.warning("Neutral Review")

        st.markdown("### Aspect Analysis")

        if aspect_results:
            for a, data in aspect_results.items():
                s = data["sentiment"]
                reason = data["reason"]
                if s == "positive":
                    st.success(f"**{a.capitalize()}** → Positive  *(because of '{reason}')*")
                elif s == "negative":
                    st.error(f"**{a.capitalize()}** → Negative  *(because of '{reason}')*")
                else:
                    st.warning(f"**{a.capitalize()}** → Neutral  *(because of '{reason}')*")
        else:
            st.info("No specific aspects detected. Try mentioning delivery, quality, price, service, etc.")

        st.markdown("### Fake Review Check")
        fake_result = detect_fake(review)
        if "Likely Genuine" in fake_result:
            st.success(fake_result)
        else:
            st.warning(fake_result)

# ================= META LEARNING =================
st.markdown("---")
st.subheader("Meta Learning - Improve the Model")
st.write("Help the model learn by correcting wrong predictions.")

if "last_review" not in st.session_state:
    st.info("Analyze a review above first, then give feedback here.")
else:
    st.write(f"**Review analyzed:** {st.session_state['last_review']}")
    st.write(f"**Model predicted:** `{st.session_state['last_prediction']}`")

    fb = st.radio(
        "Was this prediction correct?",
        ["Yes - prediction was correct", "No - prediction was wrong"],
        key="feedback_radio"
    )

    if fb == "No - prediction was wrong":
        correct = st.selectbox(
            "What is the correct sentiment?",
            ["positive", "negative", "neutral"],
            key="correct_sentiment"
        )

        if st.button("Submit Correction"):
            new_entry = pd.DataFrame({
                "cleaned_review": [preprocess(st.session_state["last_review"])],
                "sentiment": [correct]
            })
            if os.path.exists("feedback.csv"):
                new_entry.to_csv("feedback.csv", mode='a', header=False, index=False)
            else:
                new_entry.to_csv("feedback.csv", index=False)

            st.success(f"Correction saved! Model will learn from this in the next retrain.")
            st.info(f"Total feedback samples collected: {get_feedback_count()}")

    elif fb == "Yes - prediction was correct":
        if st.button("Submit Confirmation"):
            new_entry = pd.DataFrame({
                "cleaned_review": [preprocess(st.session_state["last_review"])],
                "sentiment": [st.session_state["last_prediction"]]
            })
            if os.path.exists("feedback.csv"):
                new_entry.to_csv("feedback.csv", mode='a', header=False, index=False)
            else:
                new_entry.to_csv("feedback.csv", index=False)

            st.success("Confirmation saved! This reinforces the model's learning.")

# ================= RETRAIN SECTION =================
st.markdown("---")
st.subheader("Retrain Model with Feedback")

feedback_count = get_feedback_count()

if feedback_count == 0:
    st.info("No feedback collected yet. Submit corrections above to enable retraining.")
else:
    st.write(f"**{feedback_count} feedback sample(s)** are ready to improve the model.")

    if st.button("Retrain Model Now"):
        with st.spinner("Retraining model with feedback data... Please wait."):
            try:
                result = subprocess.run(
                    ["python", "train_model.py"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode == 0:
                    st.cache_resource.clear()
                    st.success("Model retrained successfully with your feedback!")
                    st.code(result.stdout)

                    if os.path.exists("accuracy.txt"):
                        new_acc = open("accuracy.txt").read()
                        st.metric("New Model Accuracy", f"{new_acc}%")
                else:
                    st.error("Retraining failed. See error below:")
                    st.code(result.stderr)

            except subprocess.TimeoutExpired:
                st.error("Retraining timed out. Try again.")
            except Exception as e:
                st.error(f"Error during retraining: {e}")

# ================= FEEDBACK HISTORY =================
st.markdown("---")
st.subheader("Feedback History")

if os.path.exists("feedback.csv"):
    try:
        fb_df = pd.read_csv("feedback.csv")
        if len(fb_df) > 0:
            st.write(f"Total corrections/confirmations collected: **{len(fb_df)}**")

            fb_counts = fb_df["sentiment"].value_counts()
            col1, col2 = st.columns([1, 2])

            with col1:
                st.dataframe(fb_df.tail(10), use_container_width=True)

            with col2:
                fig2, ax2 = plt.subplots(figsize=(4, 3))
                ax2.pie(
                    fb_counts.values,
                    labels=fb_counts.index,
                    autopct='%1.1f%%',
                    colors=["#4CAF50", "#f44336", "#FF9800"]
                )
                ax2.set_title("Feedback Sentiment Breakdown")
                st.pyplot(fig2)
                plt.close(fig2)

            if st.button("Clear All Feedback"):
                os.remove("feedback.csv")
                st.warning("All feedback cleared. Retrain to restore base model.")
                st.rerun()
        else:
            st.info("Feedback file exists but is empty.")
    except Exception as e:
        st.error(f"Could not load feedback: {e}")
else:
    st.info("No feedback collected yet. Analyze a review and submit corrections above.")