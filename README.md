Explainable Aspect-Based Opinion Mining

An advanced NLP project that performs Aspect-Based Sentiment Analysis (ABSA) on e-commerce reviews using Machine Learning and the BERT Transformer model.

The system identifies product aspects, predicts aspect-wise sentiment, detects fake reviews, summarizes feedback, and provides explainable outputs through an interactive dashboard.

Unique Features

✅ Aspect-Based Sentiment Analysis (ABSA)
✅ BERT-based contextual sentiment classification
✅ Automatic aspect extraction using NLP & POS tagging
✅ Fake review detection
✅ Review summarization
✅ Explainable aspect-wise predictions
✅ Mixed sentiment handling in a single review
✅ Confidence score visualization
✅ Trend analysis & word cloud generation
✅ Meta-learning feedback loop for continuous improvement
✅ Interactive Streamlit/Gradio interface

---

Input
```text
Delivery was late but product quality is excellent.
```

\Output

```text
Delivery  → Negative
Quality   → Positive
```

Tech Stack

* Python
* NLTK
* spaCy
* Scikit-learn
* TF-IDF
* BERT-base-uncased
* PyTorch
* HuggingFace Transformers
* Streamlit / Gradio

Performance

| Model                     | Accuracy |
| ------------------------- | -------- |
| Naive Bayes               | 72%      |
| SVM                       | 76%      |
| LSTM                      | 82%      |
| **BERT (Proposed Model)** | **86%**  |

---
Run Project

```bash
git clone https://github.com/your-username/nlp_project.git
cd nlp_project
pip install -r requirements.txt
python app.py
```

Authors

* Khushi Sharma – 23BCE0164
* Vihan Wadhawan – 23BCE0023

Course: Natural Language Processing
