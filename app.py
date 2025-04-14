import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords


# Download stopwords if not already
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text cleaning function
stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\bnot\s+(\w+)", r"not_\1", text)
    words = text.split()
    return " ".join([w for w in words if w not in stop_words])

# Streamlit App UI
st.set_page_config(page_title="Hotel Review Sentiment Classifier", layout="centered")

st.title("🏨 Hotel Review Sentiment Classifier")
st.caption("Model: Logistic Regression + TF-IDF (1–3 grams) + SMOTE | Threshold: 0.6")

review = st.text_area("📝 Enter review text:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])
        proba = model.predict_proba(vectorized)[0][1]
        proba_neg = model.predict_proba(vectorized)[0][0]
        prediction = 1 if proba > 0.6 else 0

        label = "Positive 😊" if prediction == 1 else "Negative 😠"
        st.markdown(f"**Sentiment:** {label}")
        if prediction == 1:
            st.markdown(f"**Probability (Positive):** {proba:.2f}")
        else:
            st.markdown(f"**Probability (Negative):** {proba_neg:.2f}")