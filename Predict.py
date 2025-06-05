import streamlit as st
import pandas as pd
import numpy as np  
import joblib
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# Download NLTK once
nltk.download('punkt')
nltk.download('stopwords')

# Load the saved hashing objects + RF
hash_vectorizer   = joblib.load("hash_vectorizer.joblib")
tfidf_transformer = joblib.load("tfidf_transformer.joblib")
rf_hashing        = joblib.load("rf_hashing_model.joblib")

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return " ".join(tokens)

st.title("Crime Occurrence Prediction from Tweets")

tweet_input = st.text_area("Enter tweet text:")

if st.button("Predict"):
    if tweet_input.strip():
        processed = preprocess_text(tweet_input)

        # 1) Hash → 2) TF-IDF transform
        X_h = hash_vectorizer.transform([processed])
        X_tfidf = tfidf_transformer.transform(X_h)

        # Because hashing + tfidf never yields a truly “zero” vector (unless the input was empty),
        # there’s no need to check for nnz == 0. Every token (even a new one) falls into some hash bucket.
        pred_label = rf_hashing.predict(X_tfidf)[0]
        proba      = rf_hashing.predict_proba(X_tfidf)[0]
        classes    = rf_hashing.classes_

        st.markdown(f"**Predicted Category:** `{pred_label}`")
        # show top-2 probabilities
        top2 = np.argsort(proba)[-2:][::-1]
        for idx in top2:
            st.write(f"{classes[idx]}: {proba[idx]:.3f}")
    else:
        st.warning("Please enter some tweet text.")