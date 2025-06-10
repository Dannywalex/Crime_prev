import streamlit as st
import pandas as pd
import numpy as np  
import joblib
import string
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# Download NLTK once
nltk.download('punkt')
nltk.download('stopwords')

# Load the entire trained pipeline (includes hashing, tfidf, and RF)
pipeline = joblib.load("rf_hashing_pipeline.joblib")

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return " ".join(tokens)

st.title("Crime Tweet Classifier")

tweet_input = st.text_area("Enter tweet text:")

if st.button("Predict"):
    if tweet_input.strip():
        processed = preprocess_text(tweet_input)

        # Predict using the full pipeline
        pred_label = pipeline.predict([processed])[0]
        proba = pipeline.predict_proba([processed])[0]
        classes = pipeline.classes_

        

        # ... after predicting
        st.subheader("Full Class Probabilities:")
        fig, ax = plt.subplots()
        ax.barh(classes, proba, color='skyblue')
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Predicted Probabilities by Class")
        st.pyplot(fig)


        st.markdown(f"**Predicted Category:** `{pred_label}`")

        # Show top 2 class probabilities
        st.subheader("Top-2 Class Probabilities:")
        top2 = np.argsort(proba)[-2:][::-1]
        for idx in top2:
            st.write(f"{classes[idx]}: {proba[idx]:.3f}")
    else:
        st.warning("Please enter some tweet text.")
