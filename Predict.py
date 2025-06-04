import streamlit as st
import pandas as pd
import joblib
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
# Load the trained model and vectorizer
rf_model = joblib.load("random_forest_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Function to preprocess text
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)


# Streamlit app title
st.title("Crime Occurrence Prediction from Tweets")

# Text input for user to enter tweet
tweet_input = st.text_area("Enter the tweet text here:")

# Button to make prediction
if st.button("Predict"):
    if tweet_input:
        # Preprocess the input text
        processed_text = preprocess_text(tweet_input)
        # Vectorize the input text
        input_tfidf = vectorizer.transform([processed_text])
        # Make prediction
        prediction = rf_model.predict(input_tfidf)
        prediction_proba = rf_model.predict_proba(input_tfidf)

        # Display the prediction result
        if prediction[0] == 1:
            st.success("This tweet is likely related to a crime occurrence.")
        else:
            st.success("This tweet is likely not related to a crime occurrence.")

        # Display prediction probabilities
        st.write(f"Prediction Probability: {prediction_proba[0]}")
            
    else:
        st.warning("Please enter a tweet to make a prediction.")