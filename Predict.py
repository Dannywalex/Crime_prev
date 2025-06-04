import streamlit as st
import pandas as pd
import joblib
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from geopy.exc import GeocoderServiceError

# Load the trained model and vectorizer
rf_model = joblib.load("random_forest_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Function to preprocess text
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Function to get coordinates from location
def get_coordinates(location):
    geolocator = Nominatim(user_agent="geoapiExercises")
    try:
        loc = geolocator.geocode(location, timeout=10)
        if loc:
            return loc.latitude, loc.longitude
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        st.warning(f"Geocoding failed: {e}")
    except Exception as e:
        st.warning(f"An unexpected error occurred during geocoding: {e}")
    return None

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

        # Extract and show location
        location = None
        if any(word.isdigit() for word in processed_text.split()):
            # Fetch coordinates based on a simple location extraction logic
            location = ' '.join([word for word in processed_text.split() if word.isalnum() and not word.isdigit()])

        if location:
            coords = get_coordinates(location)
            if coords:
                lat, lon = coords
                st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}))
            else:
                st.warning("Could not find the location on the map.")
    else:
        st.warning("Please enter a tweet to make a prediction.")