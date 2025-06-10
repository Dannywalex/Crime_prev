# Crime_prev
499 project
Crime Tweet Classifier (Hashing + TF-IDF + Calibrated RF)

A Streamlit-based web app and training pipeline to classify tweets into four categories:
- `criminal_slang`
- `crime_reactive`
- `emergency_reactive`
- `general`

The model uses a HashingVectorizer → TfidfTransformer → Calibrated RandomForestClassifier pipeline to handle unseen words and provide well-calibrated probabilities.

---

 Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Repository Structure](#repository-structure)  
4. [Prerequisites](#prerequisites)  
5. [Installation](#installation)  
6. [Data Preparation](#data-preparation)  
7. [Training the Model](#training-the-model)  
8. [Evaluating the Model](#evaluating-the-model)  
9. [Streamlit App](#streamlit-app)  
10. [Deployment](#deployment)  
11. [Usage](#usage)  
12. [Configuration](#configuration)  
13. [Results & Metrics](#results--metrics)  
14. [Troubleshooting](#troubleshooting)  
15. [Contributing](#contributing)  
16. [License](#license)  
17. [Contact](#contact)

---

 Project Overview

This project implements a multi-class tweet classifier to detect different crime-related categories in tweets. It uses:

- HashingVectorizer for stateless feature extraction, allowing unseen words to map into a fixed feature space.
- TfidfTransformer to weight features.
- RandomForestClassifier wrapped in CalibratedClassifierCV (isotonic or sigmoid) for reliable probability estimates.
- Extensive evaluation: confusion matrix, per-class F1, ROC and Precision–Recall curves (one-vs-rest).
- A Streamlit app for interactive inference.

---

Features

- Multi-class classification**: distinguishes between `criminal_slang`, `crime_reactive`, `emergency_reactive`, and `general`.
- Handles unseen words robustly via HashingVectorizer.
- Probability calibration so that predicted probabilities reflect true confidence.
- Detailed evaluation: test accuracy, cross-validation F1 (macro), classification report, confusion matrix, per-class bar plots, ROC & PR curves in subplots.
- Streamlit interface: input tweet text, get predicted category and top-2 probabilities, optionally show full probability bar chart.
- Easy retraining: standalone training script to read a labeled CSV, preprocess, train with anti-overfitting hyperparameters, calibrate, evaluate, and save pipeline.
- Deployment-ready: can be hosted on Streamlit Cloud or other services; requirements specified.

---


Repository Structure

