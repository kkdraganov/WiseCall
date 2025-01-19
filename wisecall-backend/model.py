# from backend import transcribe_audio

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, VotingRegressor

from xgboost import XGBRegressor

import joblib


class SpamDetectionPipeline:
    def __init__(self):
        self.pipeline = Pipeline([
            ('classifier', VotingRegressor([
                ('xgb', XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    reg_alpha=0.1,  # L1 regularization
                    reg_lambda=1.0, # L2 regularization
                    random_state=42
                )),
                ('rf', RandomForestRegressor(
                    n_estimators=50,
                    max_depth=5,
                    min_samples_split=5,  # Increase minimum samples per split
                    min_samples_leaf=3,   # Increase minimum samples per leaf
                    random_state=42
                ))
            ], weights=[1, 2]))  # Weighted ensemble
        ])
        
    def fit(self, X, y):
        """Train the model"""
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, text, tfidf_vectorizer):
        """Predict probability of spam from raw text"""
        # Preprocess the input text
        tfidf_features = tfidf_vectorizer.transform([text]).toarray()
        additional_features = pd.DataFrame([extract_additional_features(text)])
        combined_features = np.hstack((tfidf_features, additional_features))
        
        # Make predictions
        return np.clip(self.pipeline.predict(combined_features)[0], 0, 1)
    
    def save_model(self, filepath):
        """Save the trained model"""
        joblib.dump(self.pipeline, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model"""
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        return instance

def label_numeric(label_text):
    labels = [
        'legitimate',
        'neutral',
        'slightly_suspicious',
        'suspicious',
        'highly_suspicious',
        'suggesting a dangerous situation',
        'citing urgency',
        'dismissing official protocols',
        'potential_scam',
        'scam_response',
        'scam'
    ]
    numerics = [
        0,
        0.2,
        0.5,
        0.6,
        0.75,
        0.85,
        0.7,
        0.8,
        0.9,
        0.95,
        0.99
    ]
    mapping = dict(zip(labels, numerics))
    return mapping.get(label_text.strip().lower(), 0)

def normalize_labels(labels):
    return list(map(label_numeric, labels))

def extract_additional_features(text):
    urgency_words = ['urgent', 'immediate', 'warning', 'alert', 'deadline', 'asap', 'important']
    return {
        'text_length': len(text),
        'word_count': len(text.split()),
        'contains_urgent': 1 if any(word.strip() in urgency_words for word in text.lower().split()) else 0
    }

def prepare_features(X, tfidf_vectorizer=None):
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_features = tfidf_vectorizer.fit_transform(X)
    else:
        tfidf_features = tfidf_vectorizer.transform(X)

    additional_features = pd.DataFrame([extract_additional_features(text) for text in X])
    return np.hstack((tfidf_features.toarray(), additional_features)), tfidf_vectorizer

def train_model(data_path):
    df = pd.read_csv(data_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['TEXT'], 
        normalize_labels(df['LABEL']), 
        test_size=0.2, 
        random_state=42
    )

    # Prepare features
    X_train_features, tfidf_vectorizer = prepare_features(X_train)
    X_test_features, _ = prepare_features(X_test, tfidf_vectorizer)
    
    # Save the TF-IDF vectorizer for later use
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

    # Train model
    pipeline = SpamDetectionPipeline()
    pipeline.fit(X_train_features, y_train)
    pipeline.save_model('model')

    evaluate_model(pipeline, X_test_features, y_test)

    return pipeline
    
def evaluate_model(pipeline, X_test, y_test):
    # Evaluate
    y_pred = np.clip(pipeline.pipeline.predict(X_test), 0, 1)
    sorted_indices = np.argsort(y_test)
    y_test_sorted = np.array(y_test)[sorted_indices]
    y_pred_sorted = np.array(y_pred)[sorted_indices]

    # Plot true vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_sorted, label='True Values (y_test)', marker='o')
    plt.plot(y_pred_sorted, label='Predicted Values (y_pred)', marker='x')
    plt.title('True vs Predicted Values')
    plt.xlabel('Samples (sorted by y_test)')
    plt.ylabel('Likelihood (0 to 1)')
    plt.legend()
    plt.grid()
    plt.show()


# Train the model
# train_model('extended_data.csv')

# Load the model and vectorizer for prediction
# model = SpamDetectionPipeline.load_model('model')
# tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example: Transcribe and predict from an audio file
# file_path = 'New_Recording_15.mov'
# transcription = transcribe_audio(file_path)
# print(f"Transcription: {transcription}")
# print(f"Spam Likelihood: {model.predict(transcription, tfidf_vectorizer):.2%}")
