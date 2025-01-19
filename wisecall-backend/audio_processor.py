from model import train_model, SpamDetectionPipeline
from backend import transcribe_audio
import warnings
import joblib

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)

TRAIN = False
THRESHOLD = 0.6
MODEL_DATA = 'extended_data.csv'
MODEL_PATH = 'model.json'
AUDIO_PATHS = ['normal_example.m4a', 'scam_example.m4a']

if TRAIN:
    train_model(MODEL_DATA, MODEL_PATH)

scam_model = SpamDetectionPipeline.load_model(MODEL_PATH)
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example: Transcribe and predict from an audio file
for audio_path in AUDIO_PATHS:
    transcription = transcribe_audio(audio_path)
    print(f"Transcription: {transcription}")
    scam = scam_model.predict(transcription, tfidf_vectorizer) > THRESHOLD
    likelihood = "Likely" if scam else "Unlikely"
    print(f"Spam Likelihood: {likelihood}")
    print()
