from model import SpamDetectionPipeline, train_model

import os
import requests
import tempfile
import joblib

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Dial

from pydub import AudioSegment
import whisper

# ------------------------------------------------------------------------------
# AI constants
# ------------------------------------------------------------------------------
TRAIN = True
MODEL_DATA = "extended_data.csv"
MODEL_PATH = "model.json"

AUDIO_MODEL = "base"


# ------------------------------------------------------------------------------
# Twilio Constants
# ------------------------------------------------------------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
TWILIO_RECIPIENT = os.getenv("TWILIO_RECIPIENT")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


# ------------------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------------------
app = FastAPI(title="Spam Detection")


# ------------------------------------------------------------------------------
# Spam Detection Model
# ------------------------------------------------------------------------------
if TRAIN:
    train_model(MODEL_DATA, MODEL_PATH)

spam_pipeline = SpamDetectionPipeline.load_model(MODEL_PATH)
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')


# ------------------------------------------------------------------------------
# Audio
# ------------------------------------------------------------------------------
def transcribe_audio(file_path):
    """Transcribe an audio file (e.g., m4a) to text using OpenAI Whisper"""
    model = whisper.load_model(AUDIO_MODEL)
    audio = AudioSegment.from_file(file_path, format='m4a')
    wav_path = file_path.replace('.m4a', '.wav')
    audio.export(wav_path, format="wav")

    result = model.transcribe(wav_path)
    os.remove(wav_path)  # Clean up temporary WAV file

    return result['text']

# ------------------------------------------------------------------------------
# 5. Twilio Conference Logic in FastAPI
# ------------------------------------------------------------------------------

@app.post("/start-conference")
async def start_conference(request: Request):
    """
    Endpoint to initiate a Twilio conference call by placing outbound calls 
    to two participants. Those calls will then be instructed to join a 
    named conference (defined in /join-conference).
    
    You can pass in 'from_number' and 'to_number' in form data or JSON.
    """
    form_data = await request.form()
    from_number = form_data.get('from_number', TWILIO_PHONE_NUMBER)
    to_number   = form_data.get('to_number', TWILIO_RECIPIENT)

    # Make first outbound call
    call1 = twilio_client.calls.create(
        to=from_number,
        from_=TWILIO_PHONE_NUMBER,
        url=request.url_for('join_conference')  # TwiML endpoint
    )

    # Make second outbound call
    call2 = twilio_client.calls.create(
        to=to_number,
        from_=TWILIO_PHONE_NUMBER,
        url=request.url_for('join_conference')  # TwiML endpoint
    )

    return JSONResponse({
        "status": "conference started",
        "call_sid_1": call1.sid,
        "call_sid_2": call2.sid
    })

@app.post("/join-conference")
async def join_conference():
    """
    Twilio calls this endpoint to retrieve TwiML instructing the call 
    to join a named conference. The conference is recorded from start, and 
    Twilio will send a recording status callback to /handle-recording-complete.
    """
    response = VoiceResponse()
    dial = Dial(
        record='record-from-start',
        recording_status_callback="/handle-recording-complete"
    )
    # Use a static or dynamically generated conference name
    dial.conference("MySpamDetectionConference")
    response.append(dial)

    xml_str = str(response)
    return Response(content=xml_str, media_type="application/xml")

@app.post("/handle-recording-complete")
async def handle_recording_complete(request: Request):
    """
    Twilio calls this endpoint when the conference recording is complete.
    We'll:
    1) Grab the recording URL from Twilio's POST data.
    2) Download the MP3.
    3) Transcribe with Whisper.
    4) Run the spam-detection pipeline.
    5) Return JSON with the results.
    """
    form_data = await request.form()
    recording_url = form_data.get("RecordingUrl")  # e.g. https://api.twilio.com/2010-04-01/Accounts/...
    recording_sid = form_data.get("RecordingSid")
    call_sid = form_data.get("CallSid")

    # The actual audio file is typically at <RecordingUrl>.mp3
    audio_file_url = f"{recording_url}.mp3"

    # 1) Download audio to a temp file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        tmp_file_name = tmp_file.name
        r = requests.get(audio_file_url)
        tmp_file.write(r.content)

    # 2) Transcribe
    try:
        transcribed_text = transcribe_audio(tmp_file_name)
    finally:
        # Clean up local file
        os.remove(tmp_file_name)

    # 3) Predict Spam/Scam Likelihood
    spam_score = spam_pipeline.predict(transcribed_text, tfidf_vectorizer)
    # Convert to a user-friendly float or percentage
    spam_likelihood = float(spam_score)  # 0.0 -> 1.0

    return JSONResponse({
        "call_sid": call_sid,
        "recording_sid": recording_sid,
        "transcribed_text": transcribed_text,
        "spam_likelihood": spam_likelihood
    })

@app.get("/")
async def home():
    return {"message": "Welcome to WiseCall"}

# ------------------------------------------------------------------------------
# 6. Example: Start the Server (if running locally)
# ------------------------------------------------------------------------------
# If you place this in a file named main.py, you can run:
#   uvicorn main:app --reload
#
# Then expose it publicly (via ngrok or a server) so Twilio can reach /start-conference, /join-conference,
# and /handle-recording-complete.
#
# 7. Make POST request to /start-conference with from_number & to_number form data:
#   curl -X POST -F "from_number=+15550123456" -F "to_number=+15550987654" http://localhost:8000/start-conference
#
# Twilio will then fetch /join-conference for each call, which instructs them
# to join "MySpamDetectionConference" with recording enabled. Once the call ends,
# Twilio will POST to /handle-recording-complete, we transcribe & analyze, then respond. 
