import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import queue
import threading
import whisper
import os
from scipy.signal import butter, lfilter

from model import SpamDetectionPipeline
import joblib

##########################################
# Configuration
##########################################
SAMPLE_RATE = 8000         # You mentioned 8 kHz
CHANNELS = 1               # Mono
MODEL_SIZE = "base"        # "tiny", "base", "small", etc.
TEMPERATURE = 0.0          # Lower = less "creative" guesses
WINDOW_SIZE_SEC = 7        # Each transcription window is 7 seconds
OVERLAP_INTERVAL_SEC = 4   # Perform a new transcription every 4 seconds

pipeline = SpamDetectionPipeline.load_model('model')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

audio_queue = queue.Queue()

#########################
# Optional Noise Reduction
#########################
USE_NOISEREDUCE = True
try:
    import noisereduce as nr
except ImportError:
    print("noisereduce library not found. Install with 'pip install noisereduce'.")
    USE_NOISEREDUCE = False

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=100, highcut=3500, fs=SAMPLE_RATE, order=5):
    """
    For an 8 kHz sampling rate, typical phone band is ~300-3400 Hz.
    We can expand it slightly (100-3500) in case the speech has lower fundamentals.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def record_audio():
    """
    Continuously capture audio from the microphone in a blocking manner (no callback).
    """
    print("Recording started (press Ctrl+C to stop)...")
    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16',
            blocksize=1024
        ) as stream:
            while True:
                data, overflow = stream.read(1024)
                if overflow:
                    print("Overflow detected! Samples lost.")
                audio_queue.put(data)
    except KeyboardInterrupt:
        print("Recording stopped.")

def process_audio():
    # Load the Whisper model
    model_whisper = whisper.load_model(MODEL_SIZE)

    # 7 seconds * 8000 samples/sec * 2 bytes/sample = 112,000 bytes
    max_buffer_size = int(WINDOW_SIZE_SEC * SAMPLE_RATE * 2)
    ring_buffer = bytearray()

    last_transcription_time = time.time()

    while True:
        try:
            data = audio_queue.get(timeout=1.0)
            ring_buffer.extend(data)

            # Clip ring_buffer to last 7 seconds
            if len(ring_buffer) > max_buffer_size:
                excess = len(ring_buffer) - max_buffer_size
                ring_buffer = ring_buffer[excess:]

            # Overlap-based transcription
            current_time = time.time()
            if (current_time - last_transcription_time) >= OVERLAP_INTERVAL_SEC:
                last_transcription_time = current_time

                # Convert bytes to float samples
                audio_chunk = bytes(ring_buffer)
                samples = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)

                # 1) Bandpass filter
                filtered = bandpass_filter(samples)

                # 2) Optional: Noise reduction
                if USE_NOISEREDUCE:
                    # Reduce aggression by adjusting parameters
                    reduced = nr.reduce_noise(
                        y=filtered,
                        sr=SAMPLE_RATE,
                        stationary=True,
                        prop_decrease=0.8  # Lower = less aggressive
                    )
                    final_samples = reduced
                else:
                    final_samples = filtered

                # 3) (Optional) In-memory playback test to confirm we have audio
                # sd.play(final_samples, SAMPLE_RATE)
                # sd.wait()

                # 4) Write to temporary WAV
                temp_wav = f"temp{time.time()}.wav"
                with sf.SoundFile(
                    temp_wav, mode='w',
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    subtype='PCM_16'
                ) as f:
                    f.write(final_samples)

                # 5) Transcribe with Whisper
                result = model_whisper.transcribe(
                    temp_wav,
                    temperature=TEMPERATURE
                )
                text = result["text"].strip()

                # (Optional) Remove temp
                # if os.path.exists(temp_wav):
                #     os.remove(temp_wav)

                print(f"\n[Partial Transcript]: {text}")
                spam_likelihood = pipeline.predict(text, tfidf_vectorizer)
                print(f"[Spam Likelihood]: {spam_likelihood:.2%}")

        except queue.Empty:
            continue
        except KeyboardInterrupt:
            print("\nStopping audio processing...")
            break

def main():
    record_thread = threading.Thread(target=record_audio, daemon=True)
    process_thread = threading.Thread(target=process_audio, daemon=True)

    record_thread.start()
    process_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting main program...")

if __name__ == "__main__":
    main()
