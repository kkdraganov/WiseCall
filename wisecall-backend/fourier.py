import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Function to calculate FFT and return frequency and magnitude
def calculate_fft(audio_file):
    audio_data, sample_rate = sf.read(audio_file)

    # If stereo, convert to mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Perform FFT
    fft_result = np.fft.fft(audio_data)
    fft_magnitude = np.abs(fft_result)  # Magnitude of the FFT
    frequencies = np.fft.fftfreq(len(audio_data), 1 / sample_rate)
    return frequencies[:len(frequencies)//2], fft_magnitude[:len(frequencies)//2]

# List of audio files to plot
audio_files = ["temp.wav", "temp1737268316.3017046.wav", "temp1737268395.9846897.wav", "temp1737268377.6248093.wav"]
labels = ["Audio 1", "Audio 2", "Audio 3", "Audio 4"]

plt.figure(figsize=(12, 6))

for file, label in zip(audio_files, labels):
    frequencies, fft_magnitude = calculate_fft(file)
    plt.plot(frequencies, fft_magnitude, label=label)

# Customize the plot
plt.title("Frequency Spectra Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()
plt.show()

