import numpy as np
import matplotlib.pyplot as plt
from modules.Pyin import PyinPitchExtractor

# Parameters
sr = 22050  # Sample rate
T = 10.0     # Duration in seconds
f = 440.0   # Frequency of sine wave (Hz)

# Generate 440 Hz sine wave
samples = np.arange(int(T * sr))
signal = 0.5 * np.sin(2 * np.pi * f * samples / sr)

# Extract pitch contour
extractor = PyinPitchExtractor(sample_rate=sr)
f0, _ = extractor.signal_pitch_extraction(signal)

# Plot the pitch contour
plt.figure(figsize=(10, 4))
plt.plot(f0, label="Estimated f0 (Hz)")
plt.axhline(f, color='r', linestyle='--', label="True 440 Hz")
plt.xlabel("Frame")
plt.ylabel("Frequency (Hz)")
plt.title("PYIN Pitch Extraction on 440 Hz Sine Wave")
plt.legend()
plt.tight_layout()
plt.show()

