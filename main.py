# import numpy as np
# import matplotlib.pyplot as plt
# from modules.Pyin import PyinPitchExtractor

# # Parameters
# sr = 22050  # Sample rate
# T = 10.0     # Duration in seconds
# f = 440.0   # Frequency of sine wave (Hz)

# # Generate 440 Hz sine waved
# samples = np.arange(int(T * sr))
# signal = 0.5 * np.sin(2 * np.pi * f * samples / sr)

# # Extract pitch contour
# extractor = PyinPitchExtractor(sample_rate=sr)
# f0,f0_probs,time_series = extractor.signal_pitch_extraction(signal)

# print(f0)

# print(f0_probs)

# print(time_series)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from modules.Pyin import PyinPitchExtractor

# Parameters
sr = 22050  # Sample rate
T = 10.0    # Duration in seconds
f = 440.0   # Frequency of sine wave (Hz)

# Generate 440 Hz sine wave
samples = np.arange(int(T * sr))
signal = 0.5 * np.sin(2 * np.pi * f * samples / sr)

# Extract pitch contour
extractor = PyinPitchExtractor(sample_rate=sr)
f0, f0_probs, time_series = extractor.signal_pitch_extraction(signal)

# Prepare data for 2D histogram
# Since f0 contains multiple candidates per frame, we need to flatten the data
time_points = []
freq_points = []
weights = []

for i, (time, f0_candidates, probs) in enumerate(zip(time_series, f0, f0_probs)):
    for freq, prob in zip(f0_candidates, probs):
        time_points.append(time)
        freq_points.append(freq)
        weights.append(prob)

# Create figure
plt.figure(figsize=(12, 6))

# Define bins
time_bins = np.linspace(min(time_points), max(time_points), 100)
freq_bins = np.logspace(np.log10(50), np.log10(2000), 100)  # Logarithmic frequency axis

# Create 2D histogram
hist, xedges, yedges = np.histogram2d(
    time_points, freq_points, 
    bins=[time_bins, freq_bins],
    weights=weights,
    density=True
)

# Plot the spectrogram-like histogram
plt.pcolormesh(
    xedges, yedges, hist.T,
    norm=LogNorm(vmin=1e-3, vmax=hist.max()),  # Log scale for better visibility
    cmap='viridis',
    shading='auto'
)

# Set logarithmic scale for frequency axis
plt.yscale('log')
plt.yticks([100, 200, 400, 800, 1600])
plt.gca().get_yaxis().set_major_formatter(plt.ScalarFormatter())
plt.ylim(50, 2000)

# Add labels and title
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('PYIN Pitch Probability Spectrogram')
plt.colorbar(label='Probability Density')

# Add a horizontal line at the expected frequency (440Hz)
plt.axhline(y=440, color='red', linestyle='--', alpha=0.5, label='Expected (440 Hz)')
plt.legend()

plt.tight_layout()
plt.show()