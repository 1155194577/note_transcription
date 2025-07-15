import numpy as np

def hz_to_midi(freq):
    """Convert frequency in Hz to MIDI note number. e.g. 440Hz -> 69"""
    freq = np.asarray(freq, dtype=float)
    return 69 + 12 * np.log2(freq / 440.0 + 1e-12)

def midi_to_hz(midi):
    """Convert MIDI note number to frequency in Hz. e.g. 69 -> 440Hz"""
    return 440.0 * 2 ** ((midi - 69) / 12)

def compute_energy_mask(amplitude, percentile=25, min_db=-27.8):
    """Generate boolean mask for frames above energy threshold."""
    energy_db = 10 * np.log10(amplitude / (np.max(amplitude) + 1e-12))
    threshold = np.percentile(energy_db[energy_db > -np.inf], percentile)
    threshold = max(threshold, min_db)  # Ensure not too sensitive
    return energy_db > threshold

def split_audio(y: np.ndarray, frame_length: int, hop_size: int) -> list[np.ndarray]:
    """
    Split an audio signal into overlapping frames.

    Args:
        y: Input audio signal (1D numpy array)
        frame_length: Length of each frame in samples
        hop_size: Number of samples between the start of consecutive frames

    Returns:
        frames: list of 1D numpy array where each element is a frame of the signal
    """
    num_frames = len(y)// hop_size
    
    if num_frames <= 0:
        return []
    
    frames = []
    
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_length
        frames.append(y[start:end])
    
    return frames

