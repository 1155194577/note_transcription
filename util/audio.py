import numpy as np

def hz_to_midi(freq):
    """Convert frequency in Hz to MIDI note number. e.g. 440Hz -> 69"""
    return 69 + 12 * np.log2(freq / 440.0)

def midi_to_hz(midi):
    """Convert MIDI note number to frequency in Hz. e.g. 69 -> 440Hz"""
    return 440.0 * 2 ** ((midi - 69) / 12)

def bin_to_frequency(bin_idx, fmin, n_bins_per_semitone):
    '''Convert pitch bin index to frequency'''
    return fmin * 2 ** (bin_idx / (12 * n_bins_per_semitone))

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
    num_frames = 1 + (len(y) - frame_length) // hop_size
    
    if num_frames <= 0:
        return []
    
    frames = []
    
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_length
        frames.append(y[start:end])
    
    return frames

