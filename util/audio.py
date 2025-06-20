import numpy as np

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

