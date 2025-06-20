import numpy as np
import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.audio import split_audio

def test_split_audio_basic():
    """Test basic functionality with a simple input signal."""
    # Create a simple test signal: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    signal = np.arange(10, dtype=float)
    
    # Test non-overlapping frames
    frames = split_audio(signal, frame_length=3, hop_size=3)
    expected = [
        np.array([0., 1., 2.]),
        np.array([3., 4., 5.]),
        np.array([6., 7., 8.])
    ]
    assert len(frames) == 3
    for i, frame in enumerate(frames):
        np.testing.assert_array_equal(frame, expected[i])

def test_split_audio_overlapping():
    """Test with overlapping frames."""
    signal = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    
    # Test overlapping frames (hop_size < frame_length)
    frames = split_audio(signal, frame_length=4, hop_size=2)
    expected = [
        np.array([0., 1., 2., 3.]),
        np.array([2., 3., 4., 5.]),
        np.array([4., 5., 6., 7.]),
        np.array([6., 7., 8., 9.])
    ]
    assert len(frames) == 4
    for i, frame in enumerate(frames):
        np.testing.assert_array_equal(frame, expected[i])

def test_split_audio_edge_cases():
    """Test edge cases."""
    # Test with signal shorter than frame_length
    signal = np.array([0, 1, 2], dtype=float)
    frames = split_audio(signal, frame_length=5, hop_size=2)
    assert len(frames) == 0
    
    # Test with exact fit
    signal = np.array([0, 1, 2, 3], dtype=float)
    frames = split_audio(signal, frame_length=2, hop_size=2)
    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0], np.array([0., 1.]))
    np.testing.assert_array_equal(frames[1], np.array([2., 3.]))

def test_split_audio_empty():
    """Test with empty input."""
    frames = split_audio(np.array([]), frame_length=10, hop_size=5)
    assert frames == []

if __name__ == "__main__":
    pytest.main([__file__])
