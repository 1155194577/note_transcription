import unittest
import numpy as np
from modules import pitch_extractor as pe

def autocorrelation(x, max_lag=None):
    """
    Compute autocorrelation of a 1D signal up to specified lag.
    
    Args:
        x: Input signal (1D numpy array)
        max_lag: Maximum lag to compute (default: len(x)//2)
    
    Returns:
        autocorr: Autocorrelation values for lags 0 to max_lag
    """
    n = len(x)

    print(n)

    if max_lag is None:
        max_lag = n // 2
    max_lag = min(max_lag, n - 1)
    
    # Mean-center the signal
    x_centered = x - np.mean(x)
    
    # Compute autocorrelation using FFT
    fft = np.fft.fft(x_centered, n=2*n)
    autocorr = np.fft.ifft(fft * np.conj(fft))[:max_lag+1]
    autocorr = np.real(autocorr) / np.var(x) / n
    
    return autocorr

class TestPitchExtractor(unittest.TestCase):
    def test_auto_correlation_function(self):
        pass

if __name__ == '__main__':
    unittest.main()