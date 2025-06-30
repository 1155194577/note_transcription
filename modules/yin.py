import numpy as np
from typing import Tuple
from util import audio

class YinPitchExtractor:
    def __init__(self, 
                 sample_rate=22050, 
                 window_size=2048, 
                 hop_size=None,
                 fmin=65, 
                 fmax=2093):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size if hop_size is not None else window_size // 4
        self.fmin = fmin
        self.fmax = fmax

        # The minimum and maximum detectable frequency
        self.min_lag = max(1, int(self.sample_rate / self.fmax))
        self.max_lag = min(window_size // 2, int(self.sample_rate / self.fmin))

    def _auto_correlation_function(self, frame: np.ndarray) -> np.ndarray:
        n = len(frame)
        f = np.fft.fft(frame, n=2*n)
        acf = np.fft.ifft(f * np.conj(f))[:n].real
        return acf

    def _difference_function(self, frame: np.ndarray) -> np.ndarray:
        acf = self._auto_correlation_function(frame)
        difference = 2 * (acf[0] - acf)
        return difference

    def _cumulative_mean_normalized_difference_function(self, difference: np.ndarray) -> np.ndarray:
        cmndf = np.ones_like(difference)
        cum_sum = np.cumsum(difference)
        normalized = (difference * np.arange(len(difference))) / (cum_sum + 1e-10)
        cmndf[cum_sum > 0] = normalized[cum_sum > 0]
        cmndf[0] = 1.0
        return cmndf
    
    def _parabolic_interpolation(self, cmndf: np.ndarray, index: int):
        alpha = cmndf[index - 1]
        beta = cmndf[index]
        gamma = cmndf[index + 1]

        if (2 * beta - alpha - gamma) == 0:
            return index
        return index + (gamma - alpha) / (2 * (2 * beta - alpha - gamma))