import numpy as np
from scipy import signal

class PitchExtractor:
    def __init__(self, sample_rate = 220250, window_size = 2048, threshold = 0.1):
        
        self.sample_rate = sample_rate
        self.W = window_size
        self.threshold = threshold

    def _auto_correlation_function(self, frame: np.ndarray) -> np.ndarray:
        '''
        Calculate autocorrelation using FFT

        Args:
            frame: window-sized signal (1D numpy array)
        
        Returns:
            acf: autocorrelation function for frame (1D numpy array)
        '''
        
        n = len(frame)
        f = np.fft.fft(frame, n = 2*n - 1)
        acf = np.fft.ifft(f * np.conj(f)).real
        return acf[:n]

    def _difference_function(self, frame: np.ndarray) -> np.ndarray:
        '''
        Calculate difference funtion based on above auto correlation function for frame

        Args:
            frame: window-sized signal (1D numpy array)

        Returns:
            difference: difference values for frame (1D numpy array)
        '''
        acf = self._auto_correlation_function(frame)
        # unbiased autocorrelation
        difference = 2 * (acf[0] - acf)
        return difference

    def _cumulative_mean_normalized_difference_function(self):
        pass

    def pitch_extraction(self):
        pass