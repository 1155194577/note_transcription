import numpy as np
from util import audio
class PitchExtractor:
    def __init__(self, sample_rate = 22050, window_size = 2048, hop_size = None, fmin = 65, fmax = 2093, threshold = 0.1):
        self.sample_rate = sample_rate
        self.W = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.threshold = threshold

    def _auto_correlation_function(self, frame: np.ndarray) -> np.ndarray:
        n = len(frame)
        f = np.fft.fft(frame, n = 2*n)
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
    
    def _find_local_minimas(self, cmndf: np.ndarray) -> np.ndarray:
        pass 
   
    def _filter_local_minimas(self, local_minimas: np.ndarray) -> np.ndarray:
        pass

    def _assign_probabilities(self, local_minimas: np.ndarray) -> np.ndarray:
        pass 
    
    def _frame_wise_pitch_extraction(self, frame: np.ndarray) -> np.ndarray:
        pass
    
    def signal_pitch_extraction(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass         
