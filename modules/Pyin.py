import numpy as np
import scipy
from util import audio
from scipy.stats import beta, boltzmann
from typing import Tuple
class PitchExtractor:
    def __init__(self, sample_rate = 22050, 
                 window_size = 2048, 
                 hop_size = None, 
                 fmin = 65, 
                 fmax = 2093, 
                 n_thresholds = 100, 
                 beta_params = (2, 18),
                 boltzmann_param = 2):
        self.sample_rate = sample_rate
        self.W = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.n_thresholds = n_thresholds
        self.beta_params = beta_params
        self.boltzmann_param = boltzmann_param

        # thresholds from 0 to 1, step = 0.01
        self.thresholds = np.linspace(0, 1, self.n_thresholds + 1)

        # beta distribution
        self.beta_cdf = beta.cdf(self.thresholds, beta_params[0], beta_params[1])
        self.beta_prob = np.diff(self.beta_cdf)

        # The minimum and maximum detectable frequency
        self.min_lag = max(1, int(self.sample_rate / self.fmax))
        self.max_lag = min(window_size // 2, int(self.sample_rate / self.fmin))

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
        is_local_minimas = np.full(cmndf.shape, False)
        for tau in range(1, len(cmndf) - 1):
            if cmndf[tau] < cmndf[tau + 1] and \
                cmndf[tau] < cmndf[tau - 1]:
                is_local_minimas[tau] = True
        local_minimas_index = np.where(is_local_minimas)[0]
        local_minimas = cmndf[local_minimas_index]
        return local_minimas

    def _filter_local_minimas(self, local_minimas: np.ndarray) -> np.ndarray:
        return local_minimas[self.min_lag: self.max_lag]
    
    def _normalization(self, local_minimas: np.ndarray, below_threshold: np.ndarray) -> np.ndarray:
        position = np.arange(len(local_minimas))[below_threshold]
        probs = boltzmann.pmf(position, self.boltzmann_param, len(local_minimas))
        return probs

    def _assign_probabilities(self, local_minimas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        local_minimas_probs = np.zeros_like(local_minimas)
        for i, threshold in enumerate(self.thresholds[1:]):
            below_threshold = local_minimas < threshold
            if np.any(below_threshold):
                probs = self._normalization(local_minimas, below_threshold)
                local_minimas_probs[below_threshold] += probs * self.beta_prob[i]
        
        return local_minimas, local_minimas_probs
    
    def _frame_wise_pitch_extraction(self, frame: np.ndarray) -> np.ndarray:
        pass
    
    def signal_pitch_extraction(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass         
