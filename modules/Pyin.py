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
                 boltzmann_param = 2,
                 no_local_minimum_prob = 0.):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
        self.n_thresholds = n_thresholds
        self.beta_params = beta_params
        self.boltzmann_param = boltzmann_param
        self.no_local_minimum_prob = no_local_minimum_prob

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
    
    def _find_local_minimas(self, cmndf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        is_local_minimas = np.r_[False, (cmndf[1:-1] < cmndf[:-2]) & (cmndf[1:-1] < cmndf[2:]), False]
        local_minimas_index = np.where(is_local_minimas)[0]
        local_minimas = cmndf[local_minimas_index]
        return local_minimas, local_minimas_index

    def _filter_local_minimas(self, local_minima:np.ndarray, local_minima_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        filtered = np.array([])
        index = np.array([]).astype(int)
        for minima_index in local_minima_index:
            if self.min_lag < minima_index < self.max_lag:
                index = np.append(index, minima_index)
                filtered = np.append(filtered, local_minima[np.where(local_minima_index == minima_index)])
        return filtered, index
    
    def _boltzmann(self, local_minimas: np.ndarray, below_threshold: np.ndarray) -> np.ndarray:
        position = np.arange(len(local_minimas))[below_threshold]
        probs = boltzmann.pmf(position, self.boltzmann_param, len(local_minimas))
        return probs

    def _assign_probabilities(self, local_minimas: np.ndarray, local_minima_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        local_minimas_probs = np.zeros_like(local_minimas)
        for i, threshold in enumerate(self.thresholds[1:]):
            below_threshold = local_minimas < threshold
            if np.any(below_threshold):
                probs = self._boltzmann(local_minimas, below_threshold)
                local_minimas_probs[below_threshold] += probs * self.beta_prob[i]

        total_prob = np.sum(local_minimas_probs)
        if total_prob > 0:
            local_minimas_probs = local_minimas_probs * (1 - self.no_local_minimum_prob) / total_prob
        
        return local_minimas, local_minimas_probs
    
    def _frame_wise_pitch_extraction(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cmndf_vals = self._cumulative_mean_normalized_difference_function(self._difference_function(frame))
        original_local_minima, original_minima_index = self._find_local_minimas(cmndf_vals)
        filtered_local_minima, filtered_minima_index = self._filter_local_minimas(original_local_minima, original_minima_index)

        local_minima, local_minima_probs = self._assign_probabilities(filtered_local_minima, filtered_minima_index)

        # if no local minima
        if len(filtered_local_minima) == 0:
            global_min_idx = np.argmin(cmndf_vals[self.min_lag:self.max_lag]) + self.min_lag
            f0 = self.sample_rate / global_min_idx
            return np.array([f0]), np.array([self.no_local_minimum_prob])
    
        f0_candidates = []
        f0_probs = []

        for idx in range(len(local_minima)):
            f0 = self.sample_rate / filtered_minima_index[idx]
            f0_candidates.append(f0)
            f0_probs.append(local_minima_probs[idx])

        return np.array(f0_candidates), np.array(f0_probs)
    
    def signal_pitch_extraction(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        frames = audio.split_audio(signal, self.window_size, self.hop_size)
        n_frames = len(frames)

        time_series = (np.arange(n_frames) * self.hop_size + self.W // 2) / self.sample_rate
        f0 = []
        for frame in frames:
            f0_candidates, f0_probs = self._frame_wise_pitch_extraction(frame)
            index = np.argmax(f0_probs)
            f0.append(f0_candidates[index])

        return time_series, np.array(f0)
