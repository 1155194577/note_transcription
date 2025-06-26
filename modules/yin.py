import numpy as np
from typing import Tuple
from util import audio

class YinPitchExtractor:
    def __init__(self, sample_rate=22050, 
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
    
    def _find_local_minimas(self, cmndf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        is_local_minimas = np.r_[False, (cmndf[1:-1] < cmndf[:-2]) & (cmndf[1:-1] < cmndf[2:]), False]
        local_minimas_index = np.where(is_local_minimas)[0]
        local_minimas = cmndf[local_minimas_index]
        return local_minimas, local_minimas_index

    def _filter_local_minimas(self, local_minima: np.ndarray, local_minima_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        filtered = np.array([])
        index = np.array([])
        for minima_index in local_minima_index:
            if self.min_lag < minima_index < self.max_lag:
                index = np.append(index, minima_index)
                filtered = np.append(filtered, local_minima[np.where(local_minima_index == minima_index)])
        return filtered, index.astype(int)
    
    def _assign_probabilities(self, local_minimas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Basic YIN probability assignment"""
        probs = 1 - local_minimas
        probs = np.clip(probs, 0, 1)
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        return local_minimas, probs
    
    def _frame_wise_pitch_extraction(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Core YIN pitch extraction"""
        cmndf_vals = self._cumulative_mean_normalized_difference_function(
            self._difference_function(frame))
        
        local_minima, minima_index = self._find_local_minimas(cmndf_vals)
        filtered_minima, filtered_index = self._filter_local_minimas(local_minima, minima_index)

        if len(filtered_minima) == 0:
            global_min_idx = np.argmin(cmndf_vals[self.min_lag:self.max_lag]) + self.min_lag
            f0 = self.sample_rate / global_min_idx
            return np.array([f0]), np.array([1.0])

        f0_candidates = self.sample_rate / filtered_index
        _, probs = self._assign_probabilities(filtered_minima)
        
        return f0_candidates, probs

    def signal_pitch_extraction(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        frames = audio.split_audio(signal, self.window_size, self.hop_size)
        n_frames = len(frames)

        time_series = (np.arange(n_frames) * self.hop_size + self.window_size // 2) / self.sample_rate
        f0 = []
        for frame in frames:
            f0_candidates, f0_probs = self._frame_wise_pitch_extraction(frame)
            index = np.argmax(f0_probs)
            f0.append(f0_candidates[index])

        return time_series, np.array(f0)