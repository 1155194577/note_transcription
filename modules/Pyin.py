import numpy as np
import scipy
from util import audio
from scipy.stats import beta, boltzmann
from typing import Tuple
from modules.yin import YinPitchExtractor
class PyinPitchExtractor(YinPitchExtractor):
    def __init__(self, sample_rate=22050, window_size=2048, hop_size=None,
                 fmin=65, fmax=2093, n_thresholds=100, beta_params=(2, 18),
                 boltzmann_param=2, no_local_minimum_prob=0.):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            fmin=fmin,
            fmax=fmax
        )
        self.n_thresholds = n_thresholds
        self.beta_params = beta_params
        self.boltzmann_param = boltzmann_param
        self.no_local_minimum_prob = no_local_minimum_prob

        # thresholds from 0 to 1, step = 0.01
        self.thresholds = np.linspace(0, 1, self.n_thresholds + 1)
        # beta distribution
        self.beta_cdf = beta.cdf(self.thresholds, beta_params[0], beta_params[1])
        self.beta_prob = np.diff(self.beta_cdf)

    def _boltzmann(self, local_minimas: np.ndarray, below_threshold: np.ndarray) -> np.ndarray:
        position = np.arange(len(local_minimas))[below_threshold]
        probs = boltzmann.pmf(position, self.boltzmann_param, len(local_minimas))
        return probs

    def _assign_probabilities(self, local_minimas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """PYIN version with beta and Boltzmann distributions"""
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