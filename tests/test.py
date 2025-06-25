import numpy as np
from scipy.stats import beta, boltzmann
from typing import Tuple

class PitchExtractor:
    def __init__(self, sample_rate=22050, 
                 window_size=2048, 
                 hop_size=None,
                 fmin=65,
                 fmax=2093,
                 n_thresholds=100,
                 beta_params=(2, 18),
                 boltzmann_param=2,
                 resolution=0.1,
                 max_transition_rate=35.92,
                 switch_prob=0.01,
                 no_trough_prob=0.01):
        '''
        Enhanced PYIN pitch extractor with Viterbi decoding
        
        Args:
            sample_rate: Audio sample rate in Hz
            window_size: Analysis frame length in samples
            hop_size: Hop length between frames (default window_size//4)
            fmin: Minimum frequency in Hz
            fmax: Maximum frequency in Hz
            n_thresholds: Number of thresholds for peak picking
            beta_params: Shape parameters for beta distribution
            boltzmann_param: Parameter for Boltzmann distribution
            resolution: Pitch bin resolution in semitones
            max_transition_rate: Maximum pitch transition rate in octaves/sec
            switch_prob: Probability of voicing state switch
            no_trough_prob: Probability for global minimum when no trough found
        '''
        self.sample_rate = sample_rate
        self.W = window_size
        self.hop_size = hop_size if hop_size is not None else window_size // 4
        self.fmin = fmin
        self.fmax = fmax
        self.n_thresholds = n_thresholds
        self.beta_params = beta_params
        self.boltzmann_param = boltzmann_param
        self.resolution = resolution
        self.max_transition_rate = max_transition_rate
        self.switch_prob = switch_prob
        self.no_trough_prob = no_trough_prob

        # Precompute thresholds and beta probabilities
        self.thresholds = np.linspace(0, 1, n_thresholds + 1)
        self.beta_cdf = beta.cdf(self.thresholds, *beta_params)
        self.beta_prob = np.diff(self.beta_cdf)

        # Pitch bin parameters
        self.n_bins_per_semitone = int(1 / resolution)
        self.n_pitch_bins = int(12 * self.n_bins_per_semitone * np.log2(fmax/fmin)) + 1
        self.min_period = max(1, int(sample_rate / fmax))
        self.max_period = min(window_size // 2, int(sample_rate / fmin))

    def extract_pitch(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Extract pitch using PYIN algorithm with Viterbi decoding
        
        Args:
            signal: Input audio signal (1D numpy array)
            
        Returns:
            f0: Estimated fundamental frequencies (Hz)
            voiced_flag: Boolean array indicating voiced frames
            voiced_prob: Probability of voicing for each frame
        '''
        # Split audio into frames
        frames = self._split_frames(signal)
        n_frames = frames.shape[1]

        # Initialize outputs
        yin_frames = np.zeros((self.max_period - self.min_period + 1, n_frames))
        parabolic_shifts = np.zeros_like(yin_frames)

        # Compute YIN features for each frame
        for t in range(n_frames):
            yin_frames[:, t], parabolic_shifts[:, t] = self._compute_yin_features(frames[:, t])

        # Compute observation probabilities
        obs_probs, voiced_prob = self._compute_observation_probs(yin_frames, parabolic_shifts)

        # Build transition matrix
        transition = self._build_transition_matrix()

        # Viterbi decoding
        states = self._viterbi_decode(obs_probs, transition)

        # Convert states to pitch and voicing
        f0, voiced_flag = self._states_to_pitch(states)

        return f0, voiced_flag, voiced_prob

    def _split_frames(self, signal: np.ndarray) -> np.ndarray:
        '''Split signal into overlapping frames'''
        n_frames = 1 + (len(signal) - self.W) // self.hop_size
        frames = np.zeros((self.W, n_frames))
        
        for t in range(n_frames):
            start = t * self.hop_size
            frames[:, t] = signal[start:start+self.W]
            
        return frames

    def _compute_yin_features(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''Compute YIN features for a single frame'''
        # Difference function
        acf = self._auto_correlation_function(frame)
        difference = 2 * (acf[0] - acf[self.min_period:self.max_period+1])

        # Cumulative mean normalized difference
        cmndf = np.ones_like(difference)
        cum_sum = np.cumsum(difference)
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = difference * np.arange(1, len(difference)+1) / (cum_sum + 1e-10)
        cmndf[cum_sum > 0] = normalized[cum_sum > 0]

        # Parabolic interpolation
        shifts = self._parabolic_interpolation(cmndf)

        return cmndf, shifts

    def _auto_correlation_function(self, frame: np.ndarray) -> np.ndarray:
        '''Compute autocorrelation using FFT'''
        n = len(frame)
        f = np.fft.fft(frame, n=2*n)
        acf = np.fft.ifft(f * np.conj(f))[:n].real
        return acf

    def _parabolic_interpolation(self, yin_frame: np.ndarray) -> np.ndarray:
        '''Parabolic interpolation of YIN minima'''
        shifts = np.zeros_like(yin_frame)
        for i in range(1, len(yin_frame)-1):
            alpha = yin_frame[i-1]
            beta = yin_frame[i]
            gamma = yin_frame[i+1]
            
            if 2*beta - alpha - gamma != 0:
                shifts[i] = (gamma - alpha) / (2 * (2*beta - alpha - gamma))
        return shifts

    def _compute_observation_probs(self, yin_frames: np.ndarray, 
                                 parabolic_shifts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''Compute observation probabilities for Viterbi'''
        n_frames = yin_frames.shape[1]
        obs_probs = np.zeros((2 * self.n_pitch_bins, n_frames))
        voiced_prob = np.zeros(n_frames)

        for t in range(n_frames):
            # Find troughs
            is_trough = np.r_[False, 
                            yin_frames[1:-1, t] < yin_frames[:-2, t], 
                            yin_frames[1:-1, t] < yin_frames[2:, t],
                            False]
            trough_idx = np.where(is_trough)[0]

            if len(trough_idx) == 0:
                continue

            # Compute trough probabilities
            trough_probs = np.zeros(len(trough_idx))
            trough_heights = yin_frames[trough_idx, t]
            
            for i, thresh in enumerate(self.thresholds[1:]):
                below_thresh = trough_heights < thresh
                if np.any(below_thresh):
                    # Boltzmann distribution over trough positions
                    pos = np.arange(len(trough_idx))[below_thresh]
                    probs = boltzmann.pmf(pos, self.boltzmann_param, len(trough_idx))
                    trough_probs[below_thresh] += probs * self.beta_prob[i]

            # Add probability for global minimum if no troughs below threshold
            global_min = np.argmin(trough_heights)
            n_below = np.sum(trough_heights < self.thresholds[-1])
            if n_below == 0:
                trough_probs[global_min] += self.no_trough_prob

            # Assign to pitch bins
            for i, idx in enumerate(trough_idx):
                period = self.min_period + idx + parabolic_shifts[idx, t]
                f0 = self.sample_rate / period
                bin_idx = self._frequency_to_bin(f0)
                if 0 <= bin_idx < self.n_pitch_bins:
                    obs_probs[bin_idx, t] += trough_probs[i]

            # Normalize and set unvoiced probabilities
            voiced_prob[t] = np.sum(obs_probs[:self.n_pitch_bins, t])
            obs_probs[self.n_pitch_bins:, t] = (1 - voiced_prob[t]) / self.n_pitch_bins

        return obs_probs, voiced_prob
    
    def _assign_probs(self, frame: np.ndarray):
        is_trough = np.r_[False, 
                            (frame[1:-1] < frame[:-2]) &
                            (frame[1:-1] < frame[2:]),
                            False]
        trough_idx = np.where(is_trough)[0]

        trough_probs = np.zeros(len(trough_idx))
        trough_heights = frame[trough_idx]

        for i, thresh in enumerate(self.thresholds[1:]):
            below_thresh = trough_heights < thresh
            if np.any(below_thresh):
                # Boltzmann distribution over trough positions
                pos = np.arange(len(trough_idx))[below_thresh]
                probs = boltzmann.pmf(pos, self.boltzmann_param, len(trough_idx))
                trough_probs[below_thresh] += probs * self.beta_prob[i]
        
        return trough_heights, trough_probs

    def _build_transition_matrix(self) -> np.ndarray:
        '''Build HMM transition matrix'''
        # Pitch transition matrix (encourages small pitch changes)
        pitch_trans = np.zeros((self.n_pitch_bins, self.n_pitch_bins))
        max_shift = int(self.max_transition_rate * 12 * self.hop_size / self.sample_rate)
        
        for i in range(self.n_pitch_bins):
            for j in range(max(-i, -max_shift), min(self.n_pitch_bins-i, max_shift+1)):
                if 0 <= i+j < self.n_pitch_bins:
                    # Triangle window - closer jumps more likely
                    pitch_trans[i, i+j] = max(0, 1 - abs(j)/max_shift)
        
        # Normalize rows
        pitch_trans = pitch_trans / pitch_trans.sum(axis=1, keepdims=True)

        # Voicing transition matrix
        voice_trans = np.array([[1-self.switch_prob, self.switch_prob],
                               [self.switch_prob, 1-self.switch_prob]])

        # Combine using Kronecker product
        return np.kron(voice_trans, pitch_trans)

    def _viterbi_decode(self, obs_probs: np.ndarray, 
                       transition: np.ndarray) -> np.ndarray:
        '''Viterbi algorithm for finding most likely state sequence'''
        n_states, n_frames = obs_probs.shape
        viterbi = np.zeros((n_states, n_frames))
        backpointer = np.zeros((n_states, n_frames), dtype=int)

        # Initialize
        viterbi[:, 0] = np.log(obs_probs[:, 0] + 1e-10)

        # Forward pass
        for t in range(1, n_frames):
            for j in range(n_states):
                trans_probs = viterbi[:, t-1] + np.log(transition[:, j] + 1e-10)
                best_state = np.argmax(trans_probs)
                viterbi[j, t] = trans_probs[best_state] + np.log(obs_probs[j, t] + 1e-10)
                backpointer[j, t] = best_state

        # Backtracking
        states = np.zeros(n_frames, dtype=int)
        states[-1] = np.argmax(viterbi[:, -1])
        
        for t in range(n_frames-2, -1, -1):
            states[t] = backpointer[states[t+1], t+1]

        return states

    def _states_to_pitch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''Convert state sequence to pitch and voicing'''
        f0 = np.zeros(len(states))
        voiced_flag = states < self.n_pitch_bins

        for t, s in enumerate(states):
            if voiced_flag[t]:
                f0[t] = self._bin_to_frequency(s)
            else:
                f0[t] = 0.0  # Unvoiced

        return f0, voiced_flag

    def _frequency_to_bin(self, f0: float) -> int:
        '''Convert frequency to pitch bin index'''
        return int(round(12 * self.n_bins_per_semitone * np.log2(f0/self.fmin)))

    def _bin_to_frequency(self, bin_idx: int) -> float:
        '''Convert pitch bin index to frequency'''
        return self.fmin * 2 ** (bin_idx / (12 * self.n_bins_per_semitone))