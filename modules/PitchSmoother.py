import numpy as np
from util.audio import hz_to_midi, midi_to_hz
from scipy.signal import medfilt

class PitchSmoother():
    def __init__(self, 
                 candidates_freqs,
                 candidates_probs,
                 sample_rate=22050, 
                 window_size=2048, 
                 hop_size=None,
                 fmin=65, 
                 fmax=2093, 
                 bins_per_semitone=10, 
                 voicing_prob=0.5,
                 switch_prob=0.01,
                 max_jump_semitones=2.5):
        self.sample_rate = sample_rate,
        self.window_size = window_size,
        self.hop_size = hop_size if hop_size is not None else window_size // 4,
        self.fmin = fmin,
        self.fmax = fmax
        self.candidates_freqs = candidates_freqs # a sequence of frame-wise candidates freqs
        self.candidates_probs = candidates_probs # corresponding probs above 
        self.bins_per_semitone = bins_per_semitone
        self.voicing_prob = voicing_prob # for observation matrix
        self.switch_prob = switch_prob # for transition matrix
        self.max_jump_semitones = max_jump_semitones

        # Discretize pitch space
        self.n_semitones = 12 * np.log2(fmax / fmin) # number of semitones, e.g. 55Hz ~ 880Hz would have 48 semitones
        self.n_bins = int(np.floor(self.n_semitones * bins_per_semitone)) + 1 # number of bins (e.g.) between 55Hz and 880Hz
        self.midi_min = hz_to_midi(fmin) # minimum midi note number
        self.midi_max = hz_to_midi(fmax) # maximum midi note number
        self.bin_edges = np.linspace(self.midi_min, self.midi_max, self.n_bins + 1) # discretization, each element is the bin edge with the format of midi note number
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2 # midi note number of each bin center
        self.bin_freqs = self.fmin * 2 ** (np.arange(self.n_bins) / (12 * self.bins_per_semitone))

        # maximum number of bins per jump, used in triangular weight distribution
        self.max_jump_bins = int(np.ceil(max_jump_semitones * bins_per_semitone))
        # number of frames
        self.n_frames = len(self.candidates_freqs) 

    def _build_observation_matrix(self):
        obs_probs = np.zeros((self.n_frames, 2 * self.n_bins)) # frames * states
        for i in range(self.n_frames):
            frame_freq = self.candidates_freqs[i]
            frame_prob = self.candidates_probs[i]

            # Treat as silence if no candidates, all candidates are 0 Hz, or very low probability
            if len(frame_freq) == 0 or np.all(frame_freq == 0) or np.sum(frame_prob) < 1e-3:
                obs_probs[i, self.n_bins:] = 1.0
                continue

            midi_cands = hz_to_midi(frame_freq) # transfer candidate freq to midi note number
            cands_bin_indices = np.clip(np.digitize(midi_cands, self.bin_edges) - 1, 0, self.n_bins - 1)

            # voiced part
            for j, cand_bin_idx in enumerate(cands_bin_indices):
                obs_probs[i, cand_bin_idx] += frame_prob[j]

            # # unvoiced part: small uniform probability
            # obs_probs[i, self.n_bins:] = (1 - self.voicing_prob) / self.n_bins
            # obs_probs[i, :] /= np.sum(obs_probs[i, :])
        voiced_prob = np.clip(np.sum(obs_probs[:, :self.n_bins], axis=1, keepdims=True), 0, 1)
        obs_probs[:, self.n_bins:] = (1 - voiced_prob) / self.n_bins

        return obs_probs

    def _build_pitch_transition_matrix(self): 
        pitch_trans = np.zeros((self.n_bins, self.n_bins)) # current_states * next_states

        # triangular distribution
        for i in range(self.n_bins): # current state
            for j in range(self.n_bins): # next state
                if abs(i - j) <= self.max_jump_bins:
                    pitch_trans[i, j] = max(0, self.max_jump_bins + 1 - abs(i - j))

        # normalization
        pitch_trans = pitch_trans / (pitch_trans.sum(axis=1, keepdims=True) + 1e-12)
        return pitch_trans

    def _build_voice_transition_matrix(self):
        voice_trans = np.array([[1 - self.switch_prob, self.switch_prob], 
                               [self.switch_prob, 1 - self.switch_prob]])
        return voice_trans

    def _build_transition_matrix(self):
        pitch_trans = self._build_pitch_transition_matrix()
        voice_trans = self._build_voice_transition_matrix()
        return np.kron(voice_trans, pitch_trans)
    
    def _viterbi_decoding(self):
        log_trans_matrix = np.log(self._build_transition_matrix() + 1e-12)
        log_obs_matrix = np.log(self._build_observation_matrix() + 1e-12)

        viterbi = np.full((self.n_frames, 2 * self.n_bins), -np.inf)
        backpathpointer = np.zeros((self.n_frames, 2 * self.n_bins), dtype=int)
        beststates = np.zeros(self.n_frames, dtype=int)

        # --- Initialization: allow starting in any state (voiced or unvoiced) ---
        viterbi[0, :] = log_obs_matrix[0, :]

        # --- Recursion ---
        for t in range(1, self.n_frames):
            for j in range(2 * self.n_bins):
                trans_probs = viterbi[t - 1, :] + log_trans_matrix[:, j]
                backpathpointer[t, j] = np.argmax(trans_probs)
                viterbi[t, j] = np.max(trans_probs) + log_obs_matrix[t, j]

        # --- Termination ---
        beststates[-1] = np.argmax(viterbi[-1, :])
        for t in range(self.n_frames - 2, -1, -1):
            beststates[t] = backpathpointer[t + 1, beststates[t + 1]]

        return beststates
    
    def _state_to_freq(self, states: np.ndarray, obs_matrix: np.ndarray = None, **kwargs) -> np.ndarray:
        # Initialize output
        f0 = np.zeros(len(states))
        
        # Get voiced frames (states < n_bins are voiced)
        voiced_flag = states < self.n_bins
        
        # Convert states to frequencies (modulo handles HMM's 2*n_bins structure)
        f0[voiced_flag] = self.bin_freqs[states[voiced_flag] % self.n_bins]
        f0[~voiced_flag] = np.nan
        
        # Optional: Apply observation probability threshold (librosa default: 0.1)
        if obs_matrix is not None:
            voiced_probs = np.max(obs_matrix[:, :self.n_bins], axis=1)
            threshold = kwargs.get('min_voiced_prob', 0.1)
            f0[voiced_probs < threshold] = np.nan

        if len(f0) != 0:
            midi = hz_to_midi(f0)
            valid = ~np.isnan(midi)
            for i in range(1, len(f0)):
                if valid[i] and valid[i-1]:
                    semitone_jump = abs(midi[i] - midi[i-1])
                    if semitone_jump > self.max_jump_semitones:
                        f0[i] = np.nan  # Mark jump as NaN

        f0[f0 > 1000] = np.nan
        
        return f0