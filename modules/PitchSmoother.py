import numpy as np
from util.audio import hz_to_midi, midi_to_hz

class PitchSmoother():
    def __init__(self, 
                 candidates_freqs,
                 candidates_probs,
                 sample_rate=22050, 
                 window_size=2048, 
                 hop_size=None,
                 fmin=65, 
                 fmax=2093, 
                 bins_per_semitone=1, 
                 voicing_prob=0.5,
                 switch_prob=0.01,
                 max_jump_semitones=2.5):
        self.sample_rate = sample_rate,
        self.window_size = window_size,
        self.hop_size = hop_size if hop_size is not None else window_size // 4,
        self.fmin = fmin,
        self.fmax = fmax
        self.candidates_freqs = np.array(candidates_freqs, dtype=object) # a sequence of frame-wise candidates freqs
        self.candidates_probs = np.array(candidates_probs, dtype=object) # corresponding probs above 
        self.bins_per_semitone = bins_per_semitone
        self.voicing_prob = voicing_prob # for observation matrix
        self.switch_prob = switch_prob # for transition matrix
        self.max_jump_semitones = max_jump_semitones

        # Discretize pitch space
        self.n_semitones = 12 * np.log2(fmax / fmin) # number of semitones, e.g. 55Hz ~ 880Hz would have 48 semitones
        self.n_bins = int(np.ceil(self.n_semitones * bins_per_semitone)) # number of bins (e.g.) between 55Hz and 880Hz
        self.midi_min = hz_to_midi(fmin) # minimum midi note number
        self.midi_max = hz_to_midi(fmax) # maximum midi note number
        self.bin_edges = np.linspace(self.midi_min, self.midi_max, self.n_bins + 1) # discretization, each element is the bin edge with the format of midi note number
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2 # midi note number of each bin center
        self.bin_freqs = midi_to_hz(self.bin_centers) # convert bin center to bin freq

        # maximum number of bins per jump, used in triangular weight distribution
        self.max_jump_bins = int(np.ceil(max_jump_semitones * bins_per_semitone))
        # number of frames
        self.n_frames = len(self.candidates_freqs) 

    def _build_observation_matrix(self):
        obs_probs = np.zeros((self.n_frames, 2 * self.n_bins)) # frames * states
        for i in range(self.n_frames):
            frame_freq = self.candidates_freqs[i]
            frame_prob = self.candidates_probs[i]
            if len(frame_freq) == 0:
                # all unvoiced
                obs_probs[i, self.n_bins: ] = 1.0
                continue
            midi_cands = hz_to_midi(frame_freq) # transfer candidate freq to midi note number
            cands_bin_indices = np.clip(np.digitize(midi_cands, self.bin_edges) - 1, 0, self.n_bins - 1)
            # voiced part
            for j, cand_bin_index in enumerate(cands_bin_indices):
                obs_probs[i, cand_bin_index] += self.voicing_prob * frame_prob[j]
            # unvoiced part
            obs_probs[i, self.n_bins: ] = (1 - self.voicing_prob) * (1 - np.sum(frame_prob))
        # normalization
        obs_probs = obs_probs / (obs_probs.sum(axis=1, keepdims=True) + 1e-12)
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
        # voiced->voiced: 1 - switch_prob
        # voiced->unvoiced: switch_prob
        # unvoiced->voiced: switch_prob
        # unvoiced->unvoiced: 1 - switch_prob
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

        viterbi = np.zeros((self.n_frames, 2 * self.n_bins)) # store logged prob
        backpathpointer = np.zeros((self.n_frames, 2 * self.n_bins), dtype=int) # store best path pointer
        beststates = np.zeros(self.n_frames, dtype=int) # store best path/states -> which bin is the best in each frame

        # initialize
        viterbi[0, :] = log_obs_matrix[0, :]

        for i in range(1, self.n_frames): # frames
            for j in range(2 * self.n_bins): # states
                # viterbi[i, j] = viterbi[i - 1, :] + log_trans_matrix[j, :] + log_obs_matrix[i, j]
                # backpath[i, j] = np.argmax(viterbi[i -1, :])
                prev_viterbi = viterbi[i - 1, :] + log_trans_matrix[j, :] # prev states
                backpathpointer[i, j] = np.argmax(prev_viterbi) # store best pathes except the last vitrebi's argmax
                viterbi[i, j] = prev_viterbi[backpathpointer[i, j]] + log_obs_matrix[i, j] # current state
        # store the last best path
        beststates[-1] = np.argmax(viterbi[-1, :])
        # store the remaining best path in backpath from the end to the start
        for idx in range(self.n_frames - 2, -1, -1):
            beststates[idx] = backpathpointer[idx + 1, beststates[idx + 1]]
        return beststates
    
    def _state_to_freq(self, states):
        f0 = np.zeros_like(states)
        for i, bin_idx in enumerate(states):
            if 0 <= bin_idx < self.n_bins:
                f0[i] = self.bin_freqs[bin_idx]
            else: f0[i] = 0.0
        return f0