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

        print(f"[DEBUG] n_bins={self.n_bins}, voiced: 0-{self.n_bins-1}, unvoiced: {self.n_bins}-{2*self.n_bins-1}")

    def _build_observation_matrix(self):
        obs_probs = np.zeros((self.n_frames, 2 * self.n_bins)) # frames * states
        for i in range(self.n_frames):
            frame_freq = self.candidates_freqs[i]
            frame_prob = self.candidates_probs[i]
            # Treat as silence if no candidates, all candidates are 0 Hz, or very low probability
            if len(frame_freq) == 0 or np.all(frame_freq == 0) or np.sum(frame_prob) < 1e-3:
                print(f"[DEBUG][ObsMatrix] Frame {i}: SILENCE TRIGGERED (len={len(frame_freq)}, all_zero={np.all(frame_freq == 0)}, prob_sum={np.sum(frame_prob)})")
                obs_probs[i, :] = 0.0
                obs_probs[i, self.n_bins] = 1.0
                if i < 5:
                    print(f"[DEBUG][ObsMatrix] Frame {i}: silence, unvoiced idx={self.n_bins}, block={obs_probs[i, self.n_bins:self.n_bins+5]}")
                continue
            midi_cands = hz_to_midi(frame_freq) # transfer candidate freq to midi note number
            cands_bin_indices = np.clip(np.digitize(midi_cands, self.bin_edges) - 1, 0, self.n_bins - 1)
            # voiced part
            for j, cand_bin_idx in enumerate(cands_bin_indices):
                obs_probs[i, cand_bin_idx] += frame_prob[j]
            # unvoiced part: small uniform probability
            obs_probs[i, self.n_bins:] = 1e-4
            obs_probs[i, :] /= np.sum(obs_probs[i, :])
            if i < 5:
                max_idx = np.argmax(obs_probs[i, :])
                print(f"[DEBUG][ObsMatrix] Frame {i}: max idx={max_idx}, value={obs_probs[i, max_idx]}")
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
        self.switch_prob = 0.5  # Encourage unvoiced transitions for debugging
        voice_trans = np.array([[1 - self.switch_prob, self.switch_prob], 
                               [self.switch_prob, 1 - self.switch_prob]])
        print("[DEBUG] Voice transition matrix:\n", voice_trans)
        print(f"[DEBUG] Allowed transitions into first unvoiced state (index {self.n_bins}): {voice_trans[:,1]}")
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
        print("[DEBUG] Initial Viterbi log-probs:", viterbi[0, :10], "...", viterbi[0, -10:])

        # --- Recursion ---
        for t in range(1, self.n_frames):
            for j in range(2 * self.n_bins):
                trans_probs = viterbi[t-1, :] + log_trans_matrix[:, j]
                backpathpointer[t, j] = np.argmax(trans_probs)
                viterbi[t, j] = np.max(trans_probs) + log_obs_matrix[t, j]

        # --- Termination ---
        beststates[-1] = np.argmax(viterbi[-1, :])
        for t in range(self.n_frames - 2, -1, -1):
            beststates[t] = backpathpointer[t+1, beststates[t+1]]
        print("[DEBUG] Viterbi state indices:", beststates[:10], "...", beststates[-10:])
        # Print state index and corresponding f0 for first 5 frames
        f0_map = self._state_to_freq(beststates)
        for i in range(min(5, len(beststates))):
            print(f"[DEBUG][Viterbi] Frame {i}: state={beststates[i]}, f0={f0_map[i]}")
        return beststates
    
    def _state_to_freq(self, states, obs_matrix=None, voicing_threshold=0.5):
        f0 = np.zeros_like(states, dtype=float)
        for i, bin_idx in enumerate(states):
            if i < 5:
                print(f"[DEBUG][State2Freq] Frame {i}: state={bin_idx}, n_bins={self.n_bins}")
            if 0 <= bin_idx < self.n_bins:
                # Voiced state
                if obs_matrix is not None:
                    voiced_prob = np.max(obs_matrix[i, :self.n_bins])
                    if voiced_prob >= voicing_threshold:
                        f0[i] = self.bin_freqs[bin_idx]
                    else:
                        f0[i] = 0.0
                else:
                    f0[i] = self.bin_freqs[bin_idx]
            elif bin_idx >= self.n_bins:
                # Unvoiced state
                f0[i] = 0.0
            if i < 5:
                print(f"[DEBUG][State2Freq] Frame {i}: f0={f0[i]}")
        return f0