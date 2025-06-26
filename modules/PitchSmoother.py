from modules.yin import PyinPitchExtractor
import numpy as np
from util.audio import hz_to_midi, midi_to_hz

class PitchSmoother(PyinPitchExtractor):
    def __init__(self, 
                 candidates_freq,
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
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            fmin=fmin,
            fmax=fmax
        )
        self.candidates_freqs = candidates_freq
        self.candidates_probs = candidates_probs
        self.bins_per_semitone = bins_per_semitone
        self.voicing_prob = voicing_prob # observation
        self.switch_prob = switch_prob # transition
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

        self.n_frames = len(self.candidates_freqs) # number of frames

    def _build_observation_matrix(self):
        pass

    def _build_pitch_transition_matrix(self):
        pitch_trans = np.zeros((self.n_bins, self.n_bins))
        
        return pitch_trans

    def _build_voice_transition_matrix(self):
        voice_trans = np.array([[1 - self.switch_prob, self.switch_prob], 
                               [self.switch_prob, 1 - self.switch_prob]])
        return voice_trans

    def _build_transition_matrix(self):
        pitch_trans = self._build_pitch_transition_matrix()
        voice_trans = self._build_voice_transition_matrix()

        return np.kron(voice_trans, pitch_trans)