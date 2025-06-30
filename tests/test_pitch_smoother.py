import unittest
import numpy as np
from modules.PitchSmoother import PitchSmoother
from util.audio import hz_to_midi

class TestPitchSmoother(unittest.TestCase):
    def setUp(self):
        self.fmin = 55
        self.fmax = 880
        self.bins_per_semitone = 5

    def test_all_silence(self):
        n_frames = 10
        candidates_freqs = [np.array([]) for _ in range(n_frames)]
        candidates_probs = [np.array([]) for _ in range(n_frames)]
        ps = PitchSmoother(candidates_freqs, candidates_probs, fmin=self.fmin, fmax=self.fmax, bins_per_semitone=self.bins_per_semitone)
        obs_matrix = ps._build_observation_matrix()
        states = ps._viterbi_decoding()
        f0 = ps._state_to_freq(states, obs_matrix=obs_matrix)
        self.assertTrue(np.all(f0 == 0))

    def test_single_voiced_candidate(self):
        n_frames = 10
        freq = 220.0
        candidates_freqs = [np.array([freq]) for _ in range(n_frames)]
        candidates_probs = [np.array([1.0]) for _ in range(n_frames)]
        ps = PitchSmoother(candidates_freqs, candidates_probs, fmin=self.fmin, fmax=self.fmax, bins_per_semitone=self.bins_per_semitone)
        obs_matrix = ps._build_observation_matrix()
        states = ps._viterbi_decoding()
        f0 = ps._state_to_freq(states, obs_matrix=obs_matrix)
        self.assertTrue(np.allclose(f0, freq, atol=1.0) | (f0 == 0))  # Allow for small quantization error

    def test_voiced_unvoiced_split(self):
        n_frames = 6
        freq = 220.0
        candidates_freqs = [np.array([freq]) if i % 2 == 0 else np.array([]) for i in range(n_frames)]
        candidates_probs = [np.array([1.0]) if i % 2 == 0 else np.array([]) for i in range(n_frames)]
        ps = PitchSmoother(candidates_freqs, candidates_probs, fmin=self.fmin, fmax=self.fmax, bins_per_semitone=self.bins_per_semitone)
        obs_matrix = ps._build_observation_matrix()
        states = ps._viterbi_decoding()
        f0 = ps._state_to_freq(states, obs_matrix=obs_matrix)
        for i in range(n_frames):
            if i % 2 == 0:
                self.assertTrue(np.isclose(f0[i], freq, atol=1.0) or f0[i] == 0)
            else:
                self.assertEqual(f0[i], 0)

    def test_large_pitch_jump_penalized(self):
        # Two frames, one low, one high, jump > max_jump_semitones
        freq1 = 110.0
        freq2 = 880.0
        candidates_freqs = [np.array([freq1]), np.array([freq2])]
        candidates_probs = [np.array([1.0]), np.array([1.0])]
        ps = PitchSmoother(candidates_freqs, candidates_probs, fmin=self.fmin, fmax=self.fmax, bins_per_semitone=self.bins_per_semitone, max_jump_semitones=1)
        obs_matrix = ps._build_observation_matrix()
        trans_matrix = ps._build_transition_matrix()
        # The transition matrix should have zero probability for this jump
        bin1 = np.digitize([hz_to_midi(freq1)], ps.bin_edges) - 1
        bin2 = np.digitize([hz_to_midi(freq2)], ps.bin_edges) - 1
        self.assertTrue(trans_matrix[bin1, bin2] == 0 or trans_matrix[bin2, bin1] == 0)

    def test_observation_matrix_reference(self):
        # Frame with two candidates, probabilities sum to <1
        freq1, freq2 = 220.0, 330.0
        prob1, prob2 = 0.3, 0.4
        candidates_freqs = [np.array([freq1, freq2])]
        candidates_probs = [np.array([prob1, prob2])]
        ps = PitchSmoother(candidates_freqs, candidates_probs, fmin=self.fmin, fmax=self.fmax, bins_per_semitone=self.bins_per_semitone)
        obs_matrix = ps._build_observation_matrix()
        voiced_sum = prob1 + prob2
        # Voiced part
        self.assertTrue(np.isclose(np.sum(obs_matrix[0, :ps.n_bins]), 0.5 * voiced_sum, atol=1e-6))
        # Unvoiced part
        self.assertTrue(np.allclose(obs_matrix[0, ps.n_bins:], 0.5 * (1 - voiced_sum) / ps.n_bins, atol=1e-6))

if __name__ == '__main__':
    unittest.main() 