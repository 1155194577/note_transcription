import unittest
import numpy as np
from modules.Pyin import PyinPitchExtractor
from modules.PitchSmoother import PitchSmoother


class TestPitchExtractor(unittest.TestCase):
    def test_auto_correlation_function(self):

        def slow_auto_correlation_function(signal):
            acf = np.zeros((len(signal),))
            for lag in range(len(signal)):
                for i in range(len(signal) - lag):
                    acf[lag] += signal[i] * signal[i+lag]
            return acf

        test_signal = np.random.randn(2048)
        pe1 = PyinPitchExtractor()
        acf = pe1._auto_correlation_function(test_signal)
        acf2 = slow_auto_correlation_function(test_signal)
        acf_diff = np.abs(acf - acf2)
        self.assertTrue(np.allclose(acf_diff, 0, atol=1e-6))

    def test_pitch_extraction_sine_wave(self):
        # Generate a 440 Hz sine wave
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        freq = 440.0
        y = 0.5 * np.sin(2 * np.pi * freq * t)
        pe = PyinPitchExtractor(sample_rate=sr)
        f0, probs = pe.signal_pitch_extraction(y)
        # Check that the median pitch is close to 440 Hz (A4)
        self.assertTrue(np.median(f0[np.nonzero(f0)]) - 440 < 5)

    def test_pitch_extraction_silence(self):
        sr = 22050
        duration = 1.0
        n_frames = 40
        # Use empty candidates and probabilities for each frame
        candidates_freqs = [np.array([]) for _ in range(n_frames)]
        candidates_probs = [np.array([]) for _ in range(n_frames)]
        from modules.PitchSmoother import PitchSmoother
        ps = PitchSmoother(candidates_freqs, candidates_probs)
        obs_matrix = ps._build_observation_matrix()
        states = ps._viterbi_decoding()
        f0 = ps._state_to_freq(states, obs_matrix=obs_matrix, voicing_threshold=0.5)
        print("Silence test f0:", f0)
        self.assertTrue(np.all(f0 == 0))

if __name__ == '__main__':
    unittest.main()