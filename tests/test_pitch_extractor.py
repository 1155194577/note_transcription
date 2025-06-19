import unittest
import numpy as np
from modules import pitch_extractor as pe


class TestPitchExtractor(unittest.TestCase):
    def test_auto_correlation_function(self):

        def slow_auto_correlation_function(signal):
            acf = np.zeros((len(signal),))
            for lag in range(len(signal)):
                for i in range(len(signal) - lag):
                    acf[lag] += signal[i] * signal[i+lag]
            return acf

        test_signal = np.random.randn(2048)
        pe1 = pe.PitchExtractor()
        acf = pe1._auto_correlation_function(test_signal)
        acf2 = slow_auto_correlation_function(test_signal)
        acf_diff = np.abs(acf - acf2)
        self.assertTrue(np.allclose(acf_diff, 0, atol=1e-6))

    def test_difference_function(self):
        
        def slow_difference_function(signal):
            diff = np.zeros((len(signal),))
            for lag in range(len(signal)):
                for i in range(len(signal) - lag):
                    diff[lag] += (signal[i] - signal[i+lag])**2
            return diff

        test_signal = np.random.randn(2048)
        pe1 = pe.PitchExtractor()
        diff = pe1._difference_function(test_signal)
        diff2 = slow_difference_function(test_signal)
        diff_diff = np.abs(diff - diff2)
        self.assertTrue(np.allclose(diff_diff, 0, atol=1e-6))

if __name__ == '__main__':
    unittest.main()