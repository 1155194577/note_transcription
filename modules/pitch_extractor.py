import numpy as np

class PitchExtractor:
    def __init__(self, sample_rate = 22050, window_size = 2048, threshold = 0.1):
        '''
        Args:
            sample_rate: the number of samples per second from a signal/frame
            window_size: frame length, the number of samples taken from a signal/frame in a sized window
            threshold: threshold to avoid octave error, in yin algorithm it is absolute
        '''
        self.sample_rate = sample_rate
        self.W = window_size
        self.threshold = threshold

    def _auto_correlation_function(self, frame: np.ndarray) -> np.ndarray:
        '''
        Calculate autocorrelation using FFT

        Args:
            frame: window-sized signal (1D numpy array)
        
        Returns:
            acf: autocorrelation function for frame (1D numpy array)
        '''
        
        n = len(frame)
        f = np.fft.fft(frame, n = 2*n)
        acf = np.fft.ifft(f * np.conj(f))[:n].real
        return acf

    def _difference_function(self, frame: np.ndarray) -> np.ndarray:
        '''
        Calculate difference funtion based on above unbiased autocorrelation function for frame

        Args:
            frame: window-sized signal (1D numpy array)

        Returns:
            difference: difference values for frame (1D numpy array)
        '''

        acf = self._auto_correlation_function(frame)
        # unbiased autocorrelation
        difference = 2 * (acf[0] - acf)
        return difference

    def _cumulative_mean_normalized_difference_function(self, difference: np.ndarray) -> np.ndarray:
        '''
        Transfer difference function into cumulative mean normalized difference function, it can avoid zero-lag dip

        Args:
            difference: the output of difference function (1D numpy array)

        Returns:
            cmndf: normalized difference function for frame (1D numpy array)
        '''

        cmndf = np.ones_like(difference)
        cum_sum = np.cumsum(difference)
        normalized = (difference * np.arange(len(difference))) / (cum_sum + 1e-10)
        cmndf[cum_sum > 0] = normalized[cum_sum > 0]
        cmndf[0] = 1.0

        return cmndf

    def pitch_extraction(self, frame: np.ndarray) -> float:
        '''
        For a given frame, extract its pitch through above function

        Args:
            frame: window-sized signal waiting for analysis (1D numpy array)

        Returns:
            F0: fundamental frequency for given frame (float number)
        '''

        cmndf_vals = self._cumulative_mean_normalized_difference_function(self._difference_function(frame))
        for tau in range(len(cmndf_vals)):
            if cmndf_vals[tau] < self.threshold and \
                cmndf_vals[tau] < cmndf_vals[tau + 1] and \
                cmndf_vals[tau] < cmndf_vals[tau - 1]:
                F0 = self.sample_rate / tau if tau != 0 else 0
                return F0
        tau = np.argmin(cmndf_vals[1:])
        F0 = self.sample_rate / tau if tau != 0 else 0
        return F0