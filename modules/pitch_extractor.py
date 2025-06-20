import numpy as np
from util import audio
class PitchExtractor:
    def __init__(self, sample_rate = 22050, window_size = 2048, hop_size = None, fmin = 65, fmax = 2093, threshold = 0.1):
        '''
        Args:
            sample_rate: the number of samples per second from a signal/frame

            window_size: the number of samples taken from a signal/frame in a sized window, i.e. frame length

            hop_size: number of samples, the length that the window moves after finishing extracting the fundamental frequency of a frame

            fmin: the minimum frequency that pitch should be

            fmax: the maximum frequency that pitch should be

            threshold: threshold to avoid octave error, in yin algorithm it is absolute
        '''
        self.sample_rate = sample_rate
        self.W = window_size
        self.hop_size = hop_size
        self.fmin = fmin
        self.fmax = fmax
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

    def _frame_wise_pitch_extraction(self, frame: np.ndarray) -> float:
        '''
        For a given frame, extract its pitch through above function

        Args:
            frame: window-sized signal waiting for analysis (1D numpy array)

        Returns:
            f0: fundamental frequency for given frame (float number)
        '''

        # Calculate the minimum lag and maximum lag that the range of tau should be in
        min_lag = max(1, int(self.sample_rate / self.fmax))
        max_lag = min(len(frame) // 2, int(self.sample_rate / self.fmin))

        cmndf_vals = self._cumulative_mean_normalized_difference_function(self._difference_function(frame))

        # find possible local minimum tau
        for tau in range(min_lag, max_lag + 1):
            if cmndf_vals[tau] < self.threshold and \
                cmndf_vals[tau] < cmndf_vals[tau + 1] and \
                cmndf_vals[tau] < cmndf_vals[tau - 1]:
                f0 = self.sample_rate / tau if tau != 0 else 0.0
                return f0
        
        # otherwise, find global minimum
        valid_lags = np.arange(min_lag, min(max_lag + 1, len(cmndf_vals)))
        if len(valid_lags) == 0:
            return 0.0 
        tau = valid_lags[np.argmin(cmndf_vals[valid_lags])]
        f0 = self.sample_rate / tau if tau != 0 else 0.0
        return f0
    
    def signal_pitch_extraction(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''
        Extract and output the fundamental frequency of the whole signal

        Args:
            signal: signal that need to be operated

        Returns:
            time_series: list of time corresponding to each fundamental frequency in f0
            f0: 1D numpy array which denotes the fundamental frequency for each frame of the signal
        '''

        if self.hop_size == None:
            self.hop_size = self.W // 4

        frames = audio.split_audio(signal, self.W, self.hop_size)
        time_series = np.array([])
        f0 = np.array([])

        left_time_idx = 0
        right_time_idx = self.W
        
        for frame in frames:
            f0 = np.append(f0, self._frame_wise_pitch_extraction(frame))
            time_series = np.append(time_series, ((left_time_idx + right_time_idx) // 2) / self.sample_rate)

            left_time_idx += self.hop_size
            right_time_idx += self.hop_size

        return time_series, f0
