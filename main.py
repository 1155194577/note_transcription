import numpy as np
import matplotlib.pyplot as plt
from modules.Pyin import PyinPitchExtractor
from modules.PitchSmoother import PitchSmoother


def pitch_extraction(signal, sr=22050):
    # Use PyinPitchExtractor to extract pitch candidates and probabilities
    extractor = PyinPitchExtractor(sample_rate=sr)
    f0_candidates, f0_probs, time_series = extractor.signal_pitch_extraction(signal)
    return f0_candidates, f0_probs, time_series

def hmm_smoothing(f0_candidates, f0_probs, sr=22050):
    # Use PitchSmoother to smooth the pitch candidates
    pitch_smoother = PitchSmoother(f0_candidates, f0_probs, sample_rate=sr)
    obs_matrix = pitch_smoother._build_observation_matrix()
    # --- Viterbi trellis ---
    log_trans_matrix = np.log(pitch_smoother._build_transition_matrix() + 1e-12)
    log_obs_matrix = np.log(obs_matrix + 1e-12)
    n_frames, n_states = log_obs_matrix.shape
    viterbi = np.full((n_frames, n_states), -np.inf)
    backpathpointer = np.zeros((n_frames, n_states), dtype=int)
    beststates = np.zeros(n_frames, dtype=int)
    viterbi[0, :] = log_obs_matrix[0, :]
    for t in range(1, n_frames):
        for j in range(n_states):
            trans_probs = viterbi[t - 1, :] + log_trans_matrix[:, j]
            backpathpointer[t, j] = np.argmax(trans_probs)
            viterbi[t, j] = np.max(trans_probs) + log_obs_matrix[t, j]
    beststates[-1] = np.argmax(viterbi[-1, :])
    for t in range(n_frames - 2, -1, -1):
        beststates[t] = backpathpointer[t + 1, beststates[t + 1]]
    f0_smoothed = pitch_smoother._state_to_freq(beststates, obs_matrix=obs_matrix)
    return f0_smoothed, obs_matrix, pitch_smoother, viterbi

def note_segmentation():
    pass 

def main():
    # Generate a 440 Hz sine wave signal
    sr = 22050
    T = 1.0  # seconds
    t = np.linspace(0, T, int(sr * T), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)

    f0_candidates, f0_probs, time_series = pitch_extraction(signal, sr=sr)
    print('Candidates:', f0_candidates)
    print('Probabilities:', f0_probs)
    f0_smoothed, obs_matrix, pitch_smoother, viterbi = hmm_smoothing(f0_candidates, f0_probs, sr=sr)
    print('Smoothed:', f0_smoothed)

    # Plotting
    plt.figure(figsize=(12, 6))
    # Plot candidate frequencies as a scatter plot
    for i, (tt, cands, probs) in enumerate(zip(time_series, f0_candidates, f0_probs)):
        if len(cands) > 0:
            plt.scatter([tt]*len(cands), cands, c=probs, cmap='viridis', vmin=0, vmax=1, marker='o', s=30, alpha=0.6, edgecolors='none')
    # Plot smoothed pitch track
    plt.plot(time_series, f0_smoothed, color='red', linewidth=2, label='Smoothed Pitch')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Pitch Candidates and Smoothed Track (PYIN + HMM)')
    plt.ylim(0, 2000)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot observation matrix as a heatmap
    plt.figure(figsize=(12, 6))
    plt.imshow(obs_matrix.T, aspect='auto', origin='lower', interpolation='nearest', cmap='magma')
    plt.colorbar(label='Observation Probability')
    plt.xlabel('Frame Index')
    plt.ylabel('State Index (0: voiced bins, n_bins: unvoiced bins)')
    plt.title('Observation Matrix (PYIN HMM)')
    plt.tight_layout()
    plt.show()

    # Plot Viterbi trellis (log-probability matrix)
    plt.figure(figsize=(12, 6))
    plt.imshow(viterbi.T, aspect='auto', origin='lower', interpolation='nearest', cmap='Blues')
    plt.colorbar(label='Log Viterbi Probability')
    plt.xlabel('Frame Index')
    plt.ylabel('State Index')
    plt.title('Viterbi Trellis (Dynamic Programming Table)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

