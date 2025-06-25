import numpy as np
import matplotlib.pyplot as plt


def hz_to_midi(freq):
    """Convert frequency in Hz to MIDI note number."""
    return 69 + 12 * np.log2(freq / 440.0)

def midi_to_hz(midi):
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * 2 ** ((midi - 69) / 12)


def smooth_pitch_track(candidate_freqs, candidate_probs, sample_rate, fmin, fmax, bins_per_semitone=1, voicing_prob=0.5, max_jump_semitones=2.5):
    """
    Smooth a sequence of frame-wise pitch candidates using an HMM and Viterbi decoding.

    Args:
        candidate_freqs: list of np.ndarray, each containing candidate frequencies for a frame
        candidate_probs: list of np.ndarray, each containing probabilities for the candidates in the same frame
        sample_rate: int, sample rate of the audio
        fmin: float, minimum frequency (Hz)
        fmax: float, maximum frequency (Hz)
        bins_per_semitone: int, number of bins per semitone (default 1, i.e., 100 cents)
        voicing_prob: float, prior probability of voicing (default 0.5)
        max_jump_semitones: float, maximum allowed pitch jump per frame (default 2.5)

    Returns:
        np.ndarray: Smoothed pitch track in Hz (one value per frame, 0 for unvoiced)
    """
    # 1. Discretize pitch space
    n_semitones = 12 * np.log2(fmax / fmin)
    n_bins = int(np.ceil(n_semitones * bins_per_semitone))
    midi_min = hz_to_midi(fmin)
    midi_max = hz_to_midi(fmax)
    bin_edges = np.linspace(midi_min, midi_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_freqs = midi_to_hz(bin_centers)

    n_frames = len(candidate_freqs)
    n_states = 2 * n_bins  # voiced/unvoiced for each bin

    # 2. Build observation probability matrix (frames x states)
    obs_probs = np.zeros((n_frames, n_states))
    for t in range(n_frames):
        # Map candidate freqs to bins
        if len(candidate_freqs[t]) == 0:
            obs_probs[t, n_bins:] = 1.0  # all unvoiced
            continue
        midi_cands = hz_to_midi(candidate_freqs[t])
        bin_idx = np.clip(np.digitize(midi_cands, bin_edges) - 1, 0, n_bins - 1)
        # Voiced: assign half the candidate probability to the corresponding bin
        for i, idx in enumerate(bin_idx):
            obs_probs[t, idx] += 0.5 * candidate_probs[t][i]
        # Unvoiced: half of one minus sum of candidate probabilities
        obs_probs[t, n_bins:] = 0.5 * (1 - np.sum(candidate_probs[t]))
    # Normalize each frame's obs_probs
    obs_probs = obs_probs / (obs_probs.sum(axis=1, keepdims=True) + 1e-12)

    # 3. Build transition probability matrix (states x states)
    # Voiced-voiced: triangular, allow jumps up to max_jump_semitones
    max_jump_bins = int(np.ceil(max_jump_semitones * bins_per_semitone))
    trans_probs = np.zeros((n_states, n_states))
    for i in range(n_bins):
        for j in range(n_bins):
            if abs(i - j) <= max_jump_bins:
                trans_probs[i, j] = max_jump_bins + 1 - abs(i - j)
        # Normalize
        trans_probs[i, :n_bins] /= trans_probs[i, :n_bins].sum() + 1e-12
        # Voiced-unvoiced and unvoiced-voiced: low probability
        trans_probs[i, n_bins + i] = 0.01
        trans_probs[n_bins + i, i] = 0.01
        # Unvoiced-unvoiced: high probability to stay unvoiced
        trans_probs[n_bins + i, n_bins + i] = 0.99
    # Normalize all rows
    trans_probs = trans_probs / (trans_probs.sum(axis=1, keepdims=True) + 1e-12)

    # 4. Viterbi decoding
    log_obs = np.log(obs_probs + 1e-12)
    log_trans = np.log(trans_probs + 1e-12)
    path = np.zeros((n_frames, n_states), dtype=int)
    score = np.full((n_frames, n_states), -np.inf)
    score[0] = log_obs[0]
    for t in range(1, n_frames):
        for j in range(n_states):
            prev_scores = score[t - 1] + log_trans[:, j]
            path[t, j] = np.argmax(prev_scores)
            score[t, j] = prev_scores[path[t, j]] + log_obs[t, j]
    # Backtrack
    states = np.zeros(n_frames, dtype=int)
    states[-1] = np.argmax(score[-1])
    for t in range(n_frames - 2, -1, -1):
        states[t] = path[t + 1, states[t + 1]]
    # Convert states to pitch
    pitch_track = np.zeros(n_frames)
    for t, s in enumerate(states):
        if s < n_bins:
            pitch_track[t] = bin_freqs[s]
        else:
            pitch_track[t] = 0.0  # unvoiced
    return pitch_track 

if __name__ == "__main__":
    # --- 1. Simulate candidate frequencies and probabilities for demonstration ---
    np.random.seed(42)
    n_frames = 50
    fmin, fmax = 65, 880
    sample_rate = 22050
    frame_length = 2048
    hop_length = 512

    # Generate a sine wave with a smoothly varying frequency (e.g., vibrato)
    t = np.arange(n_frames * hop_length + frame_length) / sample_rate
    # Vibrato: base frequency 220 Hz, vibrato depth 40 Hz, vibrato rate 2 Hz
    base_freq = 220
    vibrato_depth = 40
    vibrato_rate = 2
    inst_freq = base_freq + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
    phase = 2 * np.pi * np.cumsum(inst_freq) / sample_rate
    signal = np.sin(phase)

    # Compute true instantaneous frequency for each frame center
    frame_centers = np.arange(n_frames) * hop_length + frame_length // 2
    true_pitches = inst_freq[frame_centers]

    # For each frame, create candidate frequencies and probabilities
    candidate_freqs = []
    candidate_probs = []
    for f in true_pitches:
        cands = np.array([f, f * 2, f / 2])  # true, octave up, octave down
        probs = np.array([0.7, 0.2, 0.1]) * np.random.uniform(0.8, 1.2, 3)
        probs = np.clip(probs, 0, 1)
        probs /= probs.sum()
        candidate_freqs.append(cands)
        candidate_probs.append(probs)

    # --- 2. Run smoothing ---
    pitch_track = smooth_pitch_track(candidate_freqs, candidate_probs, sample_rate, fmin, fmax)

    # --- 3. Visualize pitch bins ---
    n_semitones = 12 * np.log2(fmax / fmin)
    n_bins = int(np.ceil(n_semitones))
    midi_min = hz_to_midi(fmin)
    midi_max = hz_to_midi(fmax)
    bin_edges = np.linspace(midi_min, midi_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_freqs = midi_to_hz(bin_centers)

    plt.figure(figsize=(10, 2))
    plt.plot(bin_freqs, np.zeros_like(bin_freqs), 'o')
    plt.title("Discretized Pitch Bins (Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.yticks([])
    plt.show()

    # --- 4. Visualize candidate frequencies and probabilities for a few frames ---
    plt.figure(figsize=(10, 4))
    for t in range(5):
        plt.stem(candidate_freqs[t], candidate_probs[t], basefmt=" ", label=f"Frame {t}")
    plt.title("Candidate Frequencies and Probabilities (First 5 Frames)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()

    # --- 5. Visualize observation probability matrix for first 10 frames ---
    obs_probs = np.zeros((n_frames, 2 * n_bins))
    for t in range(n_frames):
        if len(candidate_freqs[t]) == 0:
            obs_probs[t, n_bins:] = 1.0
            continue
        midi_cands = hz_to_midi(candidate_freqs[t])
        bin_idx = np.clip(np.digitize(midi_cands, bin_edges) - 1, 0, n_bins - 1)
        for i, idx in enumerate(bin_idx):
            obs_probs[t, idx] += 0.5 * candidate_probs[t][i]
        obs_probs[t, n_bins:] = 0.5 * (1 - np.sum(candidate_probs[t]))
    obs_probs = obs_probs / (obs_probs.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(12, 4))
    plt.imshow(obs_probs[:10, :n_bins], aspect='auto', origin='lower', interpolation='nearest')
    plt.colorbar(label="Observation Probability")
    plt.title("Observation Probability Matrix (First 10 Frames, Voiced States)")
    plt.xlabel("Pitch Bin")
    plt.ylabel("Frame")
    plt.show()

    # --- 6. Visualize transition probability matrix (voiced-voiced block) ---
    max_jump_semitones = 2.5
    max_jump_bins = int(np.ceil(max_jump_semitones))
    trans_probs = np.zeros((2 * n_bins, 2 * n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            if abs(i - j) <= max_jump_bins:
                trans_probs[i, j] = max_jump_bins + 1 - abs(i - j)
        trans_probs[i, :n_bins] /= trans_probs[i, :n_bins].sum() + 1e-12
        trans_probs[i, n_bins + i] = 0.01
        trans_probs[n_bins + i, i] = 0.01
        trans_probs[n_bins + i, n_bins + i] = 0.99
    trans_probs = trans_probs / (trans_probs.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(6, 5))
    plt.imshow(trans_probs[:n_bins, :n_bins], aspect='auto', origin='lower', interpolation='nearest')
    plt.colorbar(label="Transition Probability")
    plt.title("Transition Probability Matrix (Voiced-Voiced Block)")
    plt.xlabel("To Pitch Bin")
    plt.ylabel("From Pitch Bin")
    plt.show()

    # --- 7. Visualize the final smoothed pitch track ---
    plt.figure(figsize=(12, 4))
    plt.plot(true_pitches, label="True Pitch (Hz)", linestyle='--')
    plt.plot(pitch_track, label="Smoothed Pitch Track (Hz)")
    plt.title("Smoothed Pitch Track vs. True Pitch (Sine Wave Vibrato)")
    plt.xlabel("Frame")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.show() 