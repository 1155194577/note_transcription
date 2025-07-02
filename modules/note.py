import numpy as np
import matplotlib.pyplot as plt
from modules.NoteSegmentation import NoteSegmentation

# in project root directory use : python3 -m modules.note
# Demo/example usage for plotting everything in NoteSegmentation
if __name__ == "__main__":
    # Example: generate a fake pitch track
    n_frames = 100
    pitch_estimates = 60 + 2 * np.sin(np.linspace(0, 6 * np.pi, n_frames))  # MIDI numbers
    note_seg = NoteSegmentation(pitch_estimates)

    # Plot transition matrix
    trans_matrix = note_seg._build_transition_matrix()
    plt.figure(figsize=(10, 8))
    plt.imshow(trans_matrix, aspect='auto', origin='lower', interpolation='nearest', cmap='magma')
    plt.colorbar(label='Transition Probability')
    plt.xlabel('Next State Index')
    plt.ylabel('Current State Index')
    plt.title('Transition Matrix (NoteSegmentation)')
    plt.tight_layout()
    plt.show()

    # Plot observation matrix if implemented
    try:
        obs_matrix = note_seg._build_observation_matrix()
        if obs_matrix is not None:
            plt.figure(figsize=(10, 8))
            plt.imshow(obs_matrix, aspect='auto', origin='lower', interpolation='nearest', cmap='viridis')
            plt.colorbar(label='Observation Probability')
            plt.xlabel('State Index')
            plt.ylabel('Frame Index')
            plt.title('Observation Matrix (NoteSegmentation)')
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print("Observation matrix plotting skipped (not implemented):", e)

    # Plot Viterbi trellis if implemented
    try:
        # This will fail unless _viterbi_decoding is implemented
        viterbi = note_seg._viterbi_decoding()
        if viterbi is not None:
            plt.figure(figsize=(10, 8))
            plt.imshow(viterbi, aspect='auto', origin='lower', interpolation='nearest', cmap='Blues')
            plt.colorbar(label='Viterbi Value')
            plt.xlabel('State Index')
            plt.ylabel('Frame Index')
            plt.title('Viterbi Trellis (NoteSegmentation)')
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print("Viterbi trellis plotting skipped (not implemented):", e)

    # Plot pitch track
    plt.figure(figsize=(12, 4))
    plt.plot(pitch_estimates, label='Pitch Estimates (MIDI)')
    plt.xlabel('Frame Index')
    plt.ylabel('MIDI Pitch')
    plt.title('Input Pitch Track')
    plt.legend()
    plt.tight_layout()
    plt.show()

