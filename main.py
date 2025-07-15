import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import mido
import librosa
from modules.Pyin import PyinPitchExtractor
from modules.PitchSmoother import PitchSmoother
from modules.NoteSegmentation import NoteSegmentation
from util import audio
import csv
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

def pitch_extraction(signal, sr=22050, frame_length=2048, hop_size=512):
    # Use PyinPitchExtractor to extract pitch candidates and probabilities
    extractor = PyinPitchExtractor(sample_rate=sr)
    f0_candidates, f0_probs, time_series = extractor.signal_pitch_extraction(signal)
    amplitude = np.array([
        np.sqrt(np.mean(signal[i:i + frame_length] ** 2))
        for i in range(0, len(signal), hop_size)
    ])
    rise = amplitude[1:] / amplitude[:-1]
    index = np.where(rise > 1 / 0.6)[0] + 1
    print("index[:100]: ", index[:100])
    return f0_candidates, f0_probs, time_series, amplitude

def hmm_smoothing(f0_candidates, f0_probs, sr=22050):
    # Use PitchSmoother to smooth the pitch candidates
    pitch_smoother = PitchSmoother(f0_candidates, f0_probs, sample_rate=sr)
    obs_matrix = pitch_smoother._build_observation_matrix()
    print("obs_matrix: ", obs_matrix)
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
    f0_smoothed = pitch_smoother._state_to_freq(beststates)
    return f0_smoothed, obs_matrix, pitch_smoother, viterbi

def note_segmentation(f0_smoothed, amplitude):
    note_segmenter = NoteSegmentation()
    # note_segmenter.amplitude = amplitude  # Optional amplitude data for onset refinement
    # energy_mask = audio.compute_energy_mask(amplitude)
    # print("energy False: ", np.where(energy_mask == False)[0])

    best_states = note_segmenter._viterbi_decoding(f0_smoothed)

    # print("best_states[~energy_mask]: ", best_states[~energy_mask])

    states = note_segmenter._build_state()
    # silence_state_idx = [i for i, (_, state_type) in enumerate(states) if state_type == 2][0]
    # best_states[~energy_mask] = silence_state_idx

    print("best_states: ", best_states.shape)

    # notes = note_segmenter.post_process(best_states, amplitude=amplitude)
    return best_states, states

def main():
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)

    # Generate a 440 Hz sine wave signal
    # sr = 22050
    # T = 1.0  # seconds
    # t = np.linspace(0, T, int(sr * T), endpoint=False)
    # signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    signal, sr = librosa.load('./analysis/birds_of_feather.wav')
    print(f"Signal shape: {signal.shape}, Sample rate: {sr}")

    f0_candidates, f0_probs, time_series, amplitude = pitch_extraction(signal, sr=sr)
    print('Candidates:', f0_candidates)
    print('Probabilities:', f0_probs)
    print("time_series[:10]: ", time_series[:10])

    f0_smoothed, obs_matrix, pitch_smoother, viterbi = hmm_smoothing(f0_candidates, f0_probs, sr=sr)
    # pitch_smoother = PitchSmoother(f0_candidates, f0_probs, sample_rate=sr)
    # obs_matrix = pitch_smoother._build_observation_matrix()
    # print("obs_matrix: ", obs_matrix)
    # beststates = pitch_smoother._viterbi_decoding()
    print('Smoothed:', f0_smoothed)
    # f0_smoothed = pitch_smoother._state_to_freq(beststates, obs_matrix=obs_matrix)
    f0_smoothed_librosa, voiced_prob, voiced_flag = librosa.pyin(signal, fmin=65, fmax=2093, sr=sr)
    print("f0_smoothed_librosa: ", f0_smoothed_librosa)
    # print("shape: ", f0_smoothed.shape)

    # Compute similarity between f0_smoothed and f0_smoothed_librosa

    # Ensure both arrays are the same length and valid (ignore NaNs in librosa output)
    min_len = min(len(f0_smoothed), len(f0_smoothed_librosa))
    f0_custom = f0_smoothed[:min_len]
    f0_librosa = f0_smoothed_librosa[:min_len]

    # Mask out frames where either is nan or zero
    mask = (~np.isnan(f0_custom)) & (~np.isnan(f0_librosa)) & (f0_custom > 0) & (f0_librosa > 0)
    if np.sum(mask) > 0:
        corr, _ = pearsonr(f0_custom[mask], f0_librosa[mask])
        print(f"Pearson correlation between f0_smoothed and f0_smoothed_librosa: {corr:.3f}")
    else:
        print("No valid frames for correlation.")

    # Plot both pitch tracks for visual comparison
    plt.figure(figsize=(12, 6))
    plt.plot(time_series[:min_len], f0_custom, label='f0_smoothed (Custom HMM)', color='red', linewidth=2)
    plt.plot(time_series[:min_len], f0_librosa, label='f0_smoothed_librosa (librosa.pyin)', color='blue', linewidth=2, alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Comparison of Smoothed Pitch Tracks')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Ensure both arrays have the same length for plotting
    # min_len = min(len(time_series), len(f0_smoothed), len(f0_smoothed_librosa))
    # plt.figure(figsize=(12, 6))
    # plt.plot(time_series[:min_len], f0_smoothed[:min_len], label='f0_smoothed (Custom HMM)', color='red', linewidth=2)
    # plt.plot(time_series[:min_len], f0_smoothed_librosa[:min_len], label='f0_smoothed_librosa (librosa.pyin)', color='blue', linewidth=2, alpha=0.7)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    # plt.title('Comparison of Smoothed Pitch Tracks')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    min_len = min(len(time_series), len(f0_smoothed))
    plt.figure(figsize=(12, 6))
    plt.plot(time_series[:min_len], f0_smoothed[:min_len], color='red', linewidth=2, label='Smoothed Pitch')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Smoothed Pitch Track (f0_smoothed)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Note Segmentation ---
    post_processed, states = note_segmentation(f0_smoothed, amplitude)
    # print('Note states:', post_processed)

    notes = []
    frame_time = 512 / 22050
    current_note = None

    for t, state_index in enumerate(post_processed):
        midi_pitch, state_type = states[state_index]

        if state_type == 2:
            if current_note:
                current_note['end_frame'] = t - 1
                current_note['end_time'] = t * frame_time
                if current_note['end_frame'] >= current_note['start_frame']:
                    notes.append(current_note)
                current_note = None
            continue
            
        # Attack state starts new note -> onset
        if state_type == 0:
            if current_note:
                current_note['end_frame'] = t - 1
                current_note['end_time'] = t * frame_time
                if current_note['end_frame'] >= current_note['start_frame']:
                    notes.append(current_note)
            current_note = {
                'midi_pitch': midi_pitch,
                'start_frame': t,
                'end_frame': t,
                'start_time': t * frame_time,
                'end_time': (t + 1) * frame_time,
                'state_sequence': [state_index],
                'state_type': 0
            }
        # Stable state extends note
        elif state_type == 1:
            if current_note and current_note['midi_pitch'] == midi_pitch:
                current_note['end_frame'] = t
                current_note['end_time'] = (t + 1) * frame_time
                current_note['state_sequence'].append(state_index)
            else:
                if current_note:
                    current_note['end_frame'] = t - 1
                    current_note['end_time'] = t * frame_time
                    if current_note['end_frame'] >= current_note['start_frame']:
                        notes.append(current_note)
                current_note = {
                    'midi_pitch': midi_pitch,
                    'start_frame': t,
                    'end_frame': t,
                    'start_time': t * frame_time,
                    'end_time': (t + 1) * frame_time,
                    'state_sequence': [state_index],
                    'state_type': 1
                }
    
    # Finalize last note
    if current_note and current_note['end_frame'] >= current_note['start_frame']:
        notes.append(current_note)
    print("len(notes): ", len(notes))

    # Plotting
    # plt.figure(figsize=(12, 6))
    # # Plot candidate frequencies as a scatter plot
    # for i, (tt, cands, probs) in enumerate(zip(time_series, f0_candidates, f0_probs)):
    #     if len(cands) > 0:
    #         plt.scatter([tt]*len(cands), cands, c=probs, cmap='viridis', vmin=0, vmax=1, marker='o', s=30, alpha=0.6, edgecolors='none')
    # # Plot smoothed pitch track
    # plt.plot(time_series, f0_smoothed, color='red', linewidth=2, label='Smoothed Pitch')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    # plt.title('Pitch Candidates and Smoothed Track (PYIN + HMM)')
    # plt.ylim(0, 2000)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # # Plot observation matrix as a heatmap
    # plt.figure(figsize=(12, 6))
    # plt.imshow(obs_matrix.T, aspect='auto', origin='lower', interpolation='nearest', cmap='magma')
    # plt.colorbar(label='Observation Probability')
    # plt.xlabel('Frame Index')
    # plt.ylabel('State Index (0: voiced bins, n_bins: unvoiced bins)')
    # plt.title('Observation Matrix (PYIN HMM)')
    # plt.tight_layout()
    # plt.show()

    # # Plot Viterbi trellis (log-probability matrix)
    # plt.figure(figsize=(12, 6))
    # plt.imshow(viterbi.T, aspect='auto', origin='lower', interpolation='nearest', cmap='Blues')
    # plt.colorbar(label='Log Viterbi Probability')
    # plt.xlabel('Frame Index')
    # plt.ylabel('State Index')
    # plt.title('Viterbi Trellis (Dynamic Programming Table)')
    # plt.tight_layout()
    # plt.show()

    # Save the smoothed pitch to a MIDI file# Create a new MIDI file and track
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set tempo (default 500000 microseconds per beat = 120 BPM)
    tempo = 500000
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    # MIDI time conversion helpers
    def seconds_to_ticks(seconds, tempo, ticks_per_beat):
        beats = seconds * 1e6 / tempo
        return int(beats * ticks_per_beat)

    ticks_per_beat = mid.ticks_per_beat

    # Sort notes by start_time
    notes_sorted = sorted(notes, key=lambda n: n['start_time'])

    last_tick = 0
    count = 0
    for note in notes_sorted:
        if note['midi_pitch'] == 0:
            continue
        midi_pitch = int(round(note['midi_pitch']))
        start_tick = seconds_to_ticks(note['start_time'], tempo, ticks_per_beat)
        end_tick = seconds_to_ticks(note['end_time'], tempo, ticks_per_beat)
        duration = end_tick - start_tick

        while count < 30:
            print("start time: ", note['start_time'])
            count += 1

        # Add note_on
        track.append(mido.Message('note_on', note=midi_pitch, velocity=64, time=start_tick - last_tick))
        # Add note_off
        track.append(mido.Message('note_off', note=midi_pitch, velocity=64, time=duration))
        last_tick = end_tick

    # Save to file
    mid.save('output1.mid')

    # Compare output1.mid with Tony's MIDI file
    # tony_midi_path = './tony_output/tony_note_data.mid'
    # if os.path.exists(tony_midi_path):
    #     def extract_midi_notes(midi_path):
    #         midi = mido.MidiFile(midi_path)
    #         notes = []
    #         time = 0
    #         for msg in midi:
    #             time += msg.time
    #             if msg.type == 'note_on' and msg.velocity > 0:
    #                 notes.append({'time': time, 'pitch': msg.note})
    #         return notes

    #     output_notes = extract_midi_notes('output1.mid')
    #     tony_notes = extract_midi_notes(tony_midi_path)

    #     output_times = [n['time'] for n in output_notes]
    #     output_pitches = [n['pitch'] for n in output_notes]
    #     tony_times = [n['time'] for n in tony_notes]
    #     tony_pitches = [n['pitch'] for n in tony_notes]

    #     plt.figure(figsize=(12, 6))
    #     plt.scatter(output_times, output_pitches, label='Output MIDI Notes', color='red', marker='o')
    #     plt.scatter(tony_times, tony_pitches, label='Tony MIDI Notes', color='green', marker='x')
    #     plt.xlabel('Time (s)')
    #     plt.ylabel('MIDI Pitch')
    #     plt.title('Output MIDI vs Tony MIDI')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    #     # Optionally, compute correlation if lengths match
    #     min_len = min(len(output_pitches), len(tony_pitches))
    #     if min_len > 0:
    #         corr, _ = pearsonr(np.array(output_pitches[:min_len]), np.array(tony_pitches[:min_len]))
    #         print(f"Pearson correlation between output MIDI and Tony MIDI: {corr:.3f}")
    #         mse = mean_squared_error(output_pitches[:min_len], tony_pitches[:min_len])
    #         print(f"Mean Squared Error between output MIDI and Tony MIDI: {mse:.3f}")
    #     else:
    #         print("No valid notes for correlation.")
    # else:
    #     print("Tony MIDI file not found for comparison.")

    # Compare output1.mid and Tony's MIDI file for similarity (pitch and timing)
    if os.path.exists('output1.mid') and os.path.exists('./tony_output/tony_note_data.mid'):
        def extract_midi_notes_with_duration(midi_path):
            midi = mido.MidiFile(midi_path)
            notes = []
            abs_time = 0
            note_on_times = {}
            for msg in midi:
                abs_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_on_times[msg.note] = abs_time
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in note_on_times:
                        notes.append({
                            'pitch': msg.note,
                            'start': note_on_times[msg.note],
                            'end': abs_time,
                            'duration': abs_time - note_on_times[msg.note]
                        })
                        del note_on_times[msg.note]
            return notes

        notes1 = extract_midi_notes_with_duration('output1.mid')
        notes2 = extract_midi_notes_with_duration('./tony_output/tony_note_data.mid')

        # Match notes by order (frame-wise comparison)
        min_len = min(len(notes1), len(notes2))
        if min_len > 0:
            pitches1 = np.array([n['pitch'] for n in notes1[:min_len]])
            pitches2 = np.array([n['pitch'] for n in notes2[:min_len]])
            starts1 = np.array([n['start'] for n in notes1[:min_len]])
            starts2 = np.array([n['start'] for n in notes2[:min_len]])
            durations1 = np.array([n['duration'] for n in notes1[:min_len]])
            durations2 = np.array([n['duration'] for n in notes2[:min_len]])

            pitch_corr, _ = pearsonr(pitches1, pitches2)
            start_corr, _ = pearsonr(starts1, starts2)
            duration_corr, _ = pearsonr(durations1, durations2)
            pitch_mse = mean_squared_error(pitches1, pitches2)
            start_mse = mean_squared_error(starts1, starts2)
            duration_mse = mean_squared_error(durations1, durations2)

            print(f"Pitch Pearson correlation: {pitch_corr:.3f}")
            print(f"Start time Pearson correlation: {start_corr:.3f}")
            print(f"Duration Pearson correlation: {duration_corr:.3f}")
            print(f"Pitch MSE: {pitch_mse:.3f}")
            print(f"Start time MSE: {start_mse:.3f}")
            print(f"Duration MSE: {duration_mse:.3f}")
        else:
            print("No notes to compare for similarity.")
    else:
        print("One or both MIDI files not found for similarity comparison.")

    

if __name__ == "__main__":
    main()

