import numpy as np
from util import audio
import matplotlib.pyplot as plt

# Transfer continuous pitches to discrete notes
class NoteSegmentation():
    def __init__(self, 
                 sample_rate = 22050,
                 frame_length = 2048,
                 hop_size = None,
                 min_midi = 35,
                 max_midi = 85,
                 steps_per_semitone = 3,
                 n_pitches = 207,
                 obs_tau = 0.1,
                 v = 0.5,
                 sigma_attack = 5,
                 sigma_stable = 0.9,
                 min_duration = 0.1):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_size = hop_size if hop_size is not None else frame_length // 4
        self.frame_time = self.hop_size / self.sample_rate # seconds

        self.states_type = ["Attack", "Stable", "Silence"] # 0: attack, 1: stable, 2: silence

        self.min_midi = min_midi
        self.max_midi = max_midi
        self.steps_per_semitone = steps_per_semitone # 3 steps per semitone, (60, 60.333, 60.666, ...)
        self.n_pitches = n_pitches # number of discrete pitches (207), has 9-semitones buffer up and down

        self.v = v # prior prob for voiced frame
        self.obs_tau = obs_tau # observation tau
        self.sigma_attack = sigma_attack # semitones
        self.sigma_stable = sigma_stable # semitones
        self.min_duration = min_duration # seconds, minimum duration of a note to be considered valid

    def _build_state(self):
        states = []

        # MIDI buffer to ensure that the range of MIDI values is sufficient
        # to cover the range of pitches, considering the steps per semitone.
        midi_buffer = ((self.n_pitches - (self.max_midi - self.min_midi + 1) * self.steps_per_semitone) / self.steps_per_semitone) / 2
        min_midi_with_buffer = self.min_midi # MIDI 35
        max_midi_with_buffer = self.max_midi + 2 * midi_buffer # MIDI 103

        # 207 pitches
        midi_vals = np.round(np.arange(min_midi_with_buffer, max_midi_with_buffer + 1, 1 / self.steps_per_semitone), 3)

        for midi in midi_vals:
            for state_type in range(3):
                states.append((midi, state_type))

        return states

    def _build_transition_matrix(self):
        # curr_states * next_states
        states = self._build_state()
        n_states = len(states)
        trans_probs = np.zeros((n_states, n_states))

        p_attack = 0.9
        p_stable = 0.99
        p_silence = 0.999
        min_interval = 2 / 3 # semitones
        max_interval = 13 # semitones

        for i in range(n_states):
            curr_midi, curr_state = states[i]
            for j in range(n_states):
                next_midi, next_state = states[j]

                if curr_state == 0 and next_state == 1: # attack -> stable
                    if curr_midi == next_midi:
                        trans_probs[i, j] = 1 - p_attack
                    else: trans_probs[i, j] = 0
                elif curr_state == 1 and next_state == 2: # stable -> silence
                    if curr_midi == next_midi:
                        trans_probs[i, j] = 1 - p_stable
                    else: trans_probs[i, j] - 0
                elif curr_state == next_state: # self transition
                    if curr_state == 0:
                        trans_probs[i, j] = p_attack
                    elif curr_state == 1:
                        trans_probs[i, j] = p_stable
                    elif curr_state == 2:
                        trans_probs[i, j] = p_silence
                elif curr_state == 2 and next_state == 0: # silence -> attack
                    interval = abs(next_midi - curr_midi)
                    # Check if the transition is valid based on MIDI interval
                    if min_interval <= interval <= max_interval:
                        # new note
                        trans_probs[i, j] = (1 - p_silence) * (1 / interval)
                    else: trans_probs[i, j] = 0

        trans_probs = trans_probs / np.sum(trans_probs, axis=1, keepdims=True)  # Normalize each row to sum to 1
        return trans_probs

    def _build_observation_matrix(self, pitch_estimates):
        # frames * states
        states = self._build_state()
        n_states = len(states)
        n_frames = len(pitch_estimates)
        obs_probs = np.zeros((n_frames, n_states))

        pitches = np.round(audio.hz_to_midi(pitch_estimates), 3)

        for t in range(n_frames):
            pitch = pitches[t]
            
            if pitch <= 0 or pitch == np.nan:
                # unvoiced
                silence_states = [idx for idx, (_, st) in enumerate(states) if st == 2]
                prob = (1 - self.v) / len(silence_states)
                
                for j in range(n_states):
                    # Assign probability only to silence states
                    obs_probs[t, j] = prob if j in silence_states else 1e-12
                continue
            h = np.zeros(n_states)

            for i, (midi, state_type) in enumerate(states):
                if state_type == 2:
                    continue
                sigma = self.sigma_attack if state_type == 0 else self.sigma_stable
                # Gaussian observation model
                exponent = -((pitch - midi) ** 2) / (2 * sigma ** 2)
                # Calculate the observation probability for each state
                gaussian = np.exp(exponent) # / (sigma * np.sqrt(2 * np.pi))
                # Scale the observation probability by the observation tau
                h[i] = gaussian ** self.obs_tau
            
            z = np.sum(h)
            
            for j, (_, state_type) in enumerate(states):
                if state_type == 2:
                    obs_probs[t, j] = 1e-12
                else:
                    obs_probs[t, j] = self.v * (h[j] / z) if z > 1e-12 else 1e-12

        return obs_probs

    def _viterbi_decoding(self, pitch_estimates):
        log_trans_matrix = np.log(self._build_transition_matrix() + 1e-12)
        log_obs_matrix = np.log(self._build_observation_matrix(pitch_estimates) + 1e-12)
        n_frames, n_states = log_obs_matrix.shape

        viterbi = np.full((n_frames, n_states), -np.inf)
        backpathpointer = np.zeros((n_frames, n_states), dtype=int)
        beststates = np.zeros(n_frames, dtype=int)

        viterbi[0, :] = log_obs_matrix[0, :]

        # --- Recursion ---
        for t in range(1, n_frames):
            for j in range(n_states):
                trans_probs = viterbi[t - 1, :] + log_trans_matrix[:, j]
                backpathpointer[t, j] = np.argmax(trans_probs)
                viterbi[t, j] = np.max(trans_probs) + log_obs_matrix[t, j]

        # --- Termination ---
        beststates[-1] = np.argmax(viterbi[-1, :])
        for t in range(n_frames - 2, -1, -1):
            beststates[t] = backpathpointer[t + 1, beststates[t + 1]]

        print(beststates)

        return beststates

    def post_process(self, state_sequence, amplitude=None):
        # state_sequence: (n_frames, )
        # self.amplitude = amplitude  # Optional amplitude data for onset refinement
        states = self._build_state()
        notes = []
        current_note = None

        amplitude = np.clip(amplitude, 1e-12, None)  # Avoid division by zero

        window_ms = 0.046  # From paper's 2048-sample window at 22050Hz

        # First pass: Create raw note segments
        for t, state_idx in enumerate(state_sequence):
            midi_pitch, state_type = states[state_idx]
            
            # Silence state handling
            if state_type == 2:
                if current_note:
                    current_note['end_frame'] = t - 1
                    current_note['end_time'] = t * self.frame_time
                    if current_note['end_frame'] >= current_note['start_frame']:
                        notes.append(current_note)
                    current_note = None
                continue
            
            # Attack state starts new note -> onset
            if state_type == 0:
                if current_note:
                    current_note['end_frame'] = t - 1
                    current_note['end_time'] = t * self.frame_time
                    if current_note['end_frame'] >= current_note['start_frame']:
                        notes.append(current_note)
                current_note = {
                    'midi_pitch': midi_pitch,
                    'start_frame': t,
                    'end_frame': t,
                    'start_time': t * self.frame_time,
                    'end_time': (t + 1) * self.frame_time,
                    'state_sequence': [state_idx],
                    'state_type': 0
                }
            # Stable state extends note
            elif state_type == 1:
                if current_note and current_note['midi_pitch'] == midi_pitch:
                    current_note['end_frame'] = t
                    current_note['end_time'] = (t + 1) * self.frame_time
                    current_note['state_sequence'].append(state_idx)
                else:
                    if current_note:
                        current_note['end_frame'] = t - 1
                        current_note['end_time'] = t * self.frame_time
                        if current_note['end_frame'] >= current_note['start_frame']:
                            notes.append(current_note)
                    current_note = {
                        'midi_pitch': midi_pitch,
                        'start_frame': t,
                        'end_frame': t,
                        'start_time': t * self.frame_time,
                        'end_time': (t + 1) * self.frame_time,
                        'state_sequence': [state_idx],
                        'state_type': 1
                    }
        
        # Finalize last note
        if current_note and current_note['end_frame'] >= current_note['start_frame']:
            notes.append(current_note)
        
        # Refine onsets
        for i, note in enumerate(notes):
            max_rise = 0
            best_j = None

            search_start = max(1, int(round(note['start_time'] - window_ms) / self.frame_time))
            search_end = min(len(amplitude) - 1, int(round((note['start_time'] + window_ms) / self.frame_time)))
            
            for j in range(search_start, search_end):
                if amplitude[j - 1] < 1e-12 or amplitude[j + 1] < 1e-12:
                    continue
                if amplitude[j - 1] > 1e-12:
                    rise = amplitude[j + 1] / amplitude[j - 1]
                    if rise > 1 / 0.6 and rise > max_rise:
                        max_rise = rise
                        best_j = j
            
            if best_j is not None:
                print(best_j)

                new_start = best_j * self.frame_time
                if i > 0 and new_start < notes[i - 1]['end_time']:
                    new_start = notes[i - 1]['end_time']

                note['start_time'] = new_start

        # Third pass: Merge adjacent notes
        notes = self.merge_notes(notes)
        
        # Add metadata and clean up
        for note in notes:
            # note['start_time'] = note['start_frame'] * self.frame_time
            # note['end_time'] = (note['end_frame'] + 1) * self.frame_time
            note['duration'] = note['end_time'] - note['start_time']
            note['frequency'] = audio.midi_to_hz(note['midi_pitch'])
            note['confidence'] = len(note['state_sequence']) / (note['end_frame'] - note['start_frame'] + 1)
            del note['state_sequence']

        for i, note in enumerate(notes):
            if note['midi_pitch'] == 0:# or note['duration'] < self.min_duration:
                notes.remove(notes[i])

        # # Filter out notes that are too short
        notes = [note for note in notes if (note['end_frame'] - note['start_frame'] + 1) * self.frame_time >= self.min_duration]

        return notes
        
    def merge_notes(self, notes):
        merged = []
        for note in notes:
            if not merged:
                merged.append(note)
                continue
            last = merged[-1]
            gap = abs(note['start_time'] - last['end_time'])
            if (abs(note['midi_pitch'] - last['midi_pitch']) <= 1.5 and 
                gap <= 9.5 * self.frame_time):
                last['end_time'] = note['start_time']
                last['end_frame'] = note['start_frame']
                last['state_sequence'].extend(note['state_sequence'])
            else:
                if note['start_time'] < last['end_time']:
                    last['end_time'] = note['start_time']
                    last['end_frame'] = note['start_frame']
                merged.append(note)
        return merged

    # def refine_onset(self, start_frame):
    #     """
    #     Robust onset refinement with:
    #     - Full None/edge-case handling
    #     - Exact paper-specified amplitude rise ratio
    #     - Frame-accurate timing
    #     """
    #     # Safeguard 1: Check amplitude exists and has enough samples
    #     if (not hasattr(self, 'amplitude') or 
    #         (self.amplitude is None) or 
    #         (len(self.amplitude) < 3)):
    #         print("Amplitude data not available for onset refinement, using original start frame.")
    #         return start_frame  # Fallback to original if no amplitude data
        
    #     # Safeguard 2: Validate start_frame within bounds
    #     start_frame = int(np.clip(start_frame, 1, len(self.amplitude)-2))
        
    #     # Convert paper's ≈46ms analysis window to frames
    #     window_ms = 46  # From paper's 2048-sample window at 22050Hz
    #     window_frames = int(round(window_ms / (self.frame_time * 1000)))
    #     search_start = max(1, start_frame - window_frames)
    #     search_end = min(len(self.amplitude)-1, start_frame + window_frames)
        
    #     max_rise = 0
    #     best_idx = start_frame
        
    #     # Paper's exact rise ratio: r = a_{i+1}/a_{i-1}
    #     for i in range(search_start, search_end):
    #         with np.errstate(divide='ignore', invalid='ignore'):
    #             if self.amplitude[i-1] > 0:
    #                 rise = self.amplitude[i+1] / self.amplitude[i-1]
    #                 # Paper's condition: if (1/r < s) where s ≥ 0.6
    #                 if rise > (1/0.6) and rise > max_rise:  # ≡ 1/r < 0.6
    #                     max_rise = rise
    #                     best_idx = i
    
    #     return best_idx