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
                 sigma_stable = 0.9):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_size = hop_size if hop_size is not None else frame_length // 4
        self.frame_time = self.frame_length / self.sample_rate # seconds

        self.states_type = ["Attack", "Stable", "Silence"] # 0: attack, 1: stable, 2: silence

        self.min_midi = min_midi
        self.max_midi = max_midi
        self.steps_per_semitone = steps_per_semitone # 3 steps per semitone, (60, 60.333, 60.666, ...)
        self.n_pitches = n_pitches # number of discrete pitches (207), has 9-semitones buffer up and down

        self.v = v # prior prob for voiced frame
        self.obs_tau = obs_tau # observation tau
        self.sigma_attack = sigma_attack # semitones
        self.sigma_stable = sigma_stable # semitones

    def _build_state(self):
        states = []

        # MIDI buffer to ensure that the range of MIDI values is sufficient
        # to cover the range of pitches, considering the steps per semitone.
        midi_buffer = ((self.n_pitches - (self.max_midi - self.min_midi + 1) * self.steps_per_semitone) / self.steps_per_semitone) / 2
        min_midi_with_buffer = self.min_midi - midi_buffer # MIDI 26
        max_midi_with_buffer = self.max_midi + midi_buffer # MIDI 94

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
                    trans_probs[i, j] = 1 - p_attack
                elif curr_state == 1 and next_state == 2: # stable -> silence
                    trans_probs[i, j] = 1 - p_stable
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
            
            if pitch <= 0:
                # unvoiced
                obs_probs[t, :] = (1 - self.v) / n_states
                continue
            
            h = np.zeros(n_states)

            for i, (midi, state_type) in enumerate(states):
                if state_type == 2:
                    continue
                sigma = self.sigma_attack if state_type == 0 else self.sigma_stable
                # Gaussian observation model
                exponent = -((pitch - midi) ** 2) / (2 * sigma ** 2)
                # Calculate the observation probability for each state
                gaussian = np.exp(exponent) / (sigma * np.sqrt(2 * np.pi))
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
    
    def post_process(self, states):
        pass