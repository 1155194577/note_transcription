import numpy as np
from util.audio import midi_to_hz

# Transfer continuous pitches to discrete notes
class NoteSegmentation():
    def __init__(self, 
                 pitch_estimates,
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
        self.pitch_estimates = pitch_estimates # smoothed f0 candidates, shape: (n_frames, )

        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_size = hop_size if hop_size is not None else frame_length // 4

        self.states_type = ["Attack", "Stable", "Silence"]

        self.min_midi = min_midi
        self.max_midi = max_midi
        self.steps_per_semitone = steps_per_semitone # 3 steps per semitone, (60, 60.333, 60.666, ...)
        self.n_pitches = n_pitches # number of discrete pitches (207), but has 9-semitones interval up and down

        self.v = v # prior prob for voiced frame
        self.obs_tau = obs_tau # observation tau
        self.sigma_attack = sigma_attack # semitones
        self.sigma_stable = sigma_stable # semitones

    def _build_state(self):
        states = []

        midi_buffer = (self.n_pitches - (self.max_midi - self.min_midi + 1) * 3) / self.steps_per_semitone / 2
        min_midi_with_buffer = self.min_midi - midi_buffer # MIDI 26
        max_midi_with_buffer = self.max_midi + midi_buffer # MIDI 94

        # 207 pitches
        midi_vals = np.round(np.arange(min_midi_with_buffer, max_midi_with_buffer + 1 / self.steps_per_semitone, 1 / self.steps_per_semitone), 3)

        print(len(midi_vals))

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

                if curr_state == 0 and next_state == 1:
                    trans_probs[i, j] = 1 - p_attack
                elif curr_state == 1 and next_state == 2:
                    trans_probs[i, j] = 1 - p_stable
                elif curr_state == next_state:
                    if curr_state == 0:
                        trans_probs[i, j] = p_attack
                    elif curr_state == 1:
                        trans_probs[i, j] = p_stable
                    elif curr_state == 2:
                        trans_probs[i, j] = p_silence
                else:
                    # Check if the transition is valid based on MIDI interval
                    interval = abs(next_midi - curr_midi)
                    if min_interval <= interval <= max_interval:
                        # new note
                        if curr_state == 2 and next_state == 0:
                            trans_probs[i, j] = (1 - p_silence) / (1 - interval)

        return trans_probs

    def _build_observation_matrix(self):
        # frames * states
        pass

    def _viterbi_decoding(self):
        trans_probs = self._build_transition_matrix()
        obs_probs = self._build_observation_matrix()
        pass