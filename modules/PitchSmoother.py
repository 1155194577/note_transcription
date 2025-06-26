from modules.HMM import HMM
import numpy as np

class PitchSmoother(HMM):
    """
    PitchSmoother: HMM for pitch smoothing as described in PYIN (see pyin.md).

    - A: Transition matrix (pitch/voicing transitions, e.g., favoring small pitch changes and voicing continuity)
    - B: Emission matrix (probabilities of observed pitch candidates given hidden pitch/voicing state)
    - pi: Initial state distribution (probability of starting in each pitch/voicing state)

    Students: Fill in the constructor and the method to set up A, B, and pi based on the PYIN algorithm.
    """
    def __init__(self, *args, **kwargs):
        """
        TODO: Initialize PitchSmoother with appropriate A, B, pi for pitch smoothing.
        You may want to accept candidate probabilities, pitch bins, and voicing info as arguments.
        """
        # TODO: Set up A, B, pi for pitch/voicing smoothing
        # Example: super().__init__(A, B, pi)
        pass

    def setup_hmm_parameters(self, *args, **kwargs):
        """
        TODO: Implement logic to construct A, B, pi for the pitch smoothing HMM.
        See pyin.md for details on how to set up:
        - A: favoring smooth pitch tracks and few voicing changes
        - B: using candidate probabilities as emission likelihoods
        - pi: initial state distribution (e.g., uniform or based on prior knowledge)
        """
        pass 