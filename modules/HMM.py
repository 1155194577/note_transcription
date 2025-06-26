import numpy as np
class HMM:
    def __init__(self, A, B, pi):
        """
        Initialize the HMM.
        Args:
            A: (n_states, n_states) transition matrix
            B: (n_states, n_obs) emission matrix
            pi: (n_states,) initial state distribution
        """
        self.A = np.array(A)
        self.B = np.array(B)
        self.pi = np.array(pi)
        self.n_states = self.A.shape[0]
        self.n_obs = self.B.shape[1]

    def forward(self, obs_seq):
        """
        Forward algorithm: compute probability of the observation sequence.
        Returns:
            alpha: (T, n_states) forward probabilities
            prob: total probability of the sequence
        """
        T = len(obs_seq)
        alpha = np.zeros((T, self.n_states))
        # Initialization
        alpha[0] = self.pi * self.B[:, obs_seq[0]]
        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, obs_seq[t]]
        prob = np.sum(alpha[-1])
        return alpha, prob

    def backward(self, obs_seq):
        """
        Backward algorithm: compute backward probabilities.
        Returns:
            beta: (T, n_states) backward probabilities
        """
        T = len(obs_seq)
        beta = np.zeros((T, self.n_states))
        # Initialization
        beta[-1] = 1
        # Recursion
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, obs_seq[t+1]] * beta[t+1])
        return beta

    def viterbi(self, obs_seq):
        """
        Viterbi algorithm: find the most likely state sequence.
        Returns:
            path: list of most likely states
            delta: (T, n_states) max probabilities
            psi: (T, n_states) backpointers
        """
        T = len(obs_seq)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        # Initialization
        delta[0] = self.pi * self.B[:, obs_seq[0]]
        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                seq_probs = delta[t-1] * self.A[:, j]
                psi[t, j] = np.argmax(seq_probs)
                delta[t, j] = np.max(seq_probs) * self.B[j, obs_seq[t]]
        # Backtrack
        path = np.zeros(T, dtype=int)
        path[-1] = np.argmax(delta[-1])
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        return path.tolist(), delta, psi

    def sample(self, T, random_state=None):
        """
        Generate a random state and observation sequence of length T.
        Returns:
            states: list of state indices
            obs: list of observation indices
        """
        rng = np.random.default_rng(random_state)
        states = [rng.choice(self.n_states, p=self.pi)]
        obs = [rng.choice(self.n_obs, p=self.B[states[0]])]
        for t in range(1, T):
            states.append(rng.choice(self.n_states, p=self.A[states[-1]]))
            obs.append(rng.choice(self.n_obs, p=self.B[states[-1]]))
        return states, obs

# Example usage (remove or modify as needed):
if __name__ == "__main__":
    # Example: 2 states, 3 observations
    A = [[0.7, 0.3], [0.4, 0.6]]
    B = [[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]]
    pi = [0.6, 0.4]
    hmm = HMM(A, B, pi)
    obs_seq = [0, 1, 2]
    print('Forward probability:', hmm.forward(obs_seq)[1])
    print('Viterbi path:', hmm.viterbi(obs_seq)[0])
    print('Sampled sequence:', hmm.sample(5))
