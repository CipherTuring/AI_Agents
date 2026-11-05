"""Tabular Q-learning over a discretized observation space."""
import numpy as np

from agents.schedules import linear_epsilon


class QLearningAgent:
    def __init__(self, n_actions, discretizer, seed, alpha, gamma,
                 eps_start, eps_end, eps_decay_steps):
        self.n_actions = n_actions
        self.discretizer = discretizer
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.rng = np.random.default_rng(seed)
        # One row of action values per discrete state.
        self.q = np.zeros(discretizer.bins + (n_actions,))

    def epsilon(self, step):
        return linear_epsilon(step, self.eps_start, self.eps_end, self.eps_decay_steps)

    def act(self, obs, step):
        """Epsilon-greedy action used while training."""
        if self.rng.random() < self.epsilon(step):
            return int(self.rng.integers(self.n_actions))
        values = self.q[self.discretizer(obs)]
        # Random tie-breaking: np.argmax alone would always pick action 0 in
        # states that were never visited.
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))

    def greedy_action(self, obs):
        """Deterministic action used for evaluation (epsilon = 0)."""
        return int(np.argmax(self.q[self.discretizer(obs)]))

    def observe(self, obs, action, reward, next_obs, terminated):
        """Applies one Q-learning update.

        Only `terminated` removes the bootstrap term. A time-limit truncation
        does not mean the state has no future value, so the training loop
        does not pass it here.
        """
        state = self.discretizer(obs) + (action,)
        next_value = 0.0 if terminated else self.q[self.discretizer(next_obs)].max()
        target = reward + self.gamma * next_value
        self.q[state] += self.alpha * (target - self.q[state])

    def diagnostics(self):
        """The table has no training loss to report."""
        return {}

    def save(self, path):
        np.savez_compressed(path / "q_table.npz", q=self.q.astype(np.float32))

    def load(self, path):
        self.q = np.load(path / "q_table.npz")["q"].astype(np.float64)
