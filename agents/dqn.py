"""Deep Q-Network: Q-network, experience replay, target network, epsilon-greedy."""
import numpy as np
import torch
from torch import nn

from agents.schedules import linear_epsilon

LOSSES = {"mse": nn.functional.mse_loss, "huber": nn.functional.smooth_l1_loss}


class QNetwork(nn.Module):
    """MLP that maps an observation to one Q-value per action."""

    def __init__(self, obs_dim, n_actions, hidden):
        super().__init__()
        layers, size = [], obs_dim
        for width in hidden:
            layers += [nn.Linear(size, width), nn.ReLU()]
            size = width
        layers.append(nn.Linear(size, n_actions))
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        return self.layers(obs)


class ReplayBuffer:
    """Fixed-size circular store of transitions, sampled uniformly."""

    def __init__(self, capacity, obs_dim, rng):
        self.capacity = capacity
        self.rng = rng
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.position
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminated[i] = terminated
        # Once full, the oldest transition is overwritten.
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.terminated[idx]),
        )


class DQNAgent:
    def __init__(self, obs_dim, n_actions, seed, gamma, lr, hidden, buffer_size,
                 batch_size, learning_starts, train_every, target_update_every,
                 eps_start, eps_end, eps_decay_steps,
                 use_target_network=True, use_replay=True, loss="mse"):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_every = train_every
        self.target_update_every = target_update_every
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.use_target_network = use_target_network
        self.loss_fn = LOSSES[loss]
        self.rng = np.random.default_rng(seed)

        self.q_net = QNetwork(obs_dim, n_actions, hidden)
        if use_target_network:
            self.target_net = QNetwork(obs_dim, n_actions, hidden)
            self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            # Ablation: targets come from the network being trained.
            self.target_net = self.q_net
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # Ablation: with no replay the buffer only holds the latest
        # batch_size transitions, so samples stay recent and correlated
        # while the number and size of gradient updates do not change.
        capacity = buffer_size if use_replay else batch_size
        self.buffer = ReplayBuffer(capacity, obs_dim, self.rng)

        self.steps = 0
        self.losses, self.q_means = [], []

    def epsilon(self, step):
        return linear_epsilon(step, self.eps_start, self.eps_end, self.eps_decay_steps)

    def act(self, obs, step):
        """Epsilon-greedy action used while training."""
        if self.rng.random() < self.epsilon(step):
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(obs)

    def greedy_action(self, obs):
        """Deterministic action used for evaluation (epsilon = 0)."""
        with torch.no_grad():
            values = self.q_net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(values.argmax(dim=1).item())

    def compute_targets(self, rewards, next_obs, terminated):
        """Bellman targets: r + gamma * max_a Q_target(s', a).

        Kept in one method because this is the only line a Double DQN
        extension would change.
        """
        with torch.no_grad():
            next_values = self.target_net(next_obs).max(dim=1).values
        return rewards + self.gamma * (1.0 - terminated) * next_values

    def update(self):
        """Takes one gradient step on a sampled batch."""
        obs, actions, rewards, next_obs, terminated = self.buffer.sample(self.batch_size)
        targets = self.compute_targets(rewards, next_obs, terminated)
        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())
        self.q_means.append(q_values.mean().item())

    def observe(self, obs, action, reward, next_obs, terminated):
        """Stores a transition, then trains and syncs when they are due.

        As in the tabular agent, only `terminated` cuts the bootstrap.
        """
        self.buffer.add(obs, action, reward, next_obs, terminated)
        self.steps += 1

        if self.steps >= self.learning_starts and self.steps % self.train_every == 0:
            self.update()
        if self.use_target_network and self.steps % self.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def diagnostics(self):
        """Mean loss and Q-value since the previous call, for the eval log."""
        if not self.losses:
            return {"loss": float("nan"), "mean_q": float("nan")}
        stats = {"loss": float(np.mean(self.losses)),
                 "mean_q": float(np.mean(self.q_means))}
        self.losses, self.q_means = [], []
        return stats

    def save(self, path):
        torch.save(self.q_net.state_dict(), path / "q_network.pt")

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path / "q_network.pt"))
