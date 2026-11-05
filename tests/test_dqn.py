import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch

import experiments.train as train_module
from agents.dqn import DQNAgent, ReplayBuffer
from experiments.train import run

PARAMS = {
    "gamma": 0.9, "lr": 1e-3, "hidden": [16], "buffer_size": 100,
    "batch_size": 8, "learning_starts": 8, "train_every": 1,
    "target_update_every": 10, "eps_start": 1.0, "eps_end": 0.05,
    "eps_decay_steps": 100,
}


def make_agent(**overrides):
    torch.manual_seed(0)
    return DQNAgent(4, 2, seed=0, **(PARAMS | overrides))


def feed(agent, n):
    for i in range(n):
        obs = np.full(4, i % 5, dtype=np.float32)
        agent.observe(obs, i % 2, 1.0, obs + 1, False)


def test_buffer_overwrites_oldest_when_full():
    buffer = ReplayBuffer(3, 1, np.random.default_rng(0))
    for i in range(5):
        buffer.add([i], 0, float(i), [i + 1], False)
    assert buffer.size == 3
    assert sorted(buffer.rewards) == [2.0, 3.0, 4.0]


def test_buffer_sample_shapes_and_contents():
    buffer = ReplayBuffer(10, 2, np.random.default_rng(0))
    for i in range(4):
        buffer.add([i, i], 1, float(i), [i, i], True)
    obs, actions, rewards, next_obs, terminated = buffer.sample(32)
    assert obs.shape == (32, 2) and next_obs.shape == (32, 2)
    assert actions.shape == rewards.shape == terminated.shape == (32,)
    assert rewards.max() <= 3.0  # never reads the empty slots


def test_target_bootstraps_unless_terminated():
    agent = make_agent()
    next_obs = torch.zeros(2, 4)
    best = agent.target_net(next_obs).max(dim=1).values.detach()
    targets = agent.compute_targets(
        torch.tensor([1.0, 1.0]), next_obs, torch.tensor([0.0, 1.0])
    )
    assert targets[0].item() == pytest.approx(1.0 + 0.9 * best[0].item())
    assert targets[1].item() == pytest.approx(1.0)


def test_targets_carry_no_gradient():
    agent = make_agent()
    targets = agent.compute_targets(
        torch.ones(2), torch.zeros(2, 4), torch.zeros(2)
    )
    assert not targets.requires_grad


def weights(net):
    return torch.cat([p.detach().flatten().clone() for p in net.parameters()])


def test_target_network_only_changes_at_sync():
    agent = make_agent()
    before = weights(agent.target_net)
    feed(agent, 9)  # trains from step 8, sync is due at step 10
    assert not torch.equal(weights(agent.q_net), before)
    assert torch.equal(weights(agent.target_net), before)
    feed(agent, 1)
    assert torch.equal(weights(agent.target_net), weights(agent.q_net))


def test_no_target_ablation_uses_the_online_network():
    agent = make_agent(use_target_network=False)
    assert agent.target_net is agent.q_net


def test_no_replay_ablation_keeps_only_the_latest_batch():
    agent = make_agent(use_replay=False)
    assert agent.buffer.capacity == PARAMS["batch_size"]


def test_no_training_before_learning_starts():
    agent = make_agent(learning_starts=50)
    before = weights(agent.q_net)
    feed(agent, 49)
    assert torch.equal(weights(agent.q_net), before)
    feed(agent, 1)
    assert not torch.equal(weights(agent.q_net), before)


def test_save_and_load_round_trip(tmp_path):
    agent = make_agent()
    agent.save(tmp_path)
    other = DQNAgent(4, 2, seed=1, **PARAMS)
    other.load(tmp_path)
    assert torch.equal(weights(agent.q_net), weights(other.q_net))


CONFIG = {
    "name": "test_dqn", "env": "CartPole-v1", "agent": "dqn",
    "total_steps": 600, "eval_every": 300, "eval_episodes": 2,
    "final_eval_episodes": 2, "threshold": 475,
    "agent_params": PARAMS | {"learning_starts": 100},
}


def test_same_seed_reproduces_a_dqn_run(tmp_path):
    run(CONFIG, 3, tmp_path / "a")
    run(CONFIG, 3, tmp_path / "b")
    logs = [pd.read_csv(tmp_path / d / "test_dqn" / "seed_3" / "eval.csv")
            .drop(columns="train_seconds") for d in "ab"]
    pd.testing.assert_frame_equal(*logs)


class RecordingAgent:
    """Stands in for an agent and records what the training loop passes it."""

    def __init__(self):
        self.terminated_flags = []

    def act(self, obs, step):
        return step % 2

    def greedy_action(self, obs):
        return 0

    def observe(self, obs, action, reward, next_obs, terminated):
        self.terminated_flags.append(terminated)

    def epsilon(self, step):
        return 0.0

    def diagnostics(self):
        return {}

    def save(self, path):
        pass


def test_time_limit_ends_the_episode_without_marking_it_terminal(tmp_path, monkeypatch):
    agent = RecordingAgent()
    real_make = gym.make
    monkeypatch.setattr(train_module, "make_agent", lambda *args: agent)
    # A 5-step limit is too short to fail, so every episode ends by truncation.
    monkeypatch.setattr(train_module.gym, "make",
                        lambda name: real_make(name, max_episode_steps=5))
    config = CONFIG | {"total_steps": 20, "eval_every": 20, "name": "test_trunc"}
    summary = run(config, 0, tmp_path)
    assert summary["episodes"] == 4
    assert not any(agent.terminated_flags)


def test_recovers_known_q_values_on_a_small_chain():
    """Off-policy data from a 3-state chain must converge to the analytic Q*.

    Action 0 moves right and pays 1 at the last state; action 1 ends the
    episode with reward 0. So Q*(s, 0) = gamma^(2 - s) and Q*(s, 1) = 0.
    """
    gamma = 0.9
    agent = DQNAgent(3, 2, seed=0, gamma=gamma, lr=5e-3, hidden=[32],
                     buffer_size=1000, batch_size=32, learning_starts=32,
                     train_every=1, target_update_every=50, eps_start=1.0,
                     eps_end=1.0, eps_decay_steps=1)
    states = np.eye(3, dtype=np.float32)
    rng = np.random.default_rng(0)
    for _ in range(4000):
        s, a = int(rng.integers(3)), int(rng.integers(2))
        if a == 1:
            agent.observe(states[s], a, 0.0, states[s], True)
        elif s == 2:
            agent.observe(states[s], a, 1.0, states[s], True)
        else:
            agent.observe(states[s], a, 0.0, states[s + 1], False)

    with torch.no_grad():
        q = agent.q_net(torch.from_numpy(states)).numpy()
    expected = np.array([[gamma ** 2, 0.0], [gamma, 0.0], [1.0, 0.0]])
    assert np.allclose(q, expected, atol=0.03)
