import numpy as np
import pytest

from agents.schedules import linear_epsilon
from agents.tabular import QLearningAgent
from envs.discretizer import UniformDiscretizer


def make_agent(eps=0.0):
    d = UniformDiscretizer([0.0], [2.0], 2)  # obs 0.5 -> state 0, 1.5 -> state 1
    return QLearningAgent(2, d, seed=0, alpha=0.5, gamma=0.9, eps_start=eps,
                          eps_end=eps, eps_decay_steps=1)


def test_update_bootstraps_from_best_next_action():
    agent = make_agent()
    agent.q[1] = [2.0, 4.0]
    agent.observe([0.5], 0, 1.0, [1.5], terminated=False)
    # target = 1 + 0.9 * 4 = 4.6, and Q moves halfway from 0.
    assert agent.q[0, 0] == pytest.approx(2.3)


def test_terminal_transition_has_no_bootstrap():
    agent = make_agent()
    agent.q[1] = [2.0, 4.0]
    agent.observe([0.5], 0, 1.0, [1.5], terminated=True)
    assert agent.q[0, 0] == pytest.approx(0.5)


def test_greedy_action_picks_highest_value():
    agent = make_agent()
    agent.q[0] = [1.0, 3.0]
    assert agent.greedy_action([0.5]) == 1


def test_ties_are_broken_randomly_during_training():
    agent = make_agent()
    assert {agent.act([0.5], step=1) for _ in range(50)} == {0, 1}


def test_save_and_load_round_trip(tmp_path):
    agent = make_agent()
    agent.q[0] = [1.5, -2.5]
    agent.save(tmp_path)
    other = make_agent()
    other.load(tmp_path)
    assert np.allclose(other.q, agent.q)


def test_linear_epsilon_endpoints_and_midpoint():
    assert linear_epsilon(0, 1.0, 0.1, 100) == pytest.approx(1.0)
    assert linear_epsilon(50, 1.0, 0.1, 100) == pytest.approx(0.55)
    assert linear_epsilon(100, 1.0, 0.1, 100) == pytest.approx(0.1)
    assert linear_epsilon(500, 1.0, 0.1, 100) == pytest.approx(0.1)
