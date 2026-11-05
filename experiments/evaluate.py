"""Greedy policy evaluation, kept apart from training."""
import numpy as np


def evaluate(env, policy, episodes, seed):
    """Returns the undiscounted return of each greedy episode.

    Episode i always starts from reset(seed=seed + i), so every evaluation
    point sees the same start states and never touches the training RNGs.
    """
    returns = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            total += reward
            done = terminated or truncated
        returns.append(total)
    return np.array(returns)
