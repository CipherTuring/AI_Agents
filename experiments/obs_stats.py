"""Measures CartPole observation ranges to justify the discretization bounds.

Usage:
    python -m experiments.obs_stats
"""
import gymnasium as gym
import numpy as np

NAMES = ["cart position", "cart velocity", "pole angle", "pole angular velocity"]


def random_policy(env, obs):
    return env.action_space.sample()


def balancing_policy(env, obs):
    """Simple hand-written controller: push towards the side the pole falls."""
    return int(obs[2] + obs[3] > 0)


def collect(policy, steps=100_000, seed=0):
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    observations = []
    for _ in range(steps):
        observations.append(obs)
        obs, _, terminated, truncated, _ = env.step(policy(env, obs))
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return np.array(observations)


def main():
    for label, policy in [("random", random_policy), ("balancing", balancing_policy)]:
        data = collect(policy)
        print(f"\n{label} policy, {len(data)} observations")
        print(f"{'dimension':<24}{'p0.5':>9}{'p99.5':>9}{'min':>9}{'max':>9}")
        for i, name in enumerate(NAMES):
            column = data[:, i]
            low, high = np.percentile(column, [0.5, 99.5])
            print(f"{name:<24}{low:>9.3f}{high:>9.3f}"
                  f"{column.min():>9.3f}{column.max():>9.3f}")


if __name__ == "__main__":
    main()
