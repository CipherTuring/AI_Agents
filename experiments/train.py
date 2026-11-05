"""Trains one agent with one seed and writes its logs and weights.

Usage:
    python -m experiments.train --config experiments/configs/X.yaml --seed 0
"""
import argparse
import json
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import yaml

from agents.dqn import DQNAgent
from agents.tabular import QLearningAgent
from envs.discretizer import UniformDiscretizer
from experiments.evaluate import evaluate

RESULTS_DIR = Path("results")
EVAL_SEED_OFFSET = 1_000_000


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_agent(config, env, seed):
    params = dict(config["agent_params"])
    if config["agent"] == "tabular":
        discretizer = UniformDiscretizer(
            params.pop("low"), params.pop("high"), params.pop("bins")
        )
        return QLearningAgent(env.action_space.n, discretizer, seed, **params)
    if config["agent"] == "dqn":
        obs_dim = env.observation_space.shape[0]
        return DQNAgent(obs_dim, env.action_space.n, seed, **params)
    raise ValueError(f"Unknown agent: {config['agent']}")


def run(config, seed, results_dir=RESULTS_DIR):
    """Runs the training loop shared by every agent and returns its summary."""
    seed_everything(seed)
    # One thread per run: the sweep runs several at once, and tiny networks
    # gain nothing from intra-op parallelism.
    torch.set_num_threads(1)

    env = gym.make(config["env"])
    eval_env = gym.make(config["env"])
    env.action_space.seed(seed)
    agent = make_agent(config, env, seed)

    total_steps = config["total_steps"]
    eval_every = config["eval_every"]
    eval_seed = EVAL_SEED_OFFSET + seed * 1000
    threshold = config["threshold"]

    train_log, eval_log = [], []
    train_seconds = 0.0
    episode, episode_return, episode_length = 0, 0.0, 0
    obs, _ = env.reset(seed=seed)
    start = time.perf_counter()

    for step in range(1, total_steps + 1):
        action = agent.act(obs, step)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.observe(obs, action, reward, next_obs, terminated)
        obs = next_obs
        episode_return += reward
        episode_length += 1

        if terminated or truncated:
            train_log.append({
                "step": step, "episode": episode, "return": episode_return,
                "length": episode_length, "epsilon": agent.epsilon(step),
            })
            episode += 1
            episode_return, episode_length = 0.0, 0
            obs, _ = env.reset()

        if step % eval_every == 0:
            # The clock stops during evaluation, so train_seconds measures
            # the cost of learning only.
            train_seconds += time.perf_counter() - start
            returns = evaluate(
                eval_env, agent.greedy_action, config["eval_episodes"], eval_seed
            )
            eval_log.append({
                "step": step, "mean_return": returns.mean(),
                "std_return": returns.std(), "train_seconds": train_seconds,
                **agent.diagnostics(),
            })
            start = time.perf_counter()

    final_returns = evaluate(
        eval_env, agent.greedy_action, config["final_eval_episodes"],
        eval_seed + 500,
    )
    env.close()
    eval_env.close()

    eval_df = pd.DataFrame(eval_log)
    reached = eval_df[eval_df["mean_return"] >= threshold]
    summary = {
        "name": config["name"],
        "seed": seed,
        "total_steps": total_steps,
        "episodes": episode,
        "train_seconds": train_seconds,
        "final_mean_return": float(final_returns.mean()),
        "final_std_return": float(final_returns.std()),
        "threshold": threshold,
        "threshold_reached_final": bool(final_returns.mean() >= threshold),
        "steps_to_threshold": int(reached["step"].iloc[0]) if len(reached) else None,
        "best_eval_mean": float(eval_df["mean_return"].max()),
    }

    out = results_dir / config["name"] / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_log).to_csv(out / "train.csv", index=False)
    eval_df.to_csv(out / "eval.csv", index=False)
    agent.save(out)
    # Written last: its presence is what marks a run as complete.
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def apply_smoke(config):
    """Shrinks a config to a quick end-to-end check."""
    config = dict(config)
    config.update(total_steps=2000, eval_every=1000, eval_episodes=2,
                  final_eval_episodes=2)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="short run written to results/smoke")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = RESULTS_DIR
    if args.smoke:
        config = apply_smoke(config)
        results_dir = RESULTS_DIR / "smoke"
    print(json.dumps(run(config, args.seed, results_dir), indent=2))


if __name__ == "__main__":
    main()
