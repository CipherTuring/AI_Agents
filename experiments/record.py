"""Records a GIF of a trained agent playing greedily.

Usage:
    python -m experiments.record --config experiments/configs/tier1_dqn.yaml --seed 0
"""
import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio

from experiments.train import RESULTS_DIR, load_config, make_agent

VIDEOS_DIR = RESULTS_DIR / "videos"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=400)
    args = parser.parse_args()

    config = load_config(args.config)
    env = gym.make(config["env"], render_mode="rgb_array")
    agent = make_agent(config, env, args.seed)
    agent.load(RESULTS_DIR / config["name"] / f"seed_{args.seed}")

    obs, _ = env.reset(seed=args.episode_seed)
    frames, total, done = [], 0.0, False
    while not done:
        # Frames are half-size and capped to keep the file small; the return
        # still counts the whole episode.
        if len(frames) < args.max_frames:
            frames.append(env.render()[::2, ::2])
        obs, reward, terminated, truncated, _ = env.step(agent.greedy_action(obs))
        total += reward
        done = terminated or truncated
    env.close()

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    path = VIDEOS_DIR / f"{config['name']}_seed{args.seed}.gif"
    imageio.mimsave(path, frames, duration=33, loop=0)
    print(f"{path}: return {total:.1f}, {len(frames)} frames")


if __name__ == "__main__":
    main()
