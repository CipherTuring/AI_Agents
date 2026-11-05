# Reinforcement Learning and DQN on Gymnasium

Tabular Q-learning and DQN written from scratch in Python and PyTorch, compared on CartPole-v1 (Tier 1). DQN is then studied on LunarLander-v3 (Tier 2) with a hyperparameter study and an ablation. The paper source is in [report/](report/).

## Setup

Python 3.12. From the repository root:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest
```

The commands below assume the environment is active and run from the repository root.

On Windows, keep the repository in a short path such as `C:\rl`. PyTorch installs a very long file path, and pip fails with `OSError` if the total passes 260 characters and long paths are not enabled.

## Layout

| Path | Content |
|---|---|
| `agents/` | `tabular.py` (Q-learning), `dqn.py` (network, replay buffer, agent), `schedules.py` (epsilon) |
| `envs/` | `discretizer.py` (uniform bins) |
| `experiments/` | `train.py` (one run), `sweep.py` (parallel runs), `evaluate.py`, `make_figures.py`, `record.py`, `obs_stats.py` |
| `experiments/configs/` | One YAML file per experiment, including seeds and all hyperparameters |
| `results/<experiment>/seed_<n>/` | `train.csv`, `eval.csv`, `summary.json` and trained weights |
| `results/figures/`, `results/tables/`, `results/videos/` | Report figures, summary tables and GIFs |
| `tests/` | Unit tests for the discretizer, both agents and the training loop |

## Reproducing the results

`sweep.py` skips runs that already have a `summary.json`. To retrain, delete that experiment's folder in `results/` first. Times are for 4 parallel runs on an Intel i5-8300H.

| Report item | Train (time) | Build figure or table |
|---|---|---|
| Observation ranges behind the bin limits | none | `python -m experiments.obs_stats` |
| Tabular bins and ranges | `python -m experiments.sweep "experiments/configs/tier1_tabular_*.yaml"` (about 4 min) | `python -m experiments.make_figures tier1_tabular` |
| Tabular vs DQN, MSE vs Huber | `python -m experiments.sweep "experiments/configs/tier1_dqn*.yaml"` (about 7 min) | `python -m experiments.make_figures tier1_compare` |
| LunarLander study and ablation | `python -m experiments.sweep "experiments/configs/tier2_*.yaml"` (about 40 min) | `python -m experiments.make_figures tier2` |

The Tier 1 comparison also needs the tabular runs from the row above it.

To train a single run, or to check a config quickly:

```
python -m experiments.train --config experiments/configs/tier1_dqn.yaml --seed 0
python -m experiments.train --config experiments/configs/tier1_dqn.yaml --seed 0 --smoke
```

To record a GIF of a trained agent:

```
python -m experiments.record --config experiments/configs/tier1_dqn.yaml --seed 0
python -m experiments.record --config experiments/configs/tier1_tabular_bins20.yaml --seed 1
python -m experiments.record --config experiments/configs/tier2_base.yaml --seed 0
```

## Protocol

Every 5k steps (CartPole) or 10k steps (LunarLander), 10 greedy episodes are run in a separate environment. After training, 100 greedy episodes on the final weights decide whether the threshold is met. Seeds are fixed for the environment, NumPy and PyTorch, and a run repeated with the same seed gives the same logs. `train_seconds` excludes evaluation time.
