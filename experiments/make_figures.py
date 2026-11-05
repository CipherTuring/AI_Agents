"""Builds the report figures and summary tables from the logs in results/.

Usage:
    python -m experiments.make_figures tier1_tabular
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.train import RESULTS_DIR

FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Categorical colors in a fixed order, so a series keeps its color everywhere.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "text.color": TEXT,
    "savefig.dpi": 300,
})


def seed_dirs(name):
    return sorted((RESULTS_DIR / name).glob("seed_*"))


def eval_curves(name, column="mean_return"):
    """Returns one logged column as a table: a row per step, a column per seed."""
    runs = {d.name: pd.read_csv(d / "eval.csv").set_index("step")[column]
            for d in seed_dirs(name)}
    return pd.DataFrame(runs)


def summaries(name):
    return pd.DataFrame(json.loads((d / "summary.json").read_text())
                        for d in seed_dirs(name))


def visited_states(name):
    """Mean share of table states that received at least one update."""
    shares = []
    for d in seed_dirs(name):
        q = np.load(d / "q_table.npz")["q"]
        shares.append(np.any(q != 0, axis=-1).mean())
    return float(np.mean(shares))


def plot_curves(series, ylabel, filename, column="mean_return", reference=None,
                bounds=None, log_y=False, time_axis=False):
    """Plots the mean across seeds with a band of one standard deviation.

    reference is an optional (value, label) pair drawn as a horizontal line.
    bounds is the (low, high) range the metric can really take; the band is
    clipped to it, because mean +/- std can leave that range.
    With log_y the band is the min-max across seeds, since mean - std can be
    negative. With time_axis, x is the mean training time instead of steps.
    """
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for name, label, color in series:
        curves = eval_curves(name, column)
        mean, std = curves.mean(axis=1), curves.std(axis=1)
        lower, upper = mean - std, mean + std
        if log_y:
            lower, upper = curves.min(axis=1), curves.max(axis=1)
        if bounds:
            lower, upper = lower.clip(*bounds), upper.clip(*bounds)
        x = eval_curves(name, "train_seconds").mean(axis=1) if time_axis else curves.index
        ax.fill_between(x, lower, upper, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=1.5, label=label)

    if reference:
        value, text = reference
        ax.axhline(value, color=MUTED, linewidth=0.8)
        ax.annotate(text, (0, value), xytext=(2, 2), textcoords="offset points",
                    color=MUTED, fontsize=7)
    if log_y:
        ax.set_yscale("log")
    if bounds:
        ax.set_ylim(bounds[0], bounds[1] * 1.04)
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    if time_axis:
        ax.set_xlabel("Training time (s)")
    else:
        ax.set_xlabel("Environment steps")
        ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}k")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(series),
              handlelength=1.2, columnspacing=1.2)
    fig.tight_layout(pad=0.4)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{filename}.pdf")
    fig.savefig(FIGURES_DIR / f"{filename}.png")
    plt.close(fig)


def summary_row(name, label):
    s = summaries(name)
    reached = s["steps_to_threshold"].dropna()
    return {
        "config": label,
        "seeds": len(s),
        "final_mean": round(s["final_mean_return"].mean(), 1),
        "final_std_across_seeds": round(s["final_mean_return"].std(), 1),
        "seeds_final_above_threshold": int(s["threshold_reached_final"].sum()),
        "seeds_eval_hit_threshold": len(reached),
        "median_steps_to_threshold": reached.median() if len(reached) else "not reached",
        "best_eval_mean": round(s["best_eval_mean"].mean(), 1),
        "train_seconds": round(s["train_seconds"].mean(), 1),
    }


CARTPOLE_THRESHOLD = (475, "threshold 475")
CARTPOLE_BOUNDS = (0, 500)
RETURN_LABEL = "Greedy evaluation return"


def save_table(rows, filename):
    table = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / f"{filename}.csv", index=False)
    print(table.to_string(index=False))


def tier1_tabular():
    bins = [
        ("tier1_tabular_bins6", "6 bins", ORANGE),
        ("tier1_tabular_bins10", "10 bins", BLUE),
        ("tier1_tabular_bins20", "20 bins", AQUA),
    ]
    ranges = [
        ("tier1_tabular_range_narrow", "Narrow (x0.5)", ORANGE),
        ("tier1_tabular_bins10", "Base", BLUE),
        ("tier1_tabular_range_wide", "Wide (x2)", AQUA),
    ]
    plot_curves(bins, RETURN_LABEL, "tier1_tabular_bins",
                reference=CARTPOLE_THRESHOLD, bounds=CARTPOLE_BOUNDS)
    plot_curves(ranges, RETURN_LABEL, "tier1_tabular_ranges",
                reference=CARTPOLE_THRESHOLD, bounds=CARTPOLE_BOUNDS)

    rows = []
    for name, label in [
        ("tier1_tabular_bins6", "6 bins"),
        ("tier1_tabular_bins10", "10 bins (base)"),
        ("tier1_tabular_bins20", "20 bins"),
        ("tier1_tabular_range_narrow", "10 bins, narrow velocities"),
        ("tier1_tabular_range_wide", "10 bins, wide velocities"),
        ("tier1_tabular_bins20_eps20k", "20 bins, epsilon decay 20k"),
    ]:
        rows.append(summary_row(name, label) | {"visited_states": visited_states(name)})

    save_table(rows, "tier1_tabular")


def tier1_compare():
    methods = [
        ("tier1_tabular_bins20", "Tabular (20 bins)", AQUA),
        ("tier1_dqn", "DQN", BLUE),
    ]
    losses = [
        ("tier1_dqn", "MSE loss", BLUE),
        ("tier1_dqn_huber", "Huber loss", ORANGE),
    ]
    plot_curves(methods, RETURN_LABEL, "tier1_compare_steps",
                reference=CARTPOLE_THRESHOLD, bounds=CARTPOLE_BOUNDS)
    plot_curves(methods, RETURN_LABEL, "tier1_compare_time",
                reference=CARTPOLE_THRESHOLD, bounds=CARTPOLE_BOUNDS, time_axis=True)
    plot_curves(losses, RETURN_LABEL, "tier1_loss_return",
                reference=CARTPOLE_THRESHOLD, bounds=CARTPOLE_BOUNDS)
    # With reward 1 per step, no true value can exceed 1 / (1 - gamma) = 100.
    plot_curves(losses, "Mean Q-value (log scale)", "tier1_loss_q", column="mean_q",
                reference=(100, "true maximum 100"), log_y=True)

    save_table([
        summary_row("tier1_tabular_bins20", "Tabular, 20 bins"),
        summary_row("tier1_tabular_range_narrow", "Tabular, 10 bins narrow"),
        summary_row("tier1_dqn", "DQN, MSE"),
        summary_row("tier1_dqn_huber", "DQN, Huber"),
    ], "tier1_compare")


def tier2():
    threshold = (200, "threshold 200")
    label = "Greedy evaluation return"
    plot_curves([
        ("tier2_lr_low", "lr 2.5e-4", ORANGE),
        ("tier2_base", "lr 5e-4 (base)", BLUE),
        ("tier2_lr_high", "lr 1e-3", AQUA),
    ], label, "tier2_lr", reference=threshold)
    plot_curves([
        ("tier2_eps_fast", "25k steps", ORANGE),
        ("tier2_base", "50k (base)", BLUE),
        ("tier2_eps_slow", "100k steps", AQUA),
    ], label, "tier2_eps", reference=threshold)
    plot_curves([
        ("tier2_no_target", "No target net", ORANGE),
        ("tier2_base", "Full DQN", BLUE),
        ("tier2_no_replay", "No replay", AQUA),
    ], label, "tier2_ablation", reference=threshold)

    save_table([summary_row(name, text) for name, text in [
        ("tier2_base", "Base (lr 5e-4, decay 50k)"),
        ("tier2_lr_low", "lr 2.5e-4"),
        ("tier2_lr_high", "lr 1e-3"),
        ("tier2_eps_fast", "epsilon decay 25k"),
        ("tier2_eps_slow", "epsilon decay 100k"),
        ("tier2_no_target", "No target network"),
        ("tier2_no_replay", "No replay buffer"),
    ]], "tier2")


FIGURES = {"tier1_tabular": tier1_tabular, "tier1_compare": tier1_compare,
           "tier2": tier2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=sorted(FIGURES))
    FIGURES[parser.parse_args().figure]()


if __name__ == "__main__":
    main()
