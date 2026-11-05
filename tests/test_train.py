import pandas as pd

from experiments.train import run

CONFIG = {
    "name": "test_tabular",
    "env": "CartPole-v1",
    "agent": "tabular",
    "total_steps": 3000,
    "eval_every": 1000,
    "eval_episodes": 3,
    "final_eval_episodes": 3,
    "threshold": 475,
    "agent_params": {
        "low": [-2.4, -3.0, -0.21, -3.0], "high": [2.4, 3.0, 0.21, 3.0],
        "bins": 6, "alpha": 0.1, "gamma": 0.99,
        "eps_start": 1.0, "eps_end": 0.05, "eps_decay_steps": 1500,
    },
}


def logs(results_dir, seed):
    run_dir = results_dir / "test_tabular" / f"seed_{seed}"
    train = pd.read_csv(run_dir / "train.csv")
    evals = pd.read_csv(run_dir / "eval.csv").drop(columns="train_seconds")
    return train, evals


def test_same_seed_reproduces_the_run(tmp_path):
    run(CONFIG, 7, tmp_path / "a")
    run(CONFIG, 7, tmp_path / "b")
    for left, right in zip(logs(tmp_path / "a", 7), logs(tmp_path / "b", 7)):
        pd.testing.assert_frame_equal(left, right)


def test_different_seeds_differ(tmp_path):
    run(CONFIG, 1, tmp_path)
    run(CONFIG, 2, tmp_path)
    assert not logs(tmp_path, 1)[0].equals(logs(tmp_path, 2)[0])


def test_summary_and_eval_schedule(tmp_path):
    summary = run(CONFIG, 0, tmp_path)
    assert list(logs(tmp_path, 0)[1]["step"]) == [1000, 2000, 3000]
    assert summary["total_steps"] == 3000
    assert summary["steps_to_threshold"] is None
