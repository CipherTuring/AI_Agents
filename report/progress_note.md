# Project 01: progress note

**Team:** Ismael Paredes (individual). **Date:** 5 November 2026.

**Scope.** Tier 1 on CartPole-v1, Tier 2 on LunarLander-v3, plus the analysis-only component. No Tier 3 is planned at this point.

**Done so far.** Tier 1 is complete: tabular Q-learning and DQN, written from scratch, each trained for 200k steps with 5 seeds and the same epsilon schedule. Results below are the mean over seeds of a final 100-episode greedy evaluation.

| Method | Final return | Seeds at or above 475 | Training time |
|---|---|---|---|
| Tabular, 6 bins | 235.4 ± 65.5 | 0 of 5 | 22 s |
| Tabular, 10 bins | 253.3 ± 86.5 | 0 of 5 | 19 s |
| Tabular, 20 bins | 311.0 ± 65.5 | 0 of 5 | 20 s |
| DQN | 498.9 ± 2.4 | 5 of 5 | 136 s |

**Main problem found.** With a Huber loss, DQN's Q-values rose well above the true maximum of 100 and final return stayed near 73 (0 of 5 seeds reached the threshold). Changing only the loss to MSE fixed this. A test on a small MDP with known Q-values ruled out an implementation bug. My working explanation is that Huber caps the gradient of terminal transitions, which are the only ones pulling values down in CartPole. I have not yet tested this directly.

**Next.** A pilot on LunarLander to fix the step budget, then 21 runs for the hyperparameter study (learning rate, epsilon schedule) and the ablation (no target network, no replay buffer). After that, the paper and the defense.

**Question for you.** Does this scope look right for an individual submission?
