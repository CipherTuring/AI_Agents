# AI-use log

Running record for the paper's AI-use statement and prompt appendix.

## Tools

- Claude Code (Claude Opus 5.5): planning, code, debugging, writing support.

## Tasks and key prompts

| Date | Task | Prompt or decision |
|---|---|---|
| 2026-11-05 | Project planning | Gave Claude the full project brief and asked it to carry out the project individually, asking before each important decision. |
| 2026-11-05 | Scope decisions (made by the student) | CartPole-v1 for Tier 1, LunarLander-v3 for Tier 2, no Tier 3 for now, paper in English with LaTeX on Overleaf. |
| 2026-11-05 | Environment setup | Claude created the virtual environment, pinned dependencies and checked both environments. |

## Verification

Record here how AI-generated claims, hyperparameters and references were checked.

## Stage 1 notes (2026-11-05)

- Claude wrote the discretizer, Q-learning agent, training loop, sweep, figures and tests.
- Velocity ranges come from measured data (`python -m experiments.obs_stats`), not from an AI suggestion.
- The palette validator could not run (Node.js is not installed), so colors follow the documented validated set.

## Stage 2 notes (2026-11-05)

- Claude wrote the DQN agent, its tests and the comparison figures, and drafted the progress note.
- Claude's first explanation for the diverging Q-values (off-policy data from slow epsilon decay) was wrong as the main cause. Controlled runs showed the loss function mattered most: Huber diverged, MSE did not.
- Implementation checked with a test that recovers analytic Q-values on a small MDP.
- Selecting MSE used seeds 0 and 1, which are also part of the reported seeds. The tabular agent got no equivalent tuning beyond the epsilon check. Both points must be disclosed in the paper.

## Stage 3 notes (2026-11-05)

- The first three base runs at 300k steps gave final returns of 123.0, 236.3 and 35.0, and returns were still rising. All Tier 2 runs were restarted at 500k steps so every configuration keeps the same budget.

## Final stage notes (2026-11-05)

- Claude drafted the paper, the slides and the defense guide. References were checked by web search; SAC and TD3 are cited as arXiv preprints because their venue was not confirmed.
- Every number in the paper was taken from the CSV and JSON logs in `results/`.
- The paper was not compiled by Claude, because no LaTeX compiler was available. The student compiles it on Overleaf.
- The slides were checked with approximate previews, not in PowerPoint.
