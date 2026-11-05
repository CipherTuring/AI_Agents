"""Exploration schedules shared by the tabular and DQN agents."""


def linear_epsilon(step, start, end, decay_steps):
    """Moves epsilon linearly from start to end, then holds it at end."""
    fraction = min(step / decay_steps, 1.0)
    return start + fraction * (end - start)
