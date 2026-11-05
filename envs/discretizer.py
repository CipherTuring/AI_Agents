"""Maps continuous observations to integer bin indices for tabular RL."""
import numpy as np


class UniformDiscretizer:
    """Splits each observation dimension into equal-width bins.

    Values outside [low, high] fall into the first or last bin, so a range
    only has to cover the values that matter, not every possible value.
    That is what makes CartPole's unbounded velocities usable here.
    """

    def __init__(self, low, high, bins):
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        if low.shape != high.shape or np.any(low >= high):
            raise ValueError("low and high must match in shape, with low < high")

        # A single integer means "this many bins in every dimension".
        if np.isscalar(bins):
            bins = [bins] * len(low)
        if len(bins) != len(low) or min(bins) < 1:
            raise ValueError("bins needs one positive integer per dimension")

        self.bins = tuple(int(b) for b in bins)
        # n bins are separated by n - 1 interior edges.
        self.edges = [
            np.linspace(lo, hi, b + 1)[1:-1] for lo, hi, b in zip(low, high, self.bins)
        ]

    def __call__(self, obs):
        """Returns one bin index per dimension, as a tuple usable as a table key."""
        return tuple(
            int(np.searchsorted(edges, value, side="right"))
            for edges, value in zip(self.edges, obs)
        )

    @property
    def n_states(self):
        return int(np.prod(self.bins))
