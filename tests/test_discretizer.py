import numpy as np
import pytest

from envs.discretizer import UniformDiscretizer


def test_values_land_in_expected_bins():
    d = UniformDiscretizer([0.0], [10.0], 5)
    assert [d([x])[0] for x in (0.5, 2.5, 4.5, 6.5, 8.5)] == [0, 1, 2, 3, 4]


def test_edge_belongs_to_the_upper_bin():
    d = UniformDiscretizer([0.0], [10.0], 5)
    assert d([2.0]) == (1,)


def test_out_of_range_values_clip_to_outer_bins():
    d = UniformDiscretizer([-1.0, -1.0], [1.0, 1.0], 4)
    assert d([-100.0, 100.0]) == (0, 3)
    assert d([np.inf, -np.inf]) == (3, 0)


def test_bins_per_dimension_and_state_count():
    d = UniformDiscretizer([0, 0, 0], [1, 1, 1], [2, 3, 4])
    assert d.bins == (2, 3, 4)
    assert d.n_states == 24


def test_invalid_ranges_are_rejected():
    with pytest.raises(ValueError):
        UniformDiscretizer([1.0], [0.0], 3)
    with pytest.raises(ValueError):
        UniformDiscretizer([0.0, 0.0], [1.0, 1.0], [3])
