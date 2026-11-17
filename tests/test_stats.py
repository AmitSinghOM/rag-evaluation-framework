"""Bootstrap CIs: seeded determinism, coverage of the mean, degenerate inputs."""

from __future__ import annotations

import random

import pytest

from rageval import bootstrap_ci, paired_diff_ci
from rageval.stats import _quantile, mean

VALUES = [0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 0.1, 0.9, 0.3]


def test_same_seed_same_interval_different_seed_different_interval() -> None:
    first = bootstrap_ci(VALUES, resamples=500, seed=7)
    assert bootstrap_ci(VALUES, resamples=500, seed=7) == first
    assert bootstrap_ci(VALUES, resamples=500, seed=8) != first


def test_interval_contains_the_sample_mean_and_is_ordered() -> None:
    lo, hi = bootstrap_ci(VALUES, resamples=1000, seed=0)
    assert lo <= mean(VALUES) <= hi
    assert min(VALUES) <= lo < hi <= max(VALUES)


def test_wider_level_gives_a_wider_interval() -> None:
    lo95, hi95 = bootstrap_ci(VALUES, seed=0, level=0.95)
    lo80, hi80 = bootstrap_ci(VALUES, seed=0, level=0.80)
    assert lo95 <= lo80 and hi80 <= hi95


def test_single_value_gives_a_zero_width_interval() -> None:
    assert bootstrap_ci([0.4], resamples=50, seed=3) == (0.4, 0.4)


def test_constant_values_give_a_zero_width_interval() -> None:
    assert bootstrap_ci([1.0, 1.0, 1.0], resamples=50) == (1.0, 1.0)


def test_uses_random_random_with_the_seed() -> None:
    # The same draws as random.Random(seed).choices, so the interval is reproducible
    # by anyone with the stdlib: recompute one resampled mean by hand.
    rng = random.Random(42)
    draws = [sum(rng.choices(VALUES, k=len(VALUES))) / len(VALUES) for _ in range(20)]
    lo, hi = bootstrap_ci(VALUES, resamples=20, seed=42)
    draws.sort()
    assert lo == pytest.approx(_quantile(draws, 0.025))
    assert hi == pytest.approx(_quantile(draws, 0.975))


def test_quantile_interpolates_linearly_like_numpy_default() -> None:
    assert _quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert _quantile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert _quantile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
    assert _quantile([10.0], 0.3) == 10.0


@pytest.mark.parametrize(
    ("values", "kwargs", "message"),
    [
        ([], {}, "at least one value"),
        ([1.0], {"resamples": 0}, "resamples must be >= 1"),
        ([1.0], {"level": 1.0}, "level must be strictly between"),
        ([1.0], {"level": 0.0}, "level must be strictly between"),
    ],
)
def test_bootstrap_rejects_degenerate_arguments(
    values: list[float], kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        bootstrap_ci(values, **kwargs)  # type: ignore[arg-type]


def test_mean_of_empty_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        mean([])


def test_paired_diff_ci_is_the_bootstrap_of_b_minus_a() -> None:
    a = [0.25, 0.5, 0.75, 1.0]
    b = [0.5, 0.5, 1.0, 0.75]  # differences are exact binary fractions
    assert paired_diff_ci(a, b, resamples=200, seed=1) == bootstrap_ci(
        [0.25, 0.0, 0.25, -0.25], resamples=200, seed=1
    )
    lo, hi = paired_diff_ci(a, b, resamples=200, seed=1)
    assert lo <= mean(b) - mean(a) <= hi


def test_paired_diff_ci_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="differ in length: 2 != 3"):
        paired_diff_ci([0.0, 1.0], [0.0, 1.0, 2.0])
