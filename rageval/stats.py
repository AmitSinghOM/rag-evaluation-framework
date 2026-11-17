"""Bootstrap confidence intervals, pure stdlib.

A percentile bootstrap: draw ``resamples`` samples of size ``n`` with
replacement from the per-query values (``random.Random(seed)``), take the
mean of each, and report the ``(1 - level) / 2`` and ``(1 + level) / 2``
quantiles of those means. Quantiles interpolate linearly between order
statistics, the same rule as ``numpy.percentile``'s default. The same
``seed`` always gives the same interval.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from math import floor, fsum


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean using an exact-sum accumulator; ``ValueError`` when empty."""
    if not values:
        raise ValueError("mean of an empty sequence")
    return fsum(values) / len(values)


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of an already sorted sequence."""
    position = q * (len(sorted_values) - 1)
    lower = floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_ci(
    values: Sequence[float], resamples: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of ``values``.

    ``n = 1`` gives a zero-width interval at that value. Raises
    ``ValueError`` for an empty sequence, ``resamples < 1`` or a ``level``
    outside ``(0, 1)``.
    """
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    if resamples < 1:
        raise ValueError("resamples must be >= 1")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between 0 and 1")
    rng = random.Random(seed)  # noqa: S311 - statistics, not security
    n = len(values)
    means = sorted(fsum(rng.choices(values, k=n)) / n for _ in range(resamples))
    alpha = (1.0 - level) / 2.0
    return _quantile(means, alpha), _quantile(means, 1.0 - alpha)


def paired_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap CI of the mean paired difference ``b - a`` over aligned values."""
    if len(a) != len(b):
        raise ValueError(f"paired sequences differ in length: {len(a)} != {len(b)}")
    diffs = [y - x for x, y in zip(a, b, strict=True)]
    return bootstrap_ci(diffs, resamples=resamples, seed=seed, level=level)
