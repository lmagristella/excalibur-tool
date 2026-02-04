import numpy as np


def kahan_sum(x: np.ndarray) -> float:
    """Kahan compensated summation for 1D float arrays."""
    s = 0.0
    c = 0.0
    for xi in x:
        y = float(xi) - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def add_small_delta(base: float | np.ndarray, delta: float | np.ndarray) -> float | np.ndarray:
    """Add delta to base with a bit more robustness when |base| >> |delta|.

    This helps recording/integration when coordinates are ~1e24..1e26 and per-step
    increments are << 1e8, which can be swallowed by float64 rounding.

    Note: this doesn't create extra precision out of thin air; it just reduces the
    chances of total cancellation by splitting the operation.
    """
    return (base + delta) - base + base
