"""Benchmark: separable kernels (new) vs loop-based kernels (old .bak3)."""
import numpy as np
import time
import importlib
import sys, os

# --- helpers ---
class MockGrid:
    def __init__(self, ndim):
        if ndim == 3:
            self.shape = (32, 32, 32)
            self.origin = (0.0, 0.0, 0.0)
            self.spacing = (1.0, 1.0, 1.0)
        else:
            self.shape = (32, 32, 32, 16)
            self.origin = (0.0, 0.0, 0.0, 0.0)
            self.spacing = (1.0, 1.0, 1.0, 1.0)
        data = np.random.RandomState(42).randn(*self.shape)
        self.fields = {"f": data}

def bench_fn(fn, n=5000, **kwargs):
    # warmup
    for _ in range(200):
        fn(**kwargs)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(**kwargs)
    return (time.perf_counter() - t0) / n * 1e6  # µs

# ---- NEW version (current) ----
from excalibur.grid.interpolator_4d_fast import InterpolatorFast

print("=" * 60)
print("NEW (separable) version")
print("=" * 60)

g3 = MockGrid(3)
g4 = MockGrid(4)
pos = np.array([15.3, 12.7, 10.4])

for scheme in ("trilinear", "tricubic"):
    interp3 = InterpolatorFast(g3, boundary="clamp", scheme=scheme)
    us = bench_fn(interp3.full_4d, x=pos, field="f")
    print(f"  3D {scheme:10s}  full_4d:  {us:7.2f} µs")

    interp4 = InterpolatorFast(g4, boundary="clamp", scheme=scheme)
    us = bench_fn(interp4.full_4d, x=pos, field="f", t=7.5)
    print(f"  4D {scheme:10s}  full_4d:  {us:7.2f} µs")

# ---- OLD version (bak3) ----
# Load old module under a different name
spec_old = importlib.util.spec_from_file_location(
    "interpolator_old",
    os.path.join(os.path.dirname(__file__), "..", "excalibur", "grid", "interpolator_4d_fast.py.bak3"))
old_mod = importlib.util.module_from_spec(spec_old)
spec_old.loader.exec_module(old_mod)
InterpolatorFastOld = old_mod.InterpolatorFast

print()
print("=" * 60)
print("OLD (loop-based) version  (.bak3)")
print("=" * 60)

for scheme in ("trilinear", "tricubic"):
    interp3o = InterpolatorFastOld(g3, boundary="clamp", scheme=scheme)
    us = bench_fn(interp3o.full_4d, x=pos, field="f")
    print(f"  3D {scheme:10s}  full_4d:  {us:7.2f} µs")

    interp4o = InterpolatorFastOld(g4, boundary="clamp", scheme=scheme)
    us = bench_fn(interp4o.full_4d, x=pos, field="f", t=7.5)
    print(f"  4D {scheme:10s}  full_4d:  {us:7.2f} µs")
