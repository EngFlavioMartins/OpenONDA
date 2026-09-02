# VPM induction-method qualification

The production VPM driver uses one coupled Runge--Kutta state,
`(position, vortex_strength)`, and invokes the configured induction object once
for each RK stage. `DirectInduction`, `TreecodeInduction`, and `FMMInduction`
all implement the same stage contract while declaring their rate semantics:
direct is exact pairwise-transposed, and the hierarchical paths are explicitly
hierarchical-gradient approximations.

The permanent qualification suite covers:

- the documented equal- and unequal-core pair operator;
- nonsymmetric gradient orientation and pairwise total-strength cancellation;
- Gaussian, high-order Gaussian, super-Gaussian, and Winckelmans kernel limits;
- deterministic FMM tree construction, P2M/M2M/M2L/L2L/L2P helpers, and exact P2P;
- FMM velocity convergence as the requested tolerance is tightened;
- an independent kernel-level FMM stage oracle for velocity and strength rate;
- construction-time backend-by-kernel support/rejection checks;
- explicit backend capabilities, including solver-local builders and the
  treecode f64/device-resident boundary;
- common-stage RK2, SSPRK3, and RK4 state evolution;
- direct Gaussian velocity/gradient and treecode convergence against independent
  numerical references.

Run the focused qualification set with:

```bash
conda run -n OpenONDA python -m pytest -q \
  tests/vpm/test_induction_operator.py \
  tests/vpm/test_vortex_kernel_contract.py \
  tests/vpm/test_fmm_hierarchy.py \
  tests/vpm/test_core_numerical_qualification.py
```

The FMM uses exact regularized P2P for near interactions and second-order
singular Biot--Savart source moments for well-separated cells. Its strength
rate is contracted from the gradient of that same hierarchical far field and is
reported as an approximation to the exact pairwise-transposed rate; the
diagnostic counter for direct strength-rate fallbacks must remain zero. The
independent stage oracle evaluates the documented pair operator directly from
the shared radial-kernel contract rather than calling another induction
backend. The host reference transfers only the active stage prefix through
reusable staging buffers; it remains an explicit opt-in method until a
device-resident FMM backend is separately qualified.

Every configured induction object must provide ``build()``. The solver calls
that builder once during construction, so reusing one immutable ``Numerics``
object creates independent runtime evaluators and hierarchies. Backend
capabilities are checked before Taichi fields are allocated; for example, the
current f32-only LBVH treecode rejects ``precision="f64"`` at case creation.

The reproducible benchmark harness is intentionally bounded by default:

```bash
python scripts/benchmarks/benchmark_vpm_step.py \
  --induction fmm --backend CPU --n 1000 4000 --steps 5 \
  --json /tmp/vpm-fmm-benchmark.json
```

It reports complete-step timings, peak resident memory, and hierarchy
counters. Production-size performance claims require an explicitly recorded
run at the requested particle count and hardware; the harness does not invent
those measurements.
