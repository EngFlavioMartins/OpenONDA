# VPM post-sync verification results

This record preserves the bounded validation run completed after the VPM
post-sync audit. The measured implementation was at commit
`376534474e100287615ab35127fa180fc2de1769` before this report was added.

## Environment

```text
Platform: Linux x86_64
Python: 3.11.15
NumPy: 2.4.6
Taichi: 1.7.4
Backend: CPU
```

## FMM all-kernel stage matrix

The stage result was compared with an independent direct NumPy evaluation of
the shared `RadialVortexKernel` contract. The case used 16 particles, unequal
core radii, `tolerance=1e-3`, and leaf capacity 1.

| Kernel | Velocity error | Gradient error | Rate error | M2L interactions | Nonzero L2L propagations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian | 1.04% | 0.57% | 0.94% | 94 | 2 |
| High-order Gaussian | 0.98% | 0.27% | 0.65% | 94 | 2 |
| Super-Gaussian | 0.92% | 0.56% | 0.80% | 94 | 2 |
| Winckelmans | 1.03% | 0.28% | 0.69% | 94 | 2 |

Acceptance limits were 3% for velocity and gradient and 5% for the
gradient-derived strength rate. All four kernels passed. Accelerated modes
reported zero direct strength-rate fallbacks.

## Backend smoke benchmarks

These are small CPU smoke runs, not production-scale performance claims.

| Backend | Particles | Steps | Time/step | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Direct | 8 | 1 | 2.71 ms | 460.0 MB |
| Treecode | 8 | 1 | 24.36 ms | 500.7 MB |
| FMM | 16 | 1 | 3826.74 ms | 465.8 MB |

The FMM smoke recorded 2,048 P2P interactions, 128 P2M operations, 64 M2M
operations, 184 L2L operations, 128 L2P evaluations, and zero direct
strength-rate fallbacks. That particular random benchmark cloud produced no
M2L interactions; the all-kernel qualification matrix above uses a separate
deterministic cloud that exercises M2L and nonzero parent-to-child L2L
propagation.

## Automated checks

The following bounded test groups passed:

```text
tests/vpm/test_fmm_hierarchy.py                  7 passed
tests/vpm/test_vortex_kernel_contract.py        24 passed
tests/vpm/test_case_lifecycle.py                16 passed
tests/vpm/test_stage_rhs.py                      6 passed
tests/vpm/test_induction_operator.py             6 passed
tests/vpm/test_core_numerical_qualification.py   2 passed
tests/vpm/test_core_spreading_projection.py      2 passed
tests/test_public_api.py                         1 passed
```

Additional checks passed:

```text
ruff format --check [scoped VPM paths]
ruff check [scoped VPM paths]
python3 -m compileall -q [scoped VPM paths]
python3 scripts/check_api_completeness.py
```

The full repository CI and large-particle performance matrix were not run in
this bounded pass because the host previously exhausted available RAM and the
working tree contains unrelated user changes.
