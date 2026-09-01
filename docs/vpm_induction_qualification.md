# VPM induction-method qualification

The production VPM driver uses one coupled Runge--Kutta state,
`(position, vortex_strength)`, and invokes the configured induction object once
for each RK stage.  `DirectInduction`, `TreecodeInduction`, and `FMMInduction`
all implement the same stage contract.

The permanent qualification suite covers:

- the documented equal- and unequal-core pair operator;
- nonsymmetric gradient orientation and pairwise total-strength cancellation;
- Gaussian, high-order Gaussian, super-Gaussian, and Winckelmans kernel limits;
- deterministic FMM tree construction, P2M/M2M/M2L/L2L/L2P helpers, and exact P2P;
- FMM velocity convergence as the requested tolerance is tightened;
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

The FMM currently uses exact pairwise evaluation for the conservative
vortex-strength rate and the auxiliary gradient.  Its velocity path performs
exact regularized P2P for near interactions and a first-order singular
Biot--Savart multipole translation for well-separated cells.  This preserves
the canonical strength equation while leaving higher-order device translation
optimization as a separate performance task; FMM is therefore not the default
`Numerics` induction choice yet.
