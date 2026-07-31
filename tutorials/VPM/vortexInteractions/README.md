# Vortex interactions

This tutorial is the regression problem for three-dimensional VPM stability.
It provides two untreated controls and one physics-gated candidate for both
leapfrogging and colliding vortex rings:

- `baseline`: molecular-viscosity DNS;
- `les`: the same numerical method with Smagorinsky LES.
- `les_stabilized`: LES plus conservative material-line refinement and
  reference-restoring constrained Winckelmans divergence relaxation.

Both controls intentionally remain unmodified so the selected stabilization
method is compared against the same physical trajectory. A normal run executes
all six family/method combinations:

```bash
./allrun.sh
```

The equations, invariant proof, hard gates, and outstanding validation items
are recorded in [`STABILIZATION_METHOD.md`](STABILIZATION_METHOD.md).

Rerun only the two stabilized cases with:

```bash
METHODS=les_stabilized RUN_PLOTS=0 ./allrun.sh
```

The reference study requires `h/a0 <= 0.2`. The coarser `h=0.03` setup is only
an explicitly enabled stress test and is not convergence evidence.
