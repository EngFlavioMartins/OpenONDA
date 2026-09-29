# Exact SciPy direct-factor reuse in the FVM predictor

The serial immersed-boundary step has two momentum predictors and two pressure
correctors. Its standard three Cartesian momentum components already share one
sparse matrix and one three-column direct solve. Thus the six component
telemetry records correspond to two momentum factorizations, not six.

The new solver-owned matrix workspace retains **one** sparse LU. Before every
reuse it compares the canonical CSR shape, coefficient dtype, row pointers,
column indices, and a private copy of every coefficient. The comparison sees
the final matrix passed to SciPy, including pressure anchoring and mixed
boundary contributions. Mutable assembly storage cannot make a stale LU look
current. A matrix change or failed factorization clears the stored factor. The old LU is
released **before** constructing its replacement, so momentum and pressure do
not retain two LU objects during a switch; a weak-reference test checks this
at the factorization call. The LU is also released on solver close. Momentum forcing and pressure right-hand
sides still change and each new solution passes the existing algebraic
residual check. PETSc and iterative solvers are unchanged. This adds no setup
parameter and does not change corrector counts or numerical tolerances.

A real one-step CPU IBM case with 1,536 cells produced four sparse systems.
There were two LU factorizations with the workspace (one momentum, one
pressure); the unmodified `spsolve` path factorizes on each call. Replaying
those exact matrices and right-hand sides for six alternating repetitions,
including coefficient comparisons and residual verification, gave **85.57 ms**
uncached versus **67.09 ms** cached (1.28× solve-stage speedup) after the
memory-lifetime fix. An earlier replay gave 47.95 ms versus 40.32 ms (1.19×);
background host load changed absolute times, but both comparisons favored
exact reuse.
This is a bounded replay measurement on the development host, not a whole-case
speedup or a forecast for a particular MPI/GPU run. The reproducible script reports its own measurements; rerun with:

```bash
python -m studies.benchmark_fvm_direct_factorization
```

Verification compares the native one-step velocity, pressure and face flux
against fresh `spsolve` results, then runs the existing 40-step IBM/freestream
energy, no-slip, boundary-branch and rollback checks. Unit tests cover a
three-column right-hand side, coefficient/structure/dtype changes, pressure
anchor changes, singular failure and recovery. On this case one-step IBM slip
is transient; the existing 40-step check is the physical no-slip regression.
