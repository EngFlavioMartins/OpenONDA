# FVM qualification evidence map

This checkout does not certify an FVM release level. `source/solvers/fvm/capabilities.json`
therefore uses `evidence-gated` status: a capability is advertised as supported only
when a reproducible command, test path, backend, rank count, mesh/time resolution, and
measured tolerance are recorded here or in a linked machine-readable report.

## Maintained contract gate

Run from the repository root:

```sh
python -m pytest -q tests/fvm tests/coupler tests/test_public_api.py
```

This gate checks configuration admission, public exports, serial lifecycle behavior,
restart staging and history reconciliation, VTK/PVD ownership, coupling smoke behavior,
and the local non-orthogonal-correction contract. It is not evidence of spatial or
temporal order, nonlinear convergence, MPI invariance, or performance qualification.

Focused reproductions are available in:

- `tests/fvm/test_restart_and_diagnostics.py`
- `tests/fvm/test_time_step_control.py`
- `tests/fvm/test_logging.py`
- `tests/fvm/test_nonorthogonal_pressure_correction.py`

## Numerical gates still required

The following gates are specified in `fvm_audit.md` §15 but do not have measured
reports in this checkout: flux identities, manufactured spatial convergence,
controlled temporal order, Taylor–Green decay, steady SIMPLE convergence, nonlinear
outer convergence, restart trajectory equivalence across BDF1/BDF2, and serial versus
partitioned owned-row invariance. Each future report must state the exact source
revision, Python/dependency versions, precision, backend, rank count, mesh family and
resolution, time-step sequence, tolerance, measured error/order, and pass/fail result.

Until those reports exist, broad statements such as “three-level”, “qualified”,
“one/two/four-rank”, or numerical performance limits remain experimental and must not
be promoted by editing the capability label alone.
