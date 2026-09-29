# VPM capacity and restart qualification

The target-query batch size is an internal workspace detail. It is no longer a
`Numerics` input or part of the current restart identity. Old backups remain
subject to their saved SHA-256 check before their historical
`max_evaluation_points` entry is discarded for configuration comparison.

Filament refinement now either splits every eligible parent or raises before
building replacement particle arrays when the hard particle capacity is too
small. This makes a larger capacity compatible with an authenticated last
accepted backup when filament refinement is active. Regularization still
requires an exact capacity because its accepted output can depend on it.
The exception does not roll back other work already performed by a failed
time step; continuation remains from the last accepted backup.

Native verification on Python 3.11.15 with Taichi 1.7.4:

- `python -m pytest -q tests/vpm/test_case_lifecycle.py tests/vpm/test_stabilization_schedules.py`: 54 passed.
- `python -m pytest -q tests/vpm/test_backup_storage.py -k 'refinement_capacity_can_increase_after_last_accepted_backup or restart_keeps_capacity_checks or restart_ignores_authenticated_legacy_target_batch_capacity'`: 4 passed.
- Ruff check of changed source and test files: passed.

The backup test verifies rejected tampering, then loads a correctly
authenticated legacy batch-size key. Another writes a real last accepted
backup at capacity 3, proves the required split fails without changing
particle/lineage state, loads that backup at capacity 4, and completes the
conservative split.
