# Numerical-equivalence baseline

These `.npy` snapshots are the pre-refactor numerical baseline for the
nomenclature refactor, captured from commit `349d4b8` (parent of the first
rename PR) with `scripts/nomenclature/numerical_baseline.py`.

Each case directory holds the *actual solver state* (field arrays, particle
state) as raw float64/int64 arrays; `manifest.json` records a SHA-256 of every
array.  No log text is stored.

To regenerate and compare after a refactor step:

    python scripts/nomenclature/numerical_baseline.py capture --outdir /tmp/nb
    python scripts/nomenclature/numerical_baseline.py compare \
        --old samples/numerical_baseline --new /tmp/nb

The comparison is exact (bit-wise, `max_diff == 0`) for the CPU-deterministic
cases listed in the script docstring.