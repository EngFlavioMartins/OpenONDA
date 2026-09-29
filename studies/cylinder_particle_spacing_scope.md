# Independent particle-spacing sensitivity

The cylinder sensitivity study now changes particle spacing while keeping the
physical injection core radius, FVM/VPM blend width, and VPM release width fixed.
At the study baseline (`hxy = 0.08 D`, span `0.96 D`), these are respectively
`0.08 D`, `0.48 D`, and `0.16 D`. The requested particle spacing ratios 1.25
and 1.5 produce spacings `0.096 D` and `0.12 D`, with 10 and 8 uniform axial
intervals. The baseline has 12 intervals at `0.08 D`.

These changes intentionally vary `sigma/hp`: 1, 5/6, and 2/3. This overlap
change is recorded and is part of the particle-resolution experiment; it is
not a separate change in the physical core radius. The FVM mesh, exchange
clock, and hard particle capacity remain unchanged in this comparison.
Independent core/blend/release factors instead change their physical lengths
relative to the baseline spacing. Interactions combine the requested factors
before converting lengths to the existing solver ratios.

Span quantization uses the same helper as the native case builder and retains
the six-interval minimum required by the slip-span GBD stencil. A short-span
interaction can therefore realize a different spacing from the requested
value; both values and the actual interval count are reported.

New reports use `openonda-cylinder-sensitivity/3`, with `requested_factors`,
`resolved_physical_factors`, and the actual solver `overrides`. Earlier reports
without this schema changed core radius and blend/release widths together with
particle spacing. Their measurements remain valid for those combined changes,
but cannot establish an isolated particle-spacing effect. The launcher preserves
such cohorts and requires a new study directory instead of reinterpreting or
overwriting them.

Configuration verification:

```bash
python -m pytest -q tests/coupler/test_cylinder_independent_spacing.py
```

The tests compare the study's resolved quantities against `build_case` for
individual factors and all distinct two-factor combinations. They do not run
CFD and do not qualify stability, accuracy, or a runtime recommendation. Actual
paired numerical runs and statistically resolved final wake measurements are
still required. Exchange-clock sensitivity continues to vary renewal cadence
and integration/exchange error together; see `renewal_cadence_scope.md`.
