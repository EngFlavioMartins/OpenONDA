# FMM fixed-capacity workspace audit

The CPU cylinder pilot at `h=0.08 m`, span `0.96 m` had about 4,230 active
particles, but the tutorial's hard particle ceiling was 1,000,000. Binding
`FMMInduction` allocates `FMMDeviceWorkspace` and its `TaichiTreecode` for the
ceiling before the first active stage. The FMM's own allocation formula in
`source/solvers/vpm/physics/induction/fmm/device.py` estimates **2.172 GiB**
at that ceiling: 915.5 MiB for coefficients, 976.6 MiB for interaction lists,
and about 332 MiB for the remaining fields. Particle, FVM, Python and Taichi
storage add to this figure. The measured four-rank pilot process tree peaked
at about 5.76 GB; it does not determine the one-rank peak.

This is an honest hard-capacity cost, not an extra public memory knob. Reducing
the hard ceiling without proving long-run population bounds could make the
production simulation fail later. The prepared paired cylinder clock study
therefore keeps the same ceiling and waits for host memory headroom.

A future general optimization could allocate an active-count geometric FMM
workspace while retaining the same hard particle ceiling and physical
algorithm. It requires explicit ownership of every capacity-scaled Taichi
field: about 36 in `FMMDeviceWorkspace` and 53 in the nested
`TaichiTreecode`. Replacing their Python objects currently does **not** free
their auto-root fields. Taichi documents `FieldsBuilder.finalize()` and
`SNodeTree.destroy()` for manual field lifetime management
([Taichi field layout documentation](https://docs.taichi-lang.org/docs/master/layout#manual-field-allocation-and-destruction)).
Growth would need synchronization, destruction before reallocation, and
invalidation of source-tree and target-revision caches. Empty-source target
queries, fixed-source scopes, staged RK fields, restart, and repeated solver
construction all need tests. Backend-specific destruction and memory-reuse
behavior on CPU, Vulkan and Metal remains unverified. A failed growth after
destroying the old tree must not silently accept a partially advanced step.
No lazy allocation or out-of-memory fix is claimed here.

See [the cylinder optimization comparison](cylinder_execution_report.md) for
the measured four-rank runtime and convergence evidence, and
`build/cylinder-final-pair/pair_plan.json` for the frozen one-rank trial inputs.
