# Panel solver production qualification

Qualification date: 2026-08-25. The machine-readable evidence is
[`benchmarks/panel-solver-qualification-macos-arm64.json`](benchmarks/panel-solver-qualification-macos-arm64.json),
with the optional 4,000-panel extension in
[`benchmarks/panel-solver-scaling-4000-macos-arm64.json`](benchmarks/panel-solver-scaling-4000-macos-arm64.json).
The source identity and dirty-worktree flag are recorded in both artifacts.

## Verdict by capability

| Capability | Status |
| --- | --- |
| Moving, multi-body Neumann impermeability solver | **Production qualified** |
| Static potential-flow pressure and loads | **Qualified** |
| Per-body far-field acceleration | **Qualified**, default acceptance `5.0` |
| Moving or unsteady potential-flow loads | **Unsupported** |
| General VPM-coupled pressure and loads | **Not qualified** |
| VPM coupling with Dirichlet panels | **Rejected by the API** |
| Standalone Dirichlet formulation | **Experimental** |

“Production qualified” applies to the boundary solver, not to moving-body
loads. A moving body is allowed to enforce impermeability and inject its panel
velocity into VPM, but the existing steady Bernoulli load path deliberately
raises rather than reporting physically incomplete forces.

## Qualification gates

The campaign is executable with:

```bash
python scripts/benchmarks/benchmark_panel_solver.py --mode all \
  --scaling-total-panels 256 512 1000 2000 \
  --bodies 1 2 4 8 --repeats 3 --targets 512 \
  --output docs/benchmarks/panel-solver-qualification-macos-arm64.json
```

The run passed all five machine-readable gates: physical convergence,
near-contact domain, far-field sweep, projected-device/CPU-oracle agreement,
and representative scaling.

### Physical convergence

Four non-icosahedrally-symmetric, near-uniform sphere meshes avoid the
artificial exactness of the base icosahedron. Errors are relative L2 norms;
force error is normalized by dynamic pressure times projected area.

| Panels | `h` | Surface-speed error | `Cp` error | Force error | `cond2(A)` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 80 | 0.3815 | 1.656e-2 | 6.080e-2 | 1.242e-3 | 1.957 |
| 160 | 0.2749 | 1.097e-2 | 3.726e-2 | 1.940e-3 | 1.963 |
| 320 | 0.1963 | 4.283e-3 | 1.779e-2 | 5.291e-4 | 1.967 |
| 640 | 0.1394 | 3.794e-3 | 1.341e-2 | 4.283e-4 | 1.974 |

Surface-speed and pressure errors decrease monotonically. The symmetric
two-body probe self-error decreases from `4.73e-2` to `9.78e-3` to `1.37e-3`
before the 1,280-panel reference. Across the convergence campaign, the
projected Taichi solution differs from reusable CPU null-space QR by at most
`1.22e-12` in the source-strength norm.

### Near-contact domain

Two spheres were tested at two resolutions for `g/h = 8, 4, 2, 1, 0.5`.
Every case had finite strengths, passed constrained optimality and flux,
matched the CPU oracle, and remained well conditioned. At the closest tested
gap, the maximum condition number was `3.01`; the maximum-strength variation
between the two resolution levels was `2.73%`.

The supported tested domain is therefore:

```text
g/h >= 0.5
```

Smaller gaps are not claimed. Collocation residuals alone cannot qualify an
unresolved near-singular interaction.

### Far-field acceleration

Acceptance values `2, 3, 4, 5, 6, 8, 10` were swept over single- and four-body
geometries using targets that exercise mixed exact and multipole paths. The
gate was relative L2 error at or below `5e-4` for every geometry.

At acceptance `5.0`, the single-sphere error was `4.82e-7` with a `3.13x`
speedup; the four-sphere error was `3.97e-4` with a `3.77x` speedup. Acceptance
`4.0` failed the four-body error gate, so `5.0` is retained empirically.

### Scaling and solver choice

Timings below are steady-state seconds on the recorded macOS arm64 host with
Taichi’s CPU backend. “Reuse” is cached CPU QR; “refactor” rebuilds the CPU
null-space QR for a changed operator; “projected” is the matrix-free projected
CGLS implementation. The projected path was run on the CPU backend here, so
these are not GPU-hardware performance claims.

| Panels | Bodies | CPU reuse | CPU refactor/rebuild | Projected rebuild | CPU-oracle difference |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 2 | 0.0039 | 0.0106 | 0.0109 | 2.13e-13 |
| 512 | 4 | 0.0107 | 0.0479 | 0.0210 | 6.53e-13 |
| 1,000 | 4 | 0.0238 | 0.1505 | 0.0400 | 9.92e-13 |
| 2,000 | 8 | 0.0553 | 0.7388 | 0.1049 | 4.99e-13 |
| 4,000 | 4 | 0.1552 | 6.7333 | 0.3856 | 2.62e-13 |

At 4,000 panels, the retained CPU factors use `383.5 MB` and peak process RSS
was `1.30 GB`. Static CPU QR remains the default because a cached RHS solve is
small relative to surface evaluation and avoids iterative work. For moving
geometry, `BICGSTAB_GPU` selects projected CGLS for Neumann problems and avoids
the cubic refactor; the CPU QR result remains the reference oracle.

## Required regression evidence

The test suite independently covers:

- freestream/VPM incident/body velocity separation and Galilean invariance;
- pure rotation and composed translation plus rotation about a non-origin centre;
- AIC invalidation, block invariance, and fresh-rebuild equivalence for moved bodies;
- unequal three-body rotation and body-insertion permutations;
- two-body strength, pressure, force, and moment symmetry;
- simultaneous per-body zero flux and constrained KKT optimality;
- CPU QR/projected CGLS agreement for single- and multi-body cases;
- f32/f64 behavior, analytical sphere pressure/velocity/force, exact and
  accelerated panel-to-target evaluation, and VPM particle coupling.

The non-fatal Taichi cache-lock warning seen on this host is environmental and
did not change test or benchmark outcomes.
