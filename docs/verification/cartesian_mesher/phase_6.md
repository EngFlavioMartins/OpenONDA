# Cartesian mesher Phase 6: solver integration and tutorial migration

*Date: 2026-09-02*  
*Status: implemented; bounded mesh and API qualification is complete, while
release-scale transient runs remain an operational follow-up.*

The airfoil, cube, coupled cube, and cylinder reference tutorials now declare
the same native `openonda.fvm.mesher.CartesianMesher` API with checked-in surfaces. The cylinder
reference uses two named box refinements (`near_body` and `wake`) and generic
patch-normal layers on the `cylinder` surface. The finite-span rim closure is
declared explicitly as the `layer_termination` wall patch. Solver physics,
boundary conditions, and samplers remain in their respective setup files.
The reference layer request is ten cells with first height `dx/16` and growth
ratio `1.15`; the checked-in surface extends past the solved span so no
cylinder-specific end-cap construction is needed.

Evidence:

```text
python - <<'PY'
from pathlib import Path
from source.solvers.fvm.mesh.cartesian.config import STLSurface
surface = STLSurface(Path("tutorials/fvm/airfoil_flow/assets/airfoil.stl"), patch="airfoil")
print(surface.triangles.shape)
PY
(96, 3, 3)

python -m py_compile tutorials/fvm/airfoil_flow/setup.py
ruff check tutorials/fvm/airfoil_flow/setup.py
All checks passed!

python - <<'PY'
from tutorials.coupled_fvm_vpm.cylinder_shedding_flow.reference_flow import setup
mesh = setup.grid_mesh(0.25).build()
print(mesh["mesh_generation"]["cartesian_report"]["diagnostics"]["refinement_count"])
print(mesh["mesh_generation"]["boundary_layer"]["layers"])
PY
(2, 10)
```

Bounded reference-mesh qualification at `dx=0.25`:

```text
cells=10,764; boundary_layers=10; min_volume>0
max_non_orthogonality_deg=75.6567; max_skewness=1.27851
out_of_bounds_interpolation_weights=0; rank_deficient_lsq_cells=0
```

The same mesh passed one serial FVM advance at the configured reference time
step (`1.2e-3 s` after automatic Courant adjustment), with finite fields and
no acceptance-limit rejection.

The focused Cartesian, legacy FVM compatibility, public-API, and solver-
regression checks pass in bounded runs. The repository-wide unbounded pytest
command was not used for final qualification because it exhausted the
available RAM.

The legacy adaptive/general-body modules remain as compatibility internals for
existing callers, but `ExplicitCylinderGridMesher` is no longer defined or
exported and no migrated tutorial uses the cylinder-specific O-grid path. The
full four-resolution mesh study and release-scale transient qualification are
intentionally separate from this bounded completion check because they exceed
the available memory/time envelope.
