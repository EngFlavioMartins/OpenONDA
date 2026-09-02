# Cartesian mesher Phase 1: typed configuration

*Date: 2026-09-02*  
*Status: implementation gate passed for the typed configuration and native
adapter scope.*

The public `openonda.fvm` API now exposes immutable `BoxDomain`, `BoxPatches`,
`STLSurface`, `BoxRefinement`, `SphereRefinement`, `ConeRefinement`,
`LineRefinement`, `FeatureRefinement`, `BoundaryLayers`, and
`CartesianMesher` objects. Requested sizes are combined by a minimum size
field, clamped by `min_cell_size`, and recorded with their effective dyadic
level in `GenerationReport`.

The adapter preserves configured outer and wall patch names, returns the
existing native face-based mesh dictionary, and supports `build()` and
`__call__()` equivalently. The input STL bytes are hashed without modifying
the source file.

Evidence:

```text
python -m pytest -q tests/fvm/test_cartesian_config.py tests/fvm/test_cartesian_mesher_phase0.py
23 passed
ruff check source/solvers/fvm/mesh/cartesian source/solvers/fvm/mesh/adaptive_cartesian.py
All checks passed!
pyrefly check source/solvers/fvm/mesh/cartesian source/solvers/fvm/mesh/adaptive_cartesian.py
INFO 0 errors
```

The old adaptive and geometry-specific public paths remain for regression
compatibility until the migration phase. This phase does not claim that the
legacy paths have been deleted.
