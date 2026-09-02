# Cartesian mesher Phase 3: octree and native extraction

*Date: 2026-09-02*  
*Status: implemented through the established native adaptive extraction
engine and typed adapter.*

The Cartesian build combines the background, boundary, feature, and volume
requests into dyadic refinement bands. Exact lattice-aligned components are
removed component-wise for multiple disjoint bodies; general surfaces use
indexed triangle intersection and inside/outside classification. Existing
native extraction retains owner/neighbour topology, coarse/fine subfaces,
contiguous boundary ranges, deterministic point numbering, and 2:1 checks.

The coarse acceptance run is green:

```text
python -m pytest -q tests/fvm/test_cartesian_mesher_phase0.py
14 passed
```

The full required refinement-placement and convergence matrix, including
several translations and rotations at multiple resolutions, is not yet a
release qualification. The report below records the current coarse evidence.

| Fixture | Cells | Faces | Patches | Min volume | Max non-orthogonality | Max skewness | Max aspect ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rotated box | 1,600 | 5,513 | 7 | 3.22e-4 | 89.79° | 0.607 | 5.08 |
| ellipsoid | 1,640 | 5,704 | 7 | 6.40e-4 | 46.00° | 0.443 | 5.91 |
| torus | 3,432 | 11,540 | 7 | 6.64e-4 | 55.05° | 0.225 | 4.23 |
| finite wing | 792 | 2,875 | 7 | 8.06e-5 | 50.96° | 0.371 | 4.30 |
| two bodies | 1,524 | 5,264 | 8 | 1.95e-3 | 25.24° | 0.142 | 1.22 |
