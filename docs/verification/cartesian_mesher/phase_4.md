# Cartesian mesher Phase 4: surface recovery and quality

*Date: 2026-09-02*  
*Status: transactional recovery and native quality gates are implemented;
full conformal-quality qualification remains open.*

Surface recovery evaluates every candidate wall-point movement against all
referencing cell volumes and wall-face area. Direct mappings are accepted;
otherwise a bounded interpolation searches for the largest valid movement.
Collapsed faces and non-positive cells are never returned. The mesh-generation
metadata records attempted, accepted, partial, and rejected movements. Sharp
feature extraction is evaluated before sizing, and the quality snapshot uses
the repository's authoritative geometry and validation routines.

The current matrix and repeated-build tests pass:

```text
python -m pytest -q tests/fvm/test_cartesian_config.py tests/fvm/test_cartesian_mesher_phase0.py
23 passed
```

The rotated planar fixture still has a measured maximum wall distance of
approximately `0.15` at `h=0.125`, and the coarse matrix reaches about 89.8°
maximum non-orthogonality. These are recorded limitations, not hidden
successes; tighter production quality limits and multi-resolution surface-
distance convergence are still required before release certification.
