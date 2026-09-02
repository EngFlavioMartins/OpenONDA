# Cartesian mesher Phase 5: generic boundary layers

*Date: 2026-09-02*  
*Status: native patch-normal layer construction is implemented for the current
serial contract; broad collision and quality qualification remains open.*

Selected wall patches are expanded from mapped patch faces with monotone
geometric layer fractions. The new layer block is assembled as native
hexahedral cells, shares its interface faces with the Cartesian core, and is
validated with the same topology and geometry routines. The API contains only
patch names, layer count, first height, and growth ratio; no axis, interface
width, or spanwise special case is exposed.

Evidence on smooth, sharp, and finite-body fixtures:

```text
python -m pytest -q tests/fvm/test_cartesian_config.py
9 passed
```

The current builder uses a local tangent-plane approximation at patch-face
corners and reports this through the measured wall-normal recovery distance.
Opposing-surface collision limits, feature termination policies, and
multi-resolution layer-height qualification are still release work.
