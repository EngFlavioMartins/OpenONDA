# Cartesian mesher Phase 2: surface model and features

*Date: 2026-09-02*  
*Status: implemented and covered by the current deterministic fixture tests.*

`STLSurface` loads ASCII and binary STL data, retains an immutable triangle
array and source SHA-256, and validates finite coordinates, non-degenerate
triangles, watertight edge incidence, edge winding, connected-component
volume, and disconnected-component orientation. `SurfaceIndex` provides a
spatial grid for triangle overlap and expanding indexed nearest-point
queries. Ray-parity inside/outside classification retries deterministic ray
directions when a point is ambiguous. `classify_features` reports adjacent
face-angle edges and their corner vertices.

Evidence:

```text
python -m pytest -q tests/fvm/test_cartesian_config.py tests/fvm/test_cartesian_mesher_phase0.py
23 passed
```

The valid fixture families are rotated planar, smooth closed, concave closed,
finite mixed-feature, and two disjoint bodies. Open-edge, non-manifold,
inverted-component, and degenerate-triangle fixtures fail during typed surface
construction. Self-intersection detection is not yet a qualified input
feature and remains an explicit release-support item.
