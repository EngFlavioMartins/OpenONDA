# Cartesian mesher Phase 4: surface recovery and quality

*Date: 2026-09-02*  
*Status: strict conformance gates are implemented; curved boundary-layer
construction and full surface-topology recovery remain intentionally blocked.*

The native Cartesian adapter no longer claims success for curved boundary
layers. Such requests fail during construction because the existing layer
builder starts from a Cartesian staircase and cannot guarantee a conformal
layer front or a valid core/interface stitch. Curved builds without layers
also pass through hard gates that measure every wall vertex against the input
STL and classify fluid-cell centres against the closed surface. Partial
snapping, unsnapped staircase points, and fluid centres inside the body are
therefore rejected rather than reported as a successful mesh.

The strict-gate tests pass:

```text
python -m pytest -q tests/fvm/test_cartesian_config.py tests/fvm/test_cylinder_reference_cartesian.py
13 passed
```

The former rotated planar recovery path is retained only as a regression
diagnostic: at `h=0.125` it produces a measured maximum wall distance of
approximately `0.15`, and the strict gate rejects it. A conformal surface
recovery and layer/front stitching implementation is required before curved
boundary layers can be enabled in `CartesianMesher`.
