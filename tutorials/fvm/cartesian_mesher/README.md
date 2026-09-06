# Built-in Cartesian mesher

This tutorial is a complete, external-mesher-free workflow:

```bash
cd tutorials/fvm/cartesian_mesher
python setup.py
```

`setup.py` reads `assets/object.stl`, defines the surrounding box and named
patches, applies one rectangular and one named-wall refinement, builds the
native mesh, and runs a short twenty-step FVM case.  The solver factory writes
`solution/mesh.npz` and `solution/mesh.vtu`; the former can be loaded again with
`create_fvm_solver(..., mesh="solution/mesh.npz")`.

The example intentionally keeps mesh construction and boundary-condition
physics separate.  Replace `assets/object.stl` with another supported closed
STL and adjust the box or sizes in `create_mesher()` when the object requires a
different resolution.
