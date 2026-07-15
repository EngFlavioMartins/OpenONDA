# boundaryLayer — laminar flat plate (Blasius)

Flow physics: viscous boundary-layer growth on a no-slip flat plate at
Re_L = U L / nu = 1e4 (laminar over the whole plate).

Validation (theory — Blasius 1908; Schlichting, *Boundary-Layer Theory*):

* wall-normal profiles collapse onto u/U = f'(eta), eta = y sqrt(U/(nu x)),
  sampled at x/L = 0.25, 0.5, 0.75 → `figures/blasius_profiles.png`
* skin friction Cf(x) = 0.664 / sqrt(Re_x) → `figures/skin_friction.png`

Run `./allrun.sh` (mesh is generated in-memory by `assets/mesh_plate.py`;
the bottom boundary is a slip run-in upstream of the leading edge, so the
layer starts at x = 0 as the similarity solution assumes).
