# cylinderIBM — flow past a circular cylinder with the Immersed Boundary Method

Validation/monitoring case for the FVM solver's discrete direct-forcing IBM
(Pinelli et al. 2010, as implemented for FV PISO solvers by Constant et al. —
`docs/literature/Constant2016.pdf`; design in
`docs/plans/2026-07-fvm-ibm-design.md`).

The cylinder (D = 1) is **not** in the mesh: it is a ring of Lagrangian markers
on a rectilinear Cartesian grid (uniform spacing `h` in a core box around the
body, geometrically stretched to the far field). Each time step the momentum
predictor runs force-free, the velocity is interpolated to the markers with the
3-point Roma–Peskin kernel, the direct-forcing term `F = (U_target − I[u*])/Δt`
is spread back (Pinelli quadrature) and the predictor is re-solved with the
force, followed by multidirect residual-forcing iterations and the usual PISO
pressure correctors.

## Run

```bash
./allrun.sh                 # Re = 30 steady case, h = D/16, t_end = 60
./allplot.sh --Re 30        # figures + pass/fail vs reference values
```

Unsteady vortex-shedding variant:

```bash
python cylinderIBM_setup.py --Re 100 --end-time 150 --h 0.05
./allplot.sh --Re 100
```

## Quality monitors (references: Constant et al. 2017, Tables 2–3)

| Re  | quantity                | reference      | where |
|-----|-------------------------|----------------|-------|
| 30  | drag coefficient C_D    | 1.74 – 1.80    | `figures/forces_cylinder.png`, stdout of allplot |
| 30  | recirculation length L/D| 1.55 – 1.70    | `figures/wake_centreline.png` |
| 100 | mean C_D                | 1.35 – 1.38    | forces figure |
| 100 | Strouhal number         | 0.164 – 0.165  | forces figure (lift FFT) |
| any | marker no-slip error    | → small (≪ U∞), decreasing with h | forces figure, bottom panel; logged per step in `solution/ibm_forces_history.csv` |

Expected behaviour of the monitors:

- **slip** is the IBM-specific health signal `max_s |u(X_s)|`. It should sit
  orders of magnitude below U∞ (≈ 1–3×10⁻³ here) once the startup transient
  passes. If it grows or slowly sawtooths, the time step is too large: the
  direct-forcing loop needs **both** Co ≤ 0.5 **and** Fo = ν·Δt/h² ≲ 0.1
  (the driver caps Δt by `--max-fo 0.1` automatically; see the design doc for
  the experiment matrix behind this).
- **C_D converges from above** with mesh refinement: the diffuse-interface
  kernel enlarges the effective cylinder diameter by ≈ h, so coarse grids
  over-predict drag (paper Fig. 8 shows the same 1st–2nd-order approach from
  above). Measured at Re = 30: h = D/10 gives C_D ≈ 1.98 and L/D ≈ 1.84
  (≈ +11 % — and rescaling by the effective diameter D + h puts both back in
  the reference bands: 1.80 and 1.67). Expect ≈ +5–8 % at the default
  h = D/16. The domain here (24D × 16D) is also smaller than the paper's
  64D × 32D, adding a small blockage over-prediction.

## Files

- `cylinderIBM_setup.py` — case driver (mesh in memory, no gmsh needed).
- `assets/mesh_rectilinear.py` — graded rectilinear 2D mesh generator.
- `assets/plot_forces.py` — C_D/C_L/slip histories + Strouhal, reference bands.
- `assets/plot_wake.py` — wake centreline u_x, recirculation length L/D.
- `assets/plot_fields.py` — |U| and ω_z snapshots with the marker ring.
- `solution/ibm_forces_history.csv` — time, per-body forces, C_D/C_L, slip.
- `solution/ibm_markers.csv` — marker coordinates.
