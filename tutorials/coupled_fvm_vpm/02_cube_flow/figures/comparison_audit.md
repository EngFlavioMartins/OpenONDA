# Cube comparison audit

Reference: reference_flow/samples/fine/ and reference_flow/solution/fine/.
Comparison ends at t=3 s. Reference data after
this time are not used in the figures. No simulation was advanced by plotting.

## What is matched

- Density, viscosity, initial conditions, FVM spatial/time schemes, turbulence
  closure, PIMPLE correctors/relaxation, linear tolerances and force definitions
  agree in the saved configurations.
- Both meshes have cube-adjacent Cartesian spacing
  0.045 m (requested fine target: 0.06 m).
  The coupled mesh has 303,264 cells and
  3,456 cube faces; the reference has
  692,604 cells and 3,360 cube faces.
  Their fitted wall cells and outer boundaries are not identical.
- Both FVM velocity fields use the existing 3D affine reconstruction with 12
  native volume-centroid neighbours and its documented IDW fallback. MPI
  global cell IDs are checked for complete, unique coverage.
- Only coincident saved physical states are used (absolute tolerance 1e-9 s).
  The reference saves full fields at 1 s intervals, so matched profile/field
  figures use that cadence. No interpolation between times is performed.
  The original force histories retain their 0.05 s sampling.
- Forces are the raw pressure-plus-viscous cube-wall forces, normalized by
  0.5 rho U_inf^2 D^2, with rho=1, U_inf=1 and D=1. No smoothing, outlier removal
  or drag-axis clipping is used.

## Meaning of the differences

Colour shows 100 ||u_test - u_comparison||_2 / U_inf, including u_x, u_y and u_z.
The velocity panels themselves show u_x/U_inf. Normalizing by freestream speed
avoids division by a vanishing local velocity. The word “difference” is used
because a finite-mesh reference is not an exact solution.

field_differences.csv records the area-weighted sampled RMS, sampled maximum,
valid-node count and covered area for each figure. Each valid grid rectangle
distributes one quarter of its area to each vertex. The RMS is a quadrature
estimate on this sampled z=0 plane, not a 3D volume norm. All three fields must
be finite at a point, giving every pairing the same support; the cube and
rectangles with missing corners are excluded.
Unresolved strips adjacent to the wall are not counted as zero error. Covered
area is reported so this limitation is visible.

Error contours use bilinearly interpolated velocity vectors; their difference
is formed before taking the vector norm. Metrics are calculated on the original
sample grid (spacing about 0.12 m), independently of display resolution.
Contours cannot recover structures absent from those samples. No nearest-point
extrapolation or 95th-percentile colour clipping is used. Colour ranges are
shared between the two velocity panels of each figure and may change with time.

reference_fvm_fields_* compares the primary near-body solution with fine FVM.
reference_vpm_fields_* and velocity_fields_* diagnose VPM in the overlap
region, where VPM is auxiliary. They are not whole-domain hybrid error maps.
The line profiles include the sampled outer wake. A z=0 section of 3D fields
does not establish accuracy everywhere in three dimensions.

At the latest compared time, t=3 s:

| Comparison | RMS [% U_inf] | Sampled max [% U_inf] | Area [D^2] |
|---|---:|---:|---:|
| Coupled FVM / VPM (overlap consistency) | 3.192 | 17.586 | 7.5456 |
| Reference FVM / VPM (auxiliary overlap field) | 4.872 | 19.236 | 7.5456 |
| Reference FVM / Coupled FVM (primary near field) | 3.647 | 25.438 | 7.5456 |


These are instantaneous sampled differences, not time-averaged error estimates.

## Reference drag anomaly and remaining validation limits

The largest saved reference Cd after t=1 within the compared interval is
2.30890757 at t=1.2 s, with accepted dt=0.000126489445 s.
reference_force_audit.* shows raw Cd and the accepted timestep; the vertical
line marks this sample. comparison_audit.json records neighbouring pressure
extrema where solver diagnostics are available. These are observations, not
a completed diagnosis of the pressure/timestep algorithm.

At this sample the pressure span increases by a factor of 7.14, and the timestep decreases by a factor of 120 relative to the preceding step.
This diagnostic does not establish that the reference is converged or accurate.
The diagnostic flags simultaneous pressure-span growth and timestep reduction
above a factor of ten; this is a screening rule, not a convergence criterion,
and it never filters the plotted data. Hiding suspect points would invalidate
the comparison. The fine run uses adaptive timesteps and the coupled FVM uses fixed
0.01 s steps, so temporal error is not isolated even though the schemes match.
The saved fine grid alone supplies no mesh/time convergence or statistical
uncertainty estimate. These figures can document the comparison and anomaly;
they do not yet support a claim of validated hybrid accuracy.

## Figure contract

Vector PDFs are exactly 125 mm wide, with embedded NewPX text/math fonts at
10.95 pt for main text. Include at natural size. PNG previews are 400 dpi.
All axes use equal outer side margins; the shared thesis validator checks font
sizes, a minimum 5 pt text-to-canvas clearance and text overlap before saving.
Line widths are 1.1 pt (primary), 1.0 pt (reference), and 0.5 pt (axes).
Captions belong in the thesis/paper; identify the slice, time, normalization,
common support, auxiliary-field status and finite-reference limitations.
