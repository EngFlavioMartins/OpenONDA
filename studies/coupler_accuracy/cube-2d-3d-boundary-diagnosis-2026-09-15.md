# Why the compact cube interface is more fragile in 3D than an exact 2D interface

This diagnosis uses existing fully 3D cube states and controlled operators. It
does not run a new coupled trajectory. The placement experiment described below
is configured in the cube tutorial but has not been run. The claimed
success of a particular 2D hybrid run cannot yet be checked: the current
cylinder tutorial and its reference are not a matched two-dimensional pair.
The exact 2D statements below follow from the governing equations and require
spanwise-invariant velocity, vorticity, transfer weights, and infinite-filament
induction. A finite extruded 3D cloud is not that limit.

## A dimensional failure mode that vanishes in exact 2D

For a planar incompressible field `u=(u_x(x,y),u_y(x,y),0)`, its vorticity is
`omega=(0,0,omega_z(x,y))`. If the FVM and VPM fields and the authority weight
`eta(x,y)` are spanwise invariant, then

```text
div[(1-eta) omega_V + eta omega_F] = partial_z omega_mix,z = 0,
(omega . grad) u = omega_z partial_z u = 0.
```

No streamwise/transverse vortex-vector component exists, so there is no
velocity-invisible longitudinal vorticity component to be activated by the
tested 3D stretching law. This is an exact dimensional identity; it does not
assert that 2D pressure, convection or a finite domain have no error.

For 3D divergence-free donor fields, by contrast,

```text
div[(1-eta) omega_V + eta omega_F]
    = grad(eta) . (omega_F - omega_V).
```

The cube transfers all three components over a weight varying near six outer
faces. Gaussian blob reconstruction can itself have nonzero divergence:
`div(sum_p Gamma_p zeta_p) = sum_p Gamma_p . grad(zeta_p)`. Thus vorticity
transfer and the particle strength representation have an extra 3D consistency
requirement. Physical 3D stretching is also nonzero; deleting it would change
the problem rather than correct this numerical mismatch.

The reference cube wake is demonstrably not planar. Among the fine FVM cells
with reverse flow in the old compact downstream box (`0.5<x<1.5`,
`|y|,|z|<1.25`), streamwise vorticity RMS is 48.2% of vector-vorticity RMS at
time 8 and 49.6% at time 15. Exact 2D flow has zero streamwise vorticity.
These values describe the *physical reference field*; they are not themselves
a measurement of numerical coupling error.

The current controlled cube evidence is quantitative:

| Fully 3D control | Observed result | Conclusion |
| --- | --- | --- |
| A velocity-invisible 3D gradient-vorticity perturbation, with matching zero-velocity donor | Spatial cube transfer creates 0.006376 U RMS; constant-weight control approaches roundoff | Spatial blending need not preserve a matching velocity. |
| The same perturbation in exact incompressible forward strain, with no body or boundary | Positive-transpose transport creates 0.010181 U²/D RMS velocity rate; gradient-preserving diagnostic approaches roundoff | The extra representation degree of freedom can become a velocity disturbance without a boundary or recirculation. |
| Saved cube particles at time 6, 196 downstream 3D probes | RMS of reconstructed vorticity minus induced-velocity curl is 21.4% of reconstructed-vorticity RMS | The inconsistency is present in the actual 3D state. |
| Actual t=6 particle-rate budget | Longitudinal term contributes +0.00627055 U³/D to a +0.00986705 U³/D local asymmetry-energy growth | It accounts for about 64% of this **instantaneous** downstream growth, not 64% of accumulated force/field error. |

See the [transfer control](cube-transfer-cause-2026-09-15.md),
[transport mechanism](cube-wake-mechanism-2026-09-15.md), and
[budget qualification](cube-boundary-placement-evidence-2026-09-15.md). These
controls demonstrate a 3D-specific vulnerability of this formulation. They do
not establish that a fresh matched 2D hybrid is error-free or that this one
mechanism explains every part of the cube's later discrepancy.

## What is actually failing at the compact cube face

The cube's mixed FVM condition prescribes normal velocity and the normal
gradient of tangential velocity. FVM reconstructs tangential velocity from
its interior state. It does not force both tangential velocity components to
equal the VPM values. On the corrected 3D trial at time 8, converged interface
sweeps give a maximum normal mismatch of `5.46e-16 U` while the maximum full
FVM–VPM velocity mismatch at the outflow is `0.588 U`. A convergence flag
therefore does not imply complete velocity agreement.

There is a separate **discrete cut-boundary error** even with reference FVM
data instead of VPM data. A matched short 3D cropped-FVM oracle inherits
16,936 cells from a 53,752-cell full mesh, starts from the same state, and
receives native reference normal flux and native face derivatives. At time
1.5, the flux-consistent mixed condition still has `0.005415 U` whole-volume
velocity RMS error and `+0.1068%` drag error. Frozen operator comparison
identifies outward-face `linearUpwind` convection as the largest remaining
cut-face discrepancy. This oracle contains no particle errors and establishes
an independent boundary/operator floor at that early time. It cannot be
subtracted from the developed-wake error because the flows, times and models
differ. [Native-face oracle](native-face-boundary-followup-3d.md).

The larger, growing outflow disturbance is also carried by the VPM source
field. At time 8 near `(1.44,-0.18,0)D`, VPM gives `u_z=0.35793 U`, while
the reference gives `0.0000831 U`; particles with `1.25<=x_p/D<=1.65`
contribute `0.32355 U` there, versus `0.00151 U` from the near-body
population. The VPM-only reference-fed control still grows an error without
FVM boundary feedback, and the boundary-free manufactured control reproduces
the numerical mechanism. This prevents attributing the whole discrepancy to
the mathematical mixed condition itself.

## Distance is a testable remedy, not a proved cure

The former compact run had its FVM face at `x=1.5D`, one diameter downstream
of the cube's rear face. Its transfer box ended earlier, at `x=1.25D`. The
fine reference's native 3D cell velocities show reverse flow downstream of
the cube at these times, inside `|y|,|z|<1.5D`:

| Reduced time | Furthest reverse-flow cell centre, x/D | Furthest `|y|` or `|z|`, /D | Reverse-flow volume, D³ |
| ---: | ---: | ---: | ---: |
| 8 | 1.788 | 0.742 | 0.994 |
| 15 | 2.373 | 0.787 | 2.122 |
| 20 | 2.868 | 0.833 | 2.852 |
| 30 | 3.295 | 0.878 | 3.963 |

The independently sampled centreline first reaches `x=1.5D` at time 3.25,
`x=2.5D` at time 16.25, and `x=3.0D` at time 22.75. A face at `x=3.5D`
would enclose the measured 3D reference reverse-flow region through time 20
with about 0.63D downstream margin. **Moving only the FVM face is
insufficient:** the former transfer box ended 0.25D before that face and
would continue handing the recirculation to VPM at x=1.25. The first placement
experiment moves the FVM face to x=3.75 and the transfer edge to x=3.5;
the latter then has the measured 0.63D margin at time 20. These reverse-flow
extents are cell-centre measurements at saved whole-second states; no
inter-sample maximum or new coupled-solver result is claimed. The reference
fine mesh has a nearBody refinement through `x=3.0D` and a wake refinement
beyond. The expanded coupled mesh retains the 0.06D near-body target through
x=3.0 and uses the reference wake target 0.12D beyond it.

The native cell measurement is reproducible without rerunning either solver:

```bash
python studies/coupler_accuracy/measure_cube_reference_reverse_flow.py \
  --pvd tutorials/coupled_fvm_vpm/02_cube_flow/reference_flow/solution/fine/fine.pvd \
  --times 8 15 20 30
```

The current cube setup uses these enlarged streamwise bounds and two mesh
refinement boxes matching the reference's near-body and wake targets. Its
existing mesher cache detects the changed specification and rebuilds; the
saved coupled results still belong to the old compact geometry.

Relocation alone is not yet a confirmed accuracy fix. In the *existing*
coupled run, the body-complete VPM centreline velocity differs from the fine
reference by `0.215 U` at x=1.5 and `0.235 U` at x=3.5 at time 8; by time 15,
the respective differences are `0.878 U` and `0.191 U`. These are interpolated
line samples of fully 3D solver states, not cross-sectional errors. The VPM
state at x=3.5 would change in a fresh expanded-domain run, so this observation
does not predict its outcome. It shows why simply picking a farther face from
the old run cannot prove the method is solved.

Moving the face also increases FVM work. The reference has 443,672 cell
centres within `[-1.5,3.75]x[-1.5,1.5]^2`, or 64.1% of its 692,604 cells.
This is a reference-mesh count, **not** the coupled mesher's new cell count or
a measured runtime. The new geometry can be a reasonable accuracy experiment,
but it cannot be called a cheaper solver before matched timing.

The expensive fixed-basis projection and full-support curl-transfer candidates
are study-only; neither is enabled in the production cube. The current run
does use converged interface sweeps. Dropping the
second sweep would remove one FVM advance and one renewal, but the existing
short 3D controls show a measurable interface replacement jump and a drag
response when sweeps are removed. Alignment passed the old t=15.66 health
crossing in a short continuation, but a later VPM-only accuracy control found
worse seam error with it. The clean enlarged case then rejected its first
application for 0.156% particle-strength growth against the unchanged 0.1%
limit. Alignment is disabled rather than weakening that scientific guard.
The measured developed-wake cost is dominated by VPM induction, so dropping
either existing option without qualification cannot by itself meet the
reference runtime. [Developed-wake cost](cube-developed-wake-cost-2026-09-15.md),
[interface control](interface-iteration-3d.md),
[health continuation](cube-health-2026-09-15.md).

## Limit of the available 2D comparison

The stored cylinder boundary oracle is a one-layer, quasi-2D FVM control with
different geometry and numerical choices. Its vorticity-mixed cropped solve
still has `0.007679 U` whole-volume velocity RMS difference after 100 steps;
supplying a mixed pressure gradient lowers this to `0.000451 U`. This shows
that cut-boundary and pressure errors do **not** vanish merely because a flow
is planar. The coupled cylinder STL has finite caps and a 12D span, while its
reference is a 1D-extruded slip-span case. Neither is a matched exact-2D vs
fully-3D performance/accuracy comparison. [Cylinder qualification](README.md).

The defensible thesis statement is that exact 2D kinematics remove the specific
longitudinal-vorticity and vortex-stretching channels demonstrated in the cube.
The compact 3D cube additionally hands an evolving recirculation region to the
particles and has a measured independent discrete cut-boundary error. The
existing evidence supports *why* this coupling is more fragile in 3D, but it
does not establish that increasing xmax alone removes the error or determine a
universal distance rule.
