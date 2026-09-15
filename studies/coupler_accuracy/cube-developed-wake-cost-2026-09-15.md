# Developed-wake cost of the fine-resolution cube coupler

The saved coupled state at `t=15.65` does **not** meet the project's cost
criterion. The same-resolution fine FVM reference averages 2.147 s per 0.01 s
physical step over `t=12`–16. The coupled run averages 17.188 s per 0.01 s
step over steps 1400–1565 (`t=14`–15.65), about 8.0 times as long. These
measurements come from the native records listed below; the jobs ran at
different wall-clock times, so the ratio is a historical comparison, not a
controlled concurrent benchmark.

| Component, steps 1400–1565 | Mean wall time per coupled step |
| --- | ---: |
| VPM advancement | 12.769 s |
| FVM boundary query | 0.721 s |
| FVM advancement, including two interface sweeps | 1.471 s |
| Particle renewal and interface refresh | 2.227 s |
| Total | **17.188 s** |

The small FVM contains 303,264 cells versus 692,604 in the fine reference.
It is cheaper even after two interface sweeps, but only by about 0.676 s per
common 0.01 s step. The transfer phase alone exceeds that saving by over three
times. To beat the reference with the present FVM cost, **all** VPM, boundary,
and transfer work together would have to take less than 0.676 s; the measured
sum is 15.717 s. A modest kernel optimization cannot close this gap.

The particle population also contradicts the assumption of a small cloud. At
steps 1400–1565, transfer leaves an average 137,364 particles: about 24,810
new near-body renewal particles and 112,555 preserved outer particles. The
`t=15.5` native VPM backup contains 148,538 particles. The VPM is solving a
resolved three-dimensional wake, not just carrying a few vortices.

## Fixed-state induction screen

A read-only Metal treecode screen used exactly the `t=15.5` saved positions,
strengths, and Gaussian core radii. It built one LBVH, then evaluated all
148,538 particle velocities and Jacobians at each opening angle. Angles were
changed only for the screen; no production configuration or saved state was
modified. Relative RMS differences use the production angle 0.1 as a numerical
baseline, **not** an exact solution or the fine FVM field.

| Opening angle, order 1 | Evaluation time | RMS velocity change, all targets | RMS velocity change, near-body targets |
| --- | ---: | ---: | ---: |
| 0.1 | 10.91 s | baseline | baseline |
| 0.15 | 5.13 s | 0.346% | 0.173% |
| 0.2 | 2.95 s | 0.619% | 0.331% |
| 0.3 | 1.55 s | 1.420% | 0.863% |

The near-body mask is `|x|,|y|,|z| ≤ 1.5 m`, containing 26,156 particle
targets. At angle 0.15, its Jacobian RMS change is 0.182%; at 0.3 it is
1.422%. These are fixed-state induction errors. They do not qualify a new
coupled trajectory or show force/profile preservation. The field-error cost
of simply loosening the opening angle makes it unsuitable as an untested
accuracy-preserving fix.

Morton-sorting the particle target schedule at the unchanged angle 0.1 took
9.52 versus 11.85 s in a second fixed-state pass, a 1.24-times traversal
speedup. Every saved output velocity and Jacobian matched the unsorted result
bit for bit. This is a legitimate scheduling optimization, but it is much too
small to meet the cost criterion on its own. The production case is left
unchanged until an end-to-end timing and scientific qualification is available.

The source records are
`tutorials/coupled_fvm_vpm/02_cube_flow/solution/coupler_diagnostics.jsonl`,
`tutorials/coupled_fvm_vpm/02_cube_flow/reference_flow/solution/fine/performance.jsonl`,
and `tutorials/coupled_fvm_vpm/02_cube_flow/solution/backups/vpm_001550.h5`.
The isolated screen ran on the same MacBook's Metal backend with an isolated
Taichi cache. Its successive timings should be read as a local speed screen;
external system load was not controlled. A sandboxed attempt could not
initialize Metal and made no numerical measurement.

The [fixed-state screen](profile_cube_treecode_fixed_state.py) reproduces the
field/timing comparison without advancing the simulation:

```sh
python studies/coupler_accuracy/profile_cube_treecode_fixed_state.py \
    --checkpoint tutorials/coupled_fvm_vpm/02_cube_flow/solution/backups/vpm_001550.h5
```

The route to a cheaper solver must address both the growing outer-wake
representation and the frequency/cost of full particle induction and exchange,
while retaining the matched near-body resolution and 3D conservation tests.
The current fixed-basis accuracy candidate is not production-qualified, and
this cost screen does not change that status.
