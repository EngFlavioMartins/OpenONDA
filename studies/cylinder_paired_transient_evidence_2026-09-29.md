# Cylinder exchange-clock pair: accepted transient through 0.4 s

Two genuine one-rank CPU coupled runs from the installed `fc921ec0` wheel ended at
the common physical time 0.4 s with runner exit code 0 and complete manifests.
The only declared difference was exchange `dt`: 0.02 or 0.04 s. Both used
`hxy=dz=hp=0.08D`, span `0.96D`, Aitken interface iteration with three sweeps,
and the same 100,000-particle hard capacity. Core radius `0.08D`, blend width
`0.48D`, release width `0.16D`, transfer method `buffered_m4_renewal`, geometry
hash `ee241f6c06c1723eeebf44d6e5f3eaf5b4c7f372b23d357bc4a588b095b74cc3`,
and setup hash `cff3da2488182d2701aaa6d7083a21f3d841bf08c44fad39b2f475e86f74df44`
were held fixed. The [machine-readable report](cylinder_paired_transient_evidence_2026-09-29.json)
preserves exact commands, source hashes, manifest and diagnostic hashes, and all
matched profile errors.

| Exchange dt [s] | Accepted intervals | Interface-converged | Largest accepted scaled residual | Endpoint Cd | Endpoint Cl | Max transferred particles |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 20 | 18/20 | 12.267 | 1.76730 | -0.005653 | 4,899 |
| 0.04 | 10 | 10/10 | 0.917 | 1.53734 | -0.005394 | 4,227 |

The unchanged interface gate is normal and gradient RMS at most `1e-5` each,
equivalent to scaled residual at most one. At `dt=0.02`, accepted startup steps
1 and 2 missed this gate with scaled residuals 12.267 and 1.02865; the remaining
18 intervals converged. At `dt=0.04`, all ten converged. The terminal Cd
difference is 0.22995, about 15.0% of the `dt=0.04` value. These are transient
endpoints and cannot be used as settled drag or temporal convergence evidence.

Profiles were matched on both physical time and spatial position at t=0.2 and
0.4 s. The table gives the largest absolute pointwise difference and relative
L2 difference for `dt=0.02` minus `dt=0.04`:

| Profile | Quantity | Max absolute | Relative L2 |
| --- | --- | ---: | ---: |
| FVM centreline | `u_x` | 0.00770 | 0.660% |
| FVM centreline | `omega_z` | 0.15580 | 1.338% |
| FVM transverse x=1 | `u_x` | 0.00959 | 0.644% |
| FVM transverse x=1 | `omega_z` | 0.00163 | 5.578% |
| VPM transverse x=1 | `u_x` | 0.01375 | 0.515% |
| VPM transverse x=1 | `omega_z` | 0.01767 | 41.846% |
| VPM transverse x=2 | `u_x` | 0.00534 | 0.225% |
| VPM transverse x=2 | `omega_z` | 0.000281 | 135.313% |
| VPM transverse x=4 | `u_x` | 0.00161 | 0.080% |
| VPM transverse x=4 | `omega_z` | 0.0000149 | 100.634% |

The large downstream relative vorticity errors have small absolute values;
for example at VPM x=4, the `omega_x` relative L2 error is 147.06% while its
maximum absolute difference is only `2.57e-7`. Conversely, the centreline
`omega_z` absolute difference of 0.156 is material even though its relative L2
is 1.34%. Neither statistic alone describes the spatial disagreement.

Conservation diagnostics remained finite: the largest corrected boundary-flux
mismatch was `4.72e-16` versus `3.33e-16`, maximum renewal conservation error
`1.31e-8` versus `1.34e-8`, and maximum normalized GBD vortex-strength residual
`2.27e-9` versus `1.65e-9` (`dt=0.02` versus `0.04`). The exact additional
linear/angular impulse residuals are retained in the JSON.

| Exchange dt [s] | Warm intervals in (0.12, 0.4] | FVM [s] | VPM [s] | Transfer [s] | VPM boundary condition [s] | Warm total [s] | Full startup-to-finish [s] | Peak process-tree RSS [GiB] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 14 | 629.05 | 300.96 | 261.93 | 91.88 | 1283.83 | 2639.76 | 1.575 |
| 0.04 | 7 | 667.20 | 164.84 | 123.96 | 39.69 | 995.68 | 2110.99 | 1.327 |

The physical warm window is the same, but both jobs shared a host under
nonidentical load. Wall times are observations, not a controlled speed ratio.
Changing the exchange clock also changes integration and renewal cadence, so
this pair does not isolate independent injection frequency. The horizon ends
before a developed wake: no stationary Cd, lift RMS, Strouhal, grid or
12-hour qualification follows. The 100,000-particle ceiling applies only to
these bounded trials.
