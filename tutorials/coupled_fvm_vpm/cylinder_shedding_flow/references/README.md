# Cylinder-wake references

The tutorial keeps local copies only when an openly accessible PDF could be
downloaded and validated. Publisher/DOI links remain the bibliographic
authority.

| Reference | Why it is used | Local copy / link |
|---|---|---|
| Henderson & Barkley (1996), *Secondary instability in the wake of a circular cylinder* | Establishes the secondary three-dimensional instability at approximately `Re=188.5`; this motivates `Re=150`. | [local PDF](henderson_barkley_1996_secondary_instability.pdf), [CaltechAUTHORS record](https://authors.library.caltech.edu/records/qwzbs-hxs18) |
| Williamson (1996), *Vortex Dynamics in the Cylinder Wake* | Regime map, shedding physics, end effects, and experimental context. | [publisher page](https://doi.org/10.1146/annurev.fl.28.010196.002401) — publisher access required, so no local copy is committed |
| Karniadakis & Triantafyllou (1989), *Frequency selection and asymptotic states in laminar wakes* | Supports using a saturated periodic wake and a force/probe frequency estimate. | [publisher page](https://doi.org/10.1017/S0022112089000431), [public PDF mirror](https://electronicsandbooks.com/edt/manual/Magazine/J/Journal%20of%20Fluid%20Mechanics/1989%20Volume%20199/S0022112089000431.pdf) |
| Posdziech & Grundmann (2007), *A systematic approach to the numerical calculation of fundamental quantities of the two-dimensional flow over a circular cylinder* | Shows why force coefficients require independent resolution and domain-size studies; Strouhal agreement alone is insufficient. | [publisher page](https://doi.org/10.1016/j.jfluidstructs.2006.09.004), [public PDF mirror](https://www.electronicsandbooks.com/edt/manual/Magazine/J/Journal%20of%20Fluids%20and%20Structures/2007%20Volume%2023/3/479-499.pdf) |
| Barkley & Henderson (1996), *Three-dimensional Floquet stability analysis of the wake of a circular cylinder* | Independent stability map and critical spanwise wavelength. | [local PDF](barkley_henderson_1996_floquet_stability.pdf), [author-hosted PDF](https://warwick.ac.uk/fac/sci/maths/people/staff/dwight_barkley/home_page/papers/barkley_-_1996_-_journal_of_fluid_mechanics.pdf) |

The solved four-diameter span uses slip end planes and excludes the remote STL
caps, but it is still not treated as numerically identical to an infinite or
periodic cylinder. The papers are sanity bounds and methodology support; the
converged fully meshed OpenONDA result is the quantitative authority for the
coupled run.
