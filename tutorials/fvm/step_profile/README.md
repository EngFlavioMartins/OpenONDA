# Backward-facing-step flow

Laminar flow through a 2:1 channel expansion forms a recirculation region behind the step. See [FVM units](../../../docs/fvm.md#physical-model-and-units), [boundary conditions](../../../docs/fvm.md#boundary-conditions), and [time-step controls](../../../docs/fvm.md#time-and-discretisation).

From the repository root:

```bash
cd tutorials/fvm/step_profile
./allrun.sh
./allplot.sh
```

`./allrun.sh` clears previous results; `./allcontinue.sh` resumes them. Edit [setup.py](setup.py) to change the physical and mesh parameters.

## Physical setup

The step height is $h=1$ m. The inlet channel occupies $h\le y\le2h$ over $-4h\le x<0$; downstream, $0\le y\le2h$ over $0\le x\le20h$.

The bulk inlet velocity is $U_b=1$ m/s, density is $1$ kg/m³, and $Re_h=U_bh/\nu=100$, giving $\nu=0.01$ m²/s. The inlet profile is

$$
u(y)=6U_b\eta(1-\eta),
\qquad \eta=\frac{y-h}{h},
\qquad v=w=0.
$$

Walls are no-slip, the outlet fixes $p/\rho=0$, and the single spanwise cell has `empty` faces. The initial field uses parabolic profiles consistent with the channel heights on each side.

`N_UPSTREAM=24` and `N_DOWNSTREAM=120` set streamwise resolution. `N_HEIGHT=16` divides the full downstream height $2h$; the inlet therefore has eight cells. Keep `N_HEIGHT` even. The initial step is $0.02$ s, the maximum step is $0.05$ s, and `FINAL_TIME=12` s.

## Inspect reattachment

`solution/reattachment_history.csv` records the estimated $x_r/h$, near-wall reverse velocity, continuity and Courant number. `figures/step_evolution.png` shows the transient and `figures/step_comparison.png` shows the final flow.

Reattachment is estimated from the downstream sign change in near-wall streamwise velocity. Refine both streamwise and wall-normal spacing before interpreting $x_r/h$ as converged.
