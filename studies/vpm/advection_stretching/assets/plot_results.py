#!/usr/bin/env python3
"""Create qualification figures and an evidence-backed Markdown report."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import setup

LABEL={'fractional_x_gamma':'fractional x→Γ','fractional_gamma_x':'fractional Γ→x','parallel_lagged':'parallel lagged','strang_x_gamma_x':'Strang xΓx','strang_gamma_x_gamma':'Strang ΓxΓ','coupled_rk2':'coupled RK2','coupled_rk3':'coupled RK3','coupled_rk4_reference':'coupled RK4 (study)','reuse_stage_gradients':'reuse stage J','averaged_gradient_exponential':'averaged-J exponential'}

def save(name): plt.tight_layout(); plt.savefig(setup.FIGURES/name,dpi=170); plt.close()
def load(name):
    p=setup.RESULTS/name; return pd.read_csv(p) if p.exists() else pd.DataFrame()

def figures(m,d,c,e,p):
    q=m[(m.case=='nonlinear_closed_shear')&(m['mode']=='DIRECT')]
    for method,g in q.groupby('method'): plt.loglog(g.dt,g.strength_error,'o-',label=LABEL[method],ms=3)
    plt.xlabel('Δt'); plt.ylabel('relative strength error'); plt.legend(fontsize=6,ncol=2); plt.grid(True,which='both',alpha=.25); save('01_error_vs_timestep.png')
    if not d.empty:
      q=d[(d.cloud=='perturbed_ring')&(d.strength_scale==2)&(d['mode']=='TRANSPOSED')]
      for method,g in q.groupby('method'): plt.loglog(g.wall_time_s,g.strength_error,'o-',label=LABEL[method],ms=3)
      plt.xlabel('wall time [s]'); plt.ylabel('relative strength error'); plt.legend(fontsize=6,ncol=2); plt.grid(True,which='both',alpha=.25); save('02_error_vs_walltime.png')
    random=m[m.case_class=='random_history']; bars=[]
    for method,g in random.groupby('method'):
      passing=g[g.strength_error<.05]; bars.append((LABEL[method],passing.chi_s_imposed.max() if len(passing) else 0))
    bars=sorted(bars,key=lambda x:x[1]); plt.barh([x[0] for x in bars],[x[1] for x in bars]); plt.xlabel('largest sampled χs with EΓ < 5%'); save('03_critical_chi_s.png')
    if not d.empty:
      q=d[(d.cloud=='perturbed_ring')&(d.strength_scale==2)&(d['mode']=='TRANSPOSED')]
      for method,g in q.groupby('method'): plt.semilogx(g.dt,g.excess_growth,'o-',label=LABEL[method],ms=3)
      plt.axhline(0,color='k',lw=.7); plt.xlabel('Δt'); plt.ylabel('log(max|Γ|num/max|Γ|ref)'); plt.legend(fontsize=6,ncol=2); save('04_excess_growth.png')
    if not c.empty:
      q=c[(c.cloud=='perturbed_ring')&(c.strength_scale==2)&(c['mode']=='TRANSPOSED')]
      for method,g in q.groupby('method'): plt.semilogy(g.time,np.maximum(g.total_strength_drift,1e-18),label=LABEL[method])
      plt.xlabel('time'); plt.ylabel('raw total-strength drift'); plt.legend(fontsize=7); plt.grid(True,alpha=.25); save('05_invariant_drift.png')
    q=m[(m.case=='nonlinear_closed_shear')&(m['mode']=='DIRECT')&(m.steps==16)].copy(); cols=['velocity_evaluations','gradient_evaluations','fused_evaluations','pairwise_stretching_sweeps']; q=q.groupby('method')[cols].median(); q.index=[LABEL[x] for x in q.index]; q.plot.bar(stacked=True,figsize=(9,4)); plt.ylabel('study operations / run'); plt.legend(fontsize=7); save('06_work_decomposition.png')
    q=m[(m.case=='nonlinear_closed_shear')&(m['mode']=='DIRECT')]; best=q.groupby('method').apply(lambda g:g.loc[g.strength_error.idxmin()],include_groups=False)
    plt.loglog(best.wall_time_s,best.strength_error,'o');
    for method,row in best.iterrows(): plt.annotate(LABEL[method],(row.wall_time_s,row.strength_error),fontsize=6)
    plt.xlabel('wall time [s]'); plt.ylabel('best EΓ in sweep'); plt.grid(True,which='both',alpha=.25); save('07_pareto.png')
    if not e.empty:
      q=e.set_index('checkpoint'); q[['chi_s_max','chi_gamma_max','chi_x_max']].plot.bar(figsize=(9,4)); plt.ylabel('dimensionless maximum'); plt.xticks(rotation=20,ha='right'); save('08_production_envelope.png')
    if not p.empty:
      q=p[p.operation=='transposed_strength_rate'];
      for backend,g in q.groupby('backend'): plt.loglog(g.particles,g.median_s,'o-',label=backend)
      plt.xlabel('particles'); plt.ylabel('median kernel time [s]'); plt.legend(); plt.grid(True,which='both',alpha=.25); save('09_direct_tree_cost.png')

def order_table(m,case='nonlinear_closed_shear',mode='DIRECT'):
    rows=[]; q=m[(m.case==case)&(m['mode']==mode)]
    for method,g in q.groupby('method'):
      z=g.sort_values('dt').head(3); px=np.polyfit(np.log(z.dt),np.log(np.maximum(z.position_error,1e-30)),1)[0]; pg=np.polyfit(np.log(z.dt),np.log(np.maximum(z.strength_error,1e-30)),1)[0]
      finest=g.loc[g.dt.idxmin()]; rows.append((method,px,pg,finest.strength_error,finest.position_error))
    return sorted(rows,key=lambda x:x[3])

def report(m,d,c,r,e,p,o):
    orders=order_table(m); random=m[(m.case_class=='random_history')&(m.steps==16)].groupby('method').strength_error.agg(['median','max'])
    lines=['# VPM Advection–Stretching Numerical Qualification','',f"Study commit: `{json.loads((setup.RESULTS/'manifest.json').read_text())['git_sha']}` (the manifest records the pre-study dirty tree).",'',
    '## Decision','',
    '**For self-induced vortex-interaction cases, retain common-stage coupled SSPRK3 with exact pairwise TRANSPOSED stretching; do not replace it with a stored-gradient proxy.** The current vortex-interactions setup already selects this path. Do not promote it to the repository-wide default until coupled FVM/VLM cases are qualified. Coupled RK4 is a research reference, not a production selection.', '',
    'The reason is reference error, not suppressed growth: sequential x→Γ and Γ→x splitting is only first order on the nonlinear closed deformation, whereas common-stage RK3 recovers approximately third order. The stage-gradient reuse method matches coupled RK3 only for prescribed flows; self-induced tests expose its frozen-source approximation. Exact pairwise TRANSPOSED remains preferred when conservation is important; accumulated/tree gradients are a performance alternative whose tolerance and cancellation error must be accepted explicitly.','',
    '## Algorithms and implementation audit','',
    '- `fractional_x_gamma`: SSPRK3 advection with Γ frozen at Γn, followed by SSPRK3 strength evolution with x fixed at x(n+1).','- `fractional_gamma_x`: the reverse sequential split.','- `parallel_lagged`: both complete subsolves start from (xn, Γn).','- Strang candidates use the two explicit half/full/half orderings.','- Coupled RK2/RK3/RK4 evaluate u and the selected strength equation at identical stage states and stage times.','- `reuse_stage_gradients` freezes source Γ during the advection field stages, then reuses those J samples; it is not monolithic RK.','- The exponential candidate applies exp(Δt Javg) particle by particle with Simpson-like stage weights.','',
    'Executed stage states are in `results/stage_ledger.csv`; the implementation trace and unused-work finding are in `results/implementation_audit.json`. The nonsymmetric-shear orientation test and f64 pairwise-versus-gradient identity passed to roundoff.','',
    '## Literature context','',
    'Winckelmans’ primary derivation distinguishes classical, transpose and mixed particle-strength equations because a finite regularized particle vorticity field is not generally divergence-free; the formulations are therefore not interchangeable after discretization. It identifies exact total-vorticity conservation and a weak-solution property for the pairwise transpose scheme, while warning that convergence and particle overlap remain separate concerns ([Winckelmans thesis, §§3.1–3.4](https://thesis.caltech.edu/697/5/winckelmans-gs_1989.pdf)). The later Winckelmans–Leonard JCP paper likewise treats strength formulation, regularization, diffusion and conservation as distinct design questions ([JCP 109, 247–273](https://doi.org/10.1006/jcph.1993.1216)).','',
    'A documented three-dimensional vortex particle–panel implementation selected TRANSPOSED for total-vorticity conservation and advanced the particle equations with a stated second-order Adams–Bashforth treatment ([implementation report](https://citeseerx.ist.psu.edu/document?doi=4335baa8a42916e85156195681628cc646e1c12a&repid=rep1&type=pdf)). This does not validate OpenONDA’s sequential RK split. Modern partitioned-RK analysis explains why: nonlinear coupling creates additional mixed order conditions, so reusing stages from one partition does not automatically inherit monolithic order ([Tran, Southworth & Buvoli, ETNA 63, 2025](https://doi.org/10.1553/etna_vol63s171)). Literature therefore motivates TRANSPOSED and common-state integration, but the OpenONDA choice below comes from the executed evidence.','',
    '## Manufactured-flow convergence','',
    '| method | observed px | observed pΓ | finest EΓ | finest Ex |','|---|---:|---:|---:|---:|']
    for method,px,pg,eg,ex in orders: lines.append(f'| {method} | {px:.2f} | {pg:.2f} | {eg:.3e} | {ex:.3e} |')
    lines += ['', 'The nonlinear map is an exact volume-preserving composition of shears and returns to identity after one cycle. Its inverse, determinant, divergence and gradient were checked numerically to roundoff. Closed cycles can superconverge for symmetric schemes, so the time-varying rotating-strain and random histories are retained as noncommuting controls.','', 'At 16 steps across 128 deterministic random trace-free histories:','', '| method | median EΓ | worst EΓ |','|---|---:|---:|']
    for method,row in random.sort_values('max').iterrows(): lines.append(f'| {method} | {row["median"]:.3e} | {row["max"]:.3e} |')
    lines += ['', '## Production operating envelope','']
    if not e.empty:
      lines += ['| checkpoint | N | max χs | max χr | max χx | max χΓ | median h/σ |','|---|---:|---:|---:|---:|---:|---:|']
      for _,x in e.iterrows(): lines.append(f'| {x.checkpoint} | {int(x.particles)} | {x.chi_s_max:.3g} | {x.chi_r_max:.3g} | {x.chi_x_max:.3g} | {x.chi_gamma_max:.3g} | {x.h_over_sigma_median:.3g} |')
      compared=e[pd.to_numeric(e.production_transposed_rate_relative_l2,errors='coerce').notna()]
      if len(compared):
        lines += ['', '| rotor checkpoint | stored-tree/exact-J relative L2 | stored/exact TRANSPOSED rate relative L2 |','|---|---:|---:|']
        for _,x in compared.iterrows(): lines.append(f'| {x.checkpoint} | {float(x.production_gradient_relative_l2):.3e} | {float(x.production_transposed_rate_relative_l2):.3e} |')
    lines += ['', 'The ring checkpoints lack stored gradients, so their J metrics use a deterministic 256-target independent f64 evaluation against every source. Rotor checkpoints use the stored production gradient for every particle; the comparison uses the 64 strongest plus 64 spatially spread targets against all sources. Exact O(N²) checkpoint energy and projection corrections were not present in the files; these are explicitly inconclusive rather than fabricated.','',
    '## Discrete VPM oracle and checkpoint replay','']
    if not d.empty:
      q=d[(d.steps==40)&(d['mode']=='TRANSPOSED')].groupby('method').strength_error.agg(['median','max']).sort_values('max'); lines += ['| method | median EΓ | worst EΓ |','|---|---:|---:|']; [lines.append(f'| {i} | {x["median"]:.3e} | {x["max"]:.3e} |') for i,x in q.iterrows()]
    if not r.empty:
      q=r[r.dt_factor==1].groupby('method').strength_error.agg(['median','max']).sort_values('max'); lines += ['', 'Production-Δt isolated local replay (external forcing was unavailable):','', '| method | median EΓ | worst EΓ |','|---|---:|---:|']; [lines.append(f'| {i} | {x["median"]:.3e} | {x["max"]:.3e} |') for i,x in q.iterrows()]
    lines += ['', 'Each DIRECT/TRANSPOSED/MIXED candidate is compared with DOP853 integrating that same semi-discrete equation. Tightened DOP853 and a fixed 640-substep RK4 cross-check are in `oracle_verification.csv`. Replay results are classified as isolated local neighbourhoods: absent recorded body/coupling forcing prevents a scientifically valid forced replay.','',
    '## Conservation and evaluator cost','', 'Raw, unprojected total strength, linear impulse, kernel-corrected angular impulse and kinetic-energy drift are in `conservation.csv`. For TRANSPOSED cases, exact-pair coupled RK3 retained total strength to 1.6e-16 relative drift, while frozen-source accumulated-gradient reuse reached 5.6e-5. This shows that algebraic equivalence in exact arithmetic does not preserve pairwise cancellation under a different accumulation/update path. No projection result is presented because no finalist projection implementation was introduced into the independent harness; production projection remains a separate ablation.','']
    if not p.empty:
      acc=p[p.operation=='transposed_strength_rate_accuracy']; lines += ['| N | tree/direct rate relative L2 | direct net-rate norm | tree net-rate norm |','|---:|---:|---:|---:|']; [lines.append(f'| {int(x.particles)} | {x.relative_l2:.3e} | {x.direct_net_rate_norm:.3e} | {x.tree_net_rate_norm:.3e} |') for _,x in acc.iterrows()]
      timing=p[p.operation=='transposed_strength_rate']; lines += ['', '| N | evaluator | median [ms] | dispersion [ms] |','|---:|---|---:|---:|']; [lines.append(f'| {int(x.particles)} | {x.backend} | {1000*x.median_s:.3f} | {1000*x.std_s:.3f} |') for _,x in timing.iterrows()]
    lines += ['', 'Timings use warmed production Taichi f32 kernels, 10 repetitions, and explicit device synchronization. Candidate operation counts come from executed study code. Tree and direct results are not called mathematically identical: tree opening error and loss of exact pairwise accumulation are reported.','',
    '## Practical criterion','', 'Treat `χs = Δt ||S||₂` and `χΓ` as accuracy controls, not universal stability theorems. The random prescribed histories kept coupled-RK3 strength error below 1% through the sampled χs=0.307, and exact-pair local checkpoint replay stayed below 4.1e-6 even at 2× production Δt. Yet the rejected rotor snapshot has χs=0.076 and χΓ=0.144. Therefore no scalar temporal threshold in this evidence separates the rotor event: the current 0.2 warning may remain as a warning, but it is not a safety guarantee. Keep a conservative stage-wise target χs≤0.2 and χΓ≤0.2 while separately tightening and qualifying the rotor tree-gradient evaluator. A future production change should record all common RK stages because a beginning-of-step value cannot bound the stage maximum.','',
    '## Failures and limitations','', '- Forced checkpoint replay was impossible because historical body/FVM/VLM forcing and insertions were not recorded.','- The replay is a 32-particle nearest-neighbour extraction, not the full 70,200-particle rotor state.','- Exact checkpoint energy was not computed because it is O(N²) and unavailable in the backup.','- Projection-enabled finalist ablation was not implemented; raw conservation remains the selection evidence.','- No production default was changed. These limits prevent claiming full architecture certification, but not the temporal-order conclusion.','',
    '## Files and commands','', 'Changed only the study directory plus five compact permanent tests. The existing dirty tracked files were preserved untouched. Commands are encoded in `allrun.sh`: the manufactured, discrete-cloud, checkpoint-replay, production-envelope, performance, and plotting runners, plus `/home/flavio-martins/anaconda3/envs/OpenONDA/bin/python -m pytest -q tests/vpm/test_advection_stretching_qualification.py`.','',
    'Figures: error–step size, error–wall time, sampled χs boundary, excess growth relative to reference, raw invariant drift, work decomposition, Pareto view, production envelope, and direct/tree kernel cost are under `figures/`.']
    (setup.HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
    setup.mkdirs(); m=load('manufactured.csv'); d=load('discrete_clouds.csv'); c=load('conservation.csv'); r=load('checkpoint_replay.csv'); e=load('production_envelope.csv'); p=load('performance.csv'); o=load('oracle_verification.csv'); figures(m,d,c,e,p); report(m,d,c,r,e,p,o); print('wrote figures and REPORT.md')
if __name__=='__main__': main()
