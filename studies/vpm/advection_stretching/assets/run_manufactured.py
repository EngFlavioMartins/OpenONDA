#!/usr/bin/env python3
"""Audit stage paths and execute deterministic and random manufactured tests."""
from __future__ import annotations
import csv, hashlib, json, platform, subprocess, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy
import setup
from assets.core import (AffineFlow, AnalyticEvaluator, State, advance, contract, errors,
                         flows, integrate, pair_rate, gradient)

def write_csv(path,rows):
    rows=list(rows); path.parent.mkdir(parents=True,exist_ok=True)
    keys=[]
    for row in rows:
        for key in row:
            if key not in keys: keys.append(key)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n'); w.writeheader(); w.writerows(rows)

def manifest():
    sha=subprocess.check_output(('git','rev-parse','HEAD'),cwd=setup.ROOT,text=True).strip()
    dirty=subprocess.check_output(('git','status','--short'),cwd=setup.ROOT,text=True).splitlines()
    cpu=subprocess.run(('lscpu',),capture_output=True,text=True,check=False).stdout
    pci=subprocess.run(('lspci',),capture_output=True,text=True,check=False).stdout
    displays=[line for line in pci.splitlines() if any(word in line.lower() for word in ('vga','3d controller','display'))]
    payload=dict(git_sha=sha,dirty_status=dirty,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),machine=platform.machine(),processor=platform.processor(),cpu_model=next((line.split(':',1)[1].strip() for line in cpu.splitlines() if line.startswith('Model name:')),''),display_devices=displays,random_seed=setup.SEED,backend='NumPy f64 oracle; production Taichi VULKAN f32 benchmark (device selected by runtime)',precision='f64 oracle / f32 production benchmark',commands=['python assets/run_manufactured.py','python assets/run_discrete_clouds.py','python assets/run_checkpoint_replay.py','python assets/run_production_envelope.py','python assets/run_performance.py','python assets/plot_results.py'])
    payload['setup_hash']=hashlib.sha256((setup.HERE/'setup.py').read_bytes()).hexdigest()
    (setup.RESULTS/'manifest.json').write_text(json.dumps(payload,indent=2)+'\n')

def audit():
    x=np.zeros((1,3)); basis=np.eye(3); j=np.array(((0.,2.,0.),(0.,0.,0.),(0.,0.,0.)))
    orientation=[]
    for mode in setup.MODES:
        jj=np.broadcast_to(j,(3,3,3)); rates=contract(jj,basis,mode)
        for i,rate in enumerate(rates): orientation.append(dict(implementation='analytic_gradient_contraction',mode=mode,basis=i,rate_x=rate[0],rate_y=rate[1],rate_z=rate[2]))
    rng=np.random.default_rng(setup.SEED); xp=rng.normal(size=(12,3)); gp=rng.normal(size=(12,3)); sig=np.full(12,.35)
    equivalence=[]
    for mode in setup.MODES:
        exact=pair_rate(xp,gp,sig,mode); accumulated=contract(gradient(xp,gp,sig),gp,mode)
        equivalence.append(dict(mode=mode,relative_l2=np.linalg.norm(exact-accumulated)/np.linalg.norm(exact),max_abs=np.max(np.abs(exact-accumulated))))
    ledger=[]; state=State(*setup.cloud()); flow=flows()[-1]
    for method in setup.METHODS: advance(method,AnalyticEvaluator(flow),state,0.,.1,'TRANSPOSED',ledger,0)
    write_csv(setup.RESULTS/'stage_ledger.csv',ledger)
    write_csv(setup.RESULTS/'tensor_orientation.csv',orientation)
    audit_payload={
      'stage_execution':'RungeKutta evaluates u and dGamma/dt from common (x_i,Gamma_i) stage states and calls StageRHS once per tableau stage.',
      'induction_methods':['DirectInduction','TreecodeInduction','FMMInduction'],
      'integrators':['RK2','SSPRK3','RK4'],
      'contractions':{'DIRECT':'J Gamma','TRANSPOSED':'J^T Gamma','MIXED':'0.5(J+J^T) Gamma'},
      'direct_pairwise':'Separate exact per-target source summation; transposed form retains algebraic pair cancellation.',
      'tree_gradient':'The LBVH stage evaluator supplies the auxiliary gradient when requested; every coupled stage rebuilds from its complete temporary state.',
      'fmm_path':'FMM uses shared-kernel near interactions and singular Biot-Savart multipole velocity for well-separated cells; the canonical pairwise transpose remains exact until a mutual rate traversal is qualified.',
      'potential_unused_work':'The auxiliary gradient is not the canonical strength operator and is never substituted for the conservative pair sweep.',
      'pair_vs_accumulated_f64':equivalence,
      'source_files':['source/solvers/vpm/core/evolution.py','source/solvers/vpm/physics/engine.py','source/solvers/vpm/numerics/kernels_common.py','source/solvers/vpm/physics/induction/treecode/lbvh.py','source/solvers/vpm/physics/induction/fmm/evaluator.py']}
    (setup.RESULTS/'implementation_audit.json').write_text(json.dumps(audit_payload,indent=2)+'\n')
    if max(r['relative_l2'] for r in equivalence)>5e-13: raise AssertionError(equivalence)

def affine_metrics(x0,x,ref):
    design=np.column_stack((x0,np.ones(len(x0)))); c=np.linalg.lstsq(design,x,rcond=None)[0]; cr=np.linalg.lstsq(design,ref,rcond=None)[0]; fitted=design@c
    eig=np.maximum(np.linalg.eigvalsh(np.cov((x-x.mean(0)).T)),1e-30)
    return dict(affine_map_error=np.linalg.norm(c[:3]-cr[:3])/max(np.linalg.norm(cr[:3]),1e-30),nonaffine_residual=np.linalg.norm(x-fitted)/max(np.linalg.norm(ref),1e-30),cloud_covariance_condition=eig[-1]/eig[0])

def random_flow(seed):
    rng=np.random.default_rng(seed); mats=[]
    for _ in range(3):
        m=rng.normal(size=(3,3)); m-=np.eye(3)*np.trace(m)/3; m*=rng.uniform(.25,1.5)/np.linalg.norm(m,2); mats.append(m)
    phases=rng.uniform(0,2*np.pi,3)
    def matrix(t): return mats[0]*np.sin(2*np.pi*t+phases[0])+mats[1]*np.cos(3*np.pi*t+phases[1])+mats[2]*np.sin(5*np.pi*t+phases[2])
    return AffineFlow(f'random_{seed}',matrix)

def main():
    setup.mkdirs(); manifest(); audit(); x,g=setup.cloud(); initial=State(x,g); rows=[]
    for flow in flows():
      # Independent complex-step/trace/map checks for the nonlinear flow.
      validation={}
      if flow.name=='nonlinear_closed_shear':
        probe=x[:9]; t=.371; j=flow.gradient(probe,t); validation=dict(gradient_trace_max=float(np.max(np.abs(np.trace(j,axis1=1,axis2=2)))),inverse_error=float(np.max(np.abs(flow.inverse(flow.material(probe,t),t)-probe))),deformation_determinant_error=float(np.max(np.abs(np.linalg.det(flow.deformation(probe,t))-1))))
      for mode in setup.MODES:
        ref=flow.exact(initial,setup.HORIZON,mode)
        for method in setup.METHODS:
          for steps in setup.MSTEPS:
            e=AnalyticEvaluator(flow); start=time.perf_counter(); out=integrate(method,e,initial,setup.HORIZON,steps,mode)
            row=dict(case=flow.name,case_class='deterministic',mode=mode,method=method,steps=steps,dt=setup.HORIZON/steps,wall_time_s=time.perf_counter()-start,finite=np.isfinite(out.x).all() and np.isfinite(out.gamma).all(),**errors(out,ref),**asdict_safe(e.counts),**validation)
            row.update(affine_metrics(x,out.x,ref.x)); rows.append(row)
    # 128 deterministic trace-free, time-varying histories; DIRECT is used here
    # because every formulation already has its own full deterministic matrix.
    for index in range(128):
      flow=random_flow(setup.SEED+index); ref=flow.exact(initial,1.,'DIRECT')
      for method in setup.METHODS:
        for steps in (1,2,4,8,16):
          dt=1/steps
          sample_times=np.linspace(0,1,max(33,4*steps+1)); chi_s=dt*max(np.linalg.norm(.5*(flow.mat(t)+flow.mat(t).T),2) for t in sample_times)
          e=AnalyticEvaluator(flow); out=integrate(method,e,initial,1.,steps,'DIRECT')
          rows.append(dict(case=flow.name,case_class='random_history',seed=setup.SEED+index,mode='DIRECT',method=method,steps=steps,dt=dt,chi_s_imposed=chi_s,finite=np.isfinite(out.x).all() and np.isfinite(out.gamma).all(),**errors(out,ref),**asdict_safe(e.counts)))
    write_csv(setup.RESULTS/'manufactured.csv',rows); print(f'wrote {len(rows)} manufactured records')

def asdict_safe(counts): return {k:v for k,v in counts.__dict__.items()}
if __name__=='__main__': main()
