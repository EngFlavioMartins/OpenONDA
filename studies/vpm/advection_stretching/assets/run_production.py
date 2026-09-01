#!/usr/bin/env python3
"""Measure checkpoint envelopes and production direct/tree GPU kernels."""
from __future__ import annotations
import contextlib, csv, io, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'source'))
import h5py
import numpy as np
from scipy.spatial import cKDTree
import setup
from assets.core import contract, target_fields

def winckelmans_target_gradient(target,x,gamma,sigma):
    """Exact f64 direct gradient for the rotor's Winckelmans kernel."""
    d=target[:,None,:]-x[None,:,:]; d2=np.einsum('mnj,mnj->mn',d,d); dist=np.sqrt(d2); mask=dist>1e-14; safe=np.where(mask,dist,1.); rho=safe/sigma[None,:]; base=rho*rho+1
    qv=rho**3*(rho*rho+2.5)/(base*base*np.sqrt(base))/(4*np.pi)
    zetav=7.5/(base**3*np.sqrt(base))/(4*np.pi)/sigma[None,:]**3
    a=np.where(mask,qv/safe**3,0.); b=np.where(mask,3*qv/safe**5-zetav/np.where(mask,d2,1.),0.)
    skew=np.zeros((len(gamma),3,3)); skew[:,0,1]=-gamma[:,2]; skew[:,0,2]=gamma[:,1]; skew[:,1,0]=gamma[:,2]; skew[:,1,2]=-gamma[:,0]; skew[:,2,0]=-gamma[:,1]; skew[:,2,1]=gamma[:,0]
    return np.einsum('mn,nab->mab',a,skew)+np.einsum('mn,mna,mnb->mab',b,np.cross(d,gamma[None,:,:]),d)

def write_csv(path,rows):
    rows=list(rows); keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n'); w.writeheader(); w.writerows(rows)

def summary(values,prefix):
    a=np.asarray(values,float); return {prefix+'_median':np.median(a),prefix+'_p95':np.percentile(a,95),prefix+'_p99':np.percentile(a,99),prefix+'_max':np.max(a)}

def envelope():
    rows=[]
    for name,path,dt in setup.CHECKPOINTS:
      if not path.exists(): rows.append(dict(checkpoint=name,status='missing',path=str(path))); continue
      with h5py.File(path) as f:
        p=f['particles']; x=np.asarray(p['position'],float); gamma=np.asarray(p['vortex_strength'],float); sigma=np.asarray(p['core_radius'],float); u=np.asarray(p['velocity'],float)
        has_j='velocity_gradient' in p
        if has_j: j=np.asarray(p['velocity_gradient'],float).reshape(-1,3,3)
      h=cKDTree(x).query(x,k=2)[0][:,1]; chi_x=dt*np.linalg.norm(u,axis=1)/np.minimum(h,sigma); hs=h/sigma; mag=np.linalg.norm(gamma,axis=1)
      # Existing vortex-ring checkpoints do not store J. Evaluate a deterministic
      # 256-target sample against all 14,080 sources using the independent f64 kernel.
      if not has_j:
        selected=np.linspace(0,len(x)-1,min(256,len(x)),dtype=int); chunks=[]
        for start in range(0,len(selected),32): chunks.append(target_fields(x[selected[start:start+32]],x,gamma,sigma)[2])
        j=np.concatenate(chunks); eval_gamma=gamma[selected]; evaluator='independent_f64_gradient_on_256_targets'
        production_gradient_relative_l2=''; production_transposed_rate_relative_l2=''
      else:
        eval_gamma=gamma; evaluator='stored_production_gradient_all_particles'
        strongest=np.argpartition(np.linalg.norm(gamma,axis=1),-64)[-64:]; spread=np.linspace(0,len(x)-1,64,dtype=int); selected=np.unique(np.r_[strongest,spread]); exact=[]
        for start in range(0,len(selected),8): exact.append(winckelmans_target_gradient(x[selected[start:start+8]],x,gamma,sigma))
        exact=np.concatenate(exact); stored=j[selected]
        production_gradient_relative_l2=np.linalg.norm(stored-exact)/max(np.linalg.norm(exact),1e-30)
        production_transposed_rate_relative_l2=np.linalg.norm(contract(stored,gamma[selected],'TRANSPOSED')-contract(exact,gamma[selected],'TRANSPOSED'))/max(np.linalg.norm(contract(exact,gamma[selected],'TRANSPOSED')),1e-30)
      s=.5*(j+j.transpose(0,2,1)); w=.5*(j-j.transpose(0,2,1)); chi_s=dt*np.linalg.norm(s,ord=2,axis=(1,2)); chi_r=dt*np.linalg.norm(w,ord=2,axis=(1,2)); chi_g=dt*np.linalg.norm(contract(j,eval_gamma,'TRANSPOSED'),axis=1)/np.maximum(np.linalg.norm(eval_gamma,axis=1),1e-30)
      row=dict(checkpoint=name,status='measured',path=str(path.relative_to(setup.ROOT)),particles=len(x),dt=dt,gradient_evaluator=evaluator,production_gradient_relative_l2=production_gradient_relative_l2,production_transposed_rate_relative_l2=production_transposed_rate_relative_l2,total_strength_x=gamma.sum(0)[0],total_strength_y=gamma.sum(0)[1],total_strength_z=gamma.sum(0)[2],linear_impulse_norm=np.linalg.norm(.5*np.cross(x,gamma).sum(0)),angular_impulse_norm=np.linalg.norm(np.cross(x,np.cross(x,gamma)).sum(0)/3-(sigma[:,None]**2*gamma).sum(0)/3),energy_status='not_computed: exact checkpoint energy is O(N^2)',projection_correction_status='not_recorded_in_checkpoint',**summary(chi_s,'chi_s'),**summary(chi_r,'chi_r'),**summary(chi_x,'chi_x'),**summary(chi_g,'chi_gamma'),**summary(hs,'h_over_sigma'),**summary(mag,'strength'))
      rows.append(row); print(name,'chi_s max',row['chi_s_max'],'chi_gamma max',row['chi_gamma_max'])
    write_csv(setup.RESULTS/'production_envelope.csv',rows)

def performance():
    import taichi as ti
    from openonda.vpm import Backup,DirectInduction,Numerics,StabilizationConfig,TreecodeInduction,TurbulenceConfig,ViscousConfig,VPMCase,VPMSolver
    rows=[]; rates={}
    def make(n,backend):
      rng=np.random.default_rng(setup.SEED+n); position=rng.uniform(-1,1,(n,3)).astype('f4'); strength=(.02*rng.normal(size=(n,3))).astype('f4'); strength-=strength.mean(0); radius=np.full(n,.18,'f4'); volume=np.full(n,.12**3,'f4')
      case=VPMCase(directory=setup.RESULTS/'performance_work'/f'{backend}_{n}',backup=Backup(),numerics=Numerics(compute_device='VULKAN',precision='f32',induction=DirectInduction() if backend=='direct' else TreecodeInduction(theta=.2),viscous=ViscousConfig.inviscid(particle_spacing=.12),turbulence=TurbulenceConfig.inviscid(),stabilization=StabilizationConfig.disabled(),max_n_particles=n+16,max_evaluation_points=n+16,verbose=False))
      with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()): solver=VPMSolver(case)
      solver.add_vortex_particles(position=position,velocity=np.zeros_like(position),vortex_strength=strength,core_radius=radius,particle_volume=volume,kinematic_viscosity=np.zeros(n,'f4')); return solver
    def timed(op):
      op(); ti.sync(); vals=[]
      for _ in range(10): start=time.perf_counter(); op(); ti.sync(); vals.append(time.perf_counter()-start)
      return np.median(vals),np.std(vals)
    for n in (256,1024,4096):
      for backend in ('direct','tree'):
       solver=make(n,backend); p=solver.particles; physics=solver.physics; physics._resize_temp_fields(n); physics._stretching._use_treecode=backend=='tree'; physics._stretching._treecode_theta=.2
       def rate(): physics._stretching._rate(p.position,p.vortex_strength,p.core_radius,physics.dstr_dt_temp,1,n)
       med,std=timed(rate); rate(); ti.sync(); rates[n,backend]=physics.dstr_dt_temp.to_numpy()[:n].astype(float)
       rows.append(dict(particles=n,backend=backend,operation='transposed_strength_rate',median_s=med,std_s=std,repeats=10,precision='f32',theta=.2 if backend=='tree' else '',pairwise_stretching_sweeps=backend=='direct',gradient_evaluations=backend=='tree',tree_builds=backend=='tree',tree_traversals=backend=='tree',synchronisations=1,kernel_launches_minimum=1))
       def fused(): physics.compute_velocity_and_gradient_hierarchical(p,theta=.2) if backend=='tree' else physics.compute_velocity_and_gradient(p)
       med,std=timed(fused); rows.append(dict(particles=n,backend=backend,operation='velocity_and_gradient',median_s=med,std_s=std,repeats=10,precision='f32',theta=.2 if backend=='tree' else '',fused_evaluations=1,tree_builds=backend=='tree',tree_traversals=backend=='tree',synchronisations=1,kernel_launches_minimum=1)); solver.close()
      direct=rates[n,'direct']; tree=rates[n,'tree']; rows.append(dict(particles=n,backend='tree_vs_direct',operation='transposed_strength_rate_accuracy',relative_l2=np.linalg.norm(tree-direct)/max(np.linalg.norm(direct),1e-30),direct_net_rate_norm=np.linalg.norm(direct.sum(0)),tree_net_rate_norm=np.linalg.norm(tree.sum(0)),precision='f32',theta=.2))
    write_csv(setup.RESULTS/'performance.csv',rows); print(f'wrote {len(rows)} performance records')

if __name__=='__main__': envelope(); performance()
