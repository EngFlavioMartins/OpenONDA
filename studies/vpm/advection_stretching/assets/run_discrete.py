#!/usr/bin/env python3
"""Run independent self-induced VPM oracle, conservation, and checkpoint replays."""
from __future__ import annotations
import csv, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
from scipy.spatial import cKDTree
import setup
from assets.core import (ParticleEvaluator, State, advance, errors, integrate, invariants,
                         reference_particle, target_fields, velocity, pair_rate)

def write_csv(path,rows):
    rows=list(rows); keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n'); w.writeheader(); w.writerows(rows)

def ring(n,offset=(0,0,0),tilt=0,perturb=0):
    a=2*np.pi*np.arange(n)/n; radius=1+perturb*np.cos(5*a)
    x=np.column_stack((perturb*np.sin(3*a),radius*np.cos(a),radius*np.sin(a)))
    tangent=np.column_stack((3*perturb*np.cos(3*a),-radius*np.sin(a)-5*perturb*np.sin(5*a)*np.cos(a),radius*np.cos(a)-5*perturb*np.sin(5*a)*np.sin(a)))
    tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    r=np.array(((np.cos(tilt),-np.sin(tilt),0),(np.sin(tilt),np.cos(tilt),0),(0,0,1)))
    return State(x@r.T+np.asarray(offset),2*np.pi/n*tangent@r.T),np.full(n,setup.SIGMA)

def clouds():
    single,ss=ring(32); a,sa=ring(20,(-.35,0,0),.22); b,sb=ring(20,(.35,0,0),-.18)
    double=State(np.vstack((a.x,b.x)),np.vstack((a.gamma,b.gamma))); pert,sp=ring(36,perturb=.08)
    return (('single_ring',single,ss),('offset_inclined_rings',double,np.r_[sa,sb]),('perturbed_ring',pert,sp))

def invariant_scales(state):
    cross=np.cross(state.x,state.gamma)
    angular=np.cross(state.x,cross)
    return (np.linalg.norm(state.gamma,axis=1).sum(), .5*np.linalg.norm(cross,axis=1).sum(), np.linalg.norm(angular,axis=1).sum()/3)

def drift(now,base,scales):
    absolute=[np.linalg.norm(now[i]-base[i]) for i in range(3)]
    return dict(total_strength_drift=absolute[0]/max(scales[0],1e-30),total_strength_drift_abs=absolute[0],linear_impulse_drift=absolute[1]/max(scales[1],1e-30),linear_impulse_drift_abs=absolute[1],angular_impulse_drift=absolute[2]/max(scales[2],1e-30),angular_impulse_drift_abs=absolute[2],energy_drift=abs(now[3]-base[3])/max(abs(base[3]),1e-30),energy_drift_abs=abs(now[3]-base[3]))

def fixed_rk4(state,sigma,horizon,steps,mode):
    out=state.copy(); dt=horizon/steps
    def rhs(s): return State(velocity(s.x,s.gamma,sigma),pair_rate(s.x,s.gamma,sigma,mode))
    for _ in range(steps):
        a=rhs(out); b=rhs(State(out.x+.5*dt*a.x,out.gamma+.5*dt*a.gamma)); c=rhs(State(out.x+.5*dt*b.x,out.gamma+.5*dt*b.gamma)); d=rhs(State(out.x+dt*c.x,out.gamma+dt*c.gamma)); out=State(out.x+dt*(a.x+2*b.x+2*c.x+d.x)/6,out.gamma+dt*(a.gamma+2*b.gamma+2*c.gamma+d.gamma)/6)
    return out

def discrete():
    rng=np.random.default_rng(setup.SEED+19); rows=[]; conservation=[]; verify=[]
    for name,base,sigma in clouds():
      probes=rng.uniform(base.x.min(0)-.35,base.x.max(0)+.35,size=(24,3))
      for scale in (.75,2.):
       initial=State(base.x.copy(),scale*base.gamma); inv0=invariants(initial,sigma); scales=invariant_scales(initial)
       for mode in setup.MODES:
        ref=reference_particle(initial,sigma,setup.DISCRETE_HORIZON,mode); tight=reference_particle(initial,sigma,setup.DISCRETE_HORIZON,mode,2e-13,2e-15)
        verify.append(dict(cloud=name,strength_scale=scale,mode=mode,position_reference_change=errors(ref,tight)['position_error'],strength_reference_change=errors(ref,tight)['strength_error'],comparison='DOP853_tolerance'))
        ru,rw,rj=target_fields(probes,ref.x,ref.gamma,sigma)
        for method in setup.METHODS:
         for steps in setup.DSTEPS:
          e=ParticleEvaluator(sigma); start=time.perf_counter(); out=integrate(method,e,initial,setup.DISCRETE_HORIZON,steps,mode)
          u,w,j=target_fields(probes,out.x,out.gamma,sigma); row=dict(cloud=name,strength_scale=scale,mode=mode,method=method,steps=steps,dt=setup.DISCRETE_HORIZON/steps,horizon=setup.DISCRETE_HORIZON,finite=np.isfinite(out.x).all() and np.isfinite(out.gamma).all(),velocity_probe_error=np.linalg.norm(u-ru)/np.linalg.norm(ru),vorticity_probe_error=np.linalg.norm(w-rw)/np.linalg.norm(rw),gradient_probe_error=np.linalg.norm(j-rj)/np.linalg.norm(rj),wall_time_s=time.perf_counter()-start,**errors(out,ref),**e.counts.__dict__,**drift(invariants(out,sigma),inv0,scales)); rows.append(row)
        for method in ('fractional_x_gamma','coupled_rk3','coupled_rk4_reference','reuse_stage_gradients'):
         e=ParticleEvaluator(sigma); out=initial.copy(); steps=20; dt=setup.DISCRETE_HORIZON/steps
         for n in range(steps+1):
          conservation.append(dict(cloud=name,strength_scale=scale,mode=mode,method=method,step=n,time=n*dt,**drift(invariants(out,sigma),inv0,scales)))
          if n<steps: out=advance(method,e,out,n*dt,dt,mode,step=n)
    # Independent fixed-step RK4 check on a representative feedback case.
    name,base,sigma=clouds()[2]; initial=State(base.x,.75*base.gamma); dop=reference_particle(initial,sigma,setup.DISCRETE_HORIZON,'TRANSPOSED'); fixed=fixed_rk4(initial,sigma,setup.DISCRETE_HORIZON,640,'TRANSPOSED')
    verify.append(dict(cloud=name,strength_scale=.75,mode='TRANSPOSED',position_reference_change=errors(dop,fixed)['position_error'],strength_reference_change=errors(dop,fixed)['strength_error'],comparison='DOP853_vs_fixed_RK4_640'))
    write_csv(setup.RESULTS/'discrete_clouds.csv',rows); write_csv(setup.RESULTS/'conservation.csv',conservation); write_csv(setup.RESULTS/'oracle_verification.csv',verify)
    print(f'wrote {len(rows)} discrete and {len(conservation)} conservation records')

def local(path,count=32):
    with h5py.File(path) as f:
        p=f['particles']; x=np.asarray(p['position'],float); g=np.asarray(p['vortex_strength'],float); s=np.asarray(p['core_radius'],float)
    anchor=int(np.argmax(np.linalg.norm(g,axis=1))); chosen=np.sort(np.atleast_1d(cKDTree(x).query(x[anchor],k=min(count,len(x)))[1])); xx=x[chosen]; xx-=xx.mean(0)
    return State(xx,g[chosen]),s[chosen],dict(source_particles=len(x),selected_particles=len(chosen),anchor=anchor,source_strength_max=np.linalg.norm(g,axis=1).max(),selected_strength_max=np.linalg.norm(g[chosen],axis=1).max(),external_forcing_available=False,replay_class='isolated_local_neighbourhood')

def replay():
    rows=[]
    for name,path,dt0 in setup.CHECKPOINTS[1:]:
      if not path.exists(): rows.append(dict(checkpoint=name,status='missing',path=str(path))); continue
      initial,sigma,meta=local(path); kernel='WINCKELMANS' if name.startswith('rotor_') else 'GAUSSIAN'
      for factor in (.5,1.,2.):
       steps=10; dt=dt0*factor; horizon=steps*dt; ref=reference_particle(initial,sigma,horizon,'TRANSPOSED',kernel=kernel)
       for method in setup.METHODS:
        e=ParticleEvaluator(sigma,kernel=kernel); start=time.perf_counter(); out=integrate(method,e,initial,horizon,steps,'TRANSPOSED')
        rows.append(dict(checkpoint=name,status='completed' if np.isfinite(out.x).all() and np.isfinite(out.gamma).all() else 'nonfinite',path=str(path.relative_to(setup.ROOT)),particle_kernel=kernel,dt_factor=factor,dt=dt,steps=steps,horizon=horizon,method=method,wall_time_s=time.perf_counter()-start,**meta,**errors(out,ref),**e.counts.__dict__))
    write_csv(setup.RESULTS/'checkpoint_replay.csv',rows); print(f'wrote {len(rows)} replay records')

if __name__=='__main__': discrete(); replay()
