import os, sys, time, json, functools, cProfile, pstats
from pathlib import Path
from dataclasses import replace, asdict
ROOT=Path('/tmp/openonda-delta-profile')
sys.path.insert(0,str(ROOT/'frozen'))
import delta_setup as setup
import taichi as ti
from threadpoolctl import threadpool_info
run=ROOT/os.environ.get('PROFILE_LABEL','cpu-default');run.mkdir(exist_ok=True)
setup.TUTORIAL_DIR=run
(run/'assets').mkdir(exist_ok=True)
case=setup.build_case()
case=replace(case,numerics=replace(case.numerics,diagnostics=setup.vpm.DiagnosticsConfig(detailed_timing=True)))
t0=time.perf_counter();solver=setup.vpm.VPMSolver(case)
checkpoint=Path('/home/flavio-martins/Projects/OpenONDA/tutorials/vpm/05_delta_wing/solution/vpm/vpm_002780.h5')
solver.load_backup(checkpoint)
print('PROFILE_READY',json.dumps({'init_load':time.perf_counter()-t0,'threads':ti.lang.impl.current_cfg().cpu_max_num_threads,'threadpools':threadpool_info(),'source':setup.vpm.VPMSolver.__module__,'count':len(solver.particles)}),flush=True)
timings={};events=[];phase=''
def wrap(obj,name,label=None):
 old=getattr(obj,name);label=label or name
 @functools.wraps(old)
 def timed(*a,**kw):
  start=time.perf_counter()
  try:return old(*a,**kw)
  finally:
   dt=time.perf_counter()-start
   timings.setdefault(label,[]).append(dt)
   if label=='fmm.evaluate_stage':
    events.append({'phase':phase,'seconds':dt,'passes':dict(solver.induction.workspace.last_phase_seconds),'diagnostics':asdict(solver.induction.diagnostics)})
 setattr(obj,name,timed)
for obj,names in [
 (solver,['advance','_refresh_accepted_step_health','_refresh_backup_particle_fields','_update_all_flow_integrals','save_backup']),
 (solver.vlm_solver,['solve_stage_boundary','add_stage_rates','advance_coupled','_near_wake_stage_influence']),
 (solver.physics,['compute_target_velocity','compute_vorticities','compute_target_vorticity']),
 (solver.output_manager,['dispatch'])]:
 for name in names:wrap(obj,name)
wrap(solver.induction,'evaluate_stage','fmm.evaluate_stage')
# Workspace can grow on first call; enable pass timing after growth too.
old_ensure=solver.induction._ensure_workspace
def ensure(n):
 old_ensure(n);solver.induction.workspace.profile_passes=True
solver.induction._ensure_workspace=ensure
report={'threads':ti.lang.impl.current_cfg().cpu_max_num_threads,'threadpools':threadpool_info(),'phases':[]}
def measure(label,fn):
 global phase
 phase=label;timings.clear();events.clear();pr=cProfile.Profile();start=time.perf_counter();cpu=time.process_time();pr.enable()
 fn();ti.sync();pr.disable()
 item={'phase':label,'wall':time.perf_counter()-start,'cpu':time.process_time()-cpu,'step':solver.step,'count':len(solver.particles),'timings':{k:{'calls':len(v),'seconds':sum(v),'individual':list(v)} for k,v in timings.items()},'fmm':list(events)}
 report['phases'].append(item);(run/'results.json').write_text(json.dumps(report,indent=2));pr.dump_stats(str(run/(label+'.prof')))
 print('PROFILE_PHASE',json.dumps(item),flush=True)
 with (run/(label+'.txt')).open('w') as out:pstats.Stats(pr,stream=out).strip_dirs().sort_stats('cumulative').print_stats(65)
measure('warmup_advance',solver.advance)
measure('steady_advance',solver.advance)
measure('backup',solver.save_backup)
measure('flow_integrals',solver._update_all_flow_integrals)
print('PROFILE_DONE',str(run/'results.json'),flush=True)
