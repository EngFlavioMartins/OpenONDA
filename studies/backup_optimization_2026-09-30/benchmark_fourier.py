"""Same synthetic long-wake diagnostic, old/new code in separate processes."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
import numpy as np

root = Path(__file__).resolve().parent
mode = sys.argv[1]
path = (Path('/tmp/openonda-delta-profile/frozen') if mode == 'before' else root/'frozen') / 'source/solvers/vpm/numerics/fourier_integrals.py'
spec = importlib.util.spec_from_file_location('fourier_bench', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
fft = module.fft.rfftn
def bounded_fft(*args, **kwargs):
    kwargs['workers'] = 2
    return fft(*args, **kwargs)
module.fft.rfftn = bounded_fft
rng = np.random.default_rng(4417)
count = 6000
position = rng.uniform([-7,-2.7,-2.3], [7.5,2.8,2.3], (count,3))
strength = rng.normal(size=(count,3)) * 0.003
radii = rng.uniform(0.12,0.28,count)
volume = np.full(count,0.001)
viscosity = np.full(count,0.001)
grid = module.CartesianGrid(np.array([-7.5,-3.5,-3.1]),0.1,(160,72,64))
start = perf_counter()
result = module.gaussian_fourier_integrals(position,strength,radii,volume,viscosity,grid=grid)
report = {'mode':mode,'grid':grid.shape,'particles':count,'fft_workers':2,'seconds':perf_counter()-start,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'result':asdict(result)}
(root / ('fourier_'+mode+'.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
