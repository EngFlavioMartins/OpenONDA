#!/bin/bash -e
cd -- "$(dirname -- "$0")"
# Cube reference: r=1.5 spatial refinement, then dt/2 on the identical fine mesh.
python setup.py --campaign --lean --name grid_h010125 --dx .10125 --output-root campaigns/geometric_r15_30s_fine --cores 2 --end-time 120 --max-dt .005 --courant .5
python setup.py --campaign --lean --name grid_h00675 --dx .0675 --output-root campaigns/geometric_r15_30s_fine --cores 2 --end-time 120 --max-dt .005 --courant .5
python setup.py --campaign --lean --name grid_h0045 --dx .045 --output-root campaigns/geometric_r15_30s_fine --cores 2 --end-time 30 --max-dt .005 --courant .5 --output-interval .25 --backup-interval .25
python setup.py --campaign --lean --name grid_h003 --dx .03 --output-root campaigns/geometric_r15_30s_fine --cores 2 --end-time 120 --max-dt .005 --courant .5
python setup.py --campaign --lean --name time_h0045_dt_half --dx .045 --output-root campaigns/geometric_r15_30s_fine/temporal --cores 2 --end-time 120 --max-dt .0025 --courant .5 --mesh campaigns/geometric_r15_30s_fine/solution/grid_h0045/fvm/mesh.npz
