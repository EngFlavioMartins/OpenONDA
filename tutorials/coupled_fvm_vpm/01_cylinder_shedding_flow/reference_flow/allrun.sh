#!/bin/bash -e
cd -- "$(dirname -- "$0")"

# In-plane ratio sqrt(2); four span layers isolate XY refinement.
python setup.py --name xy_coarse --dx 0.08 --output-root campaigns/geometric_xy_100s/spatial --cores 2 --end-time 100 --span 1.0 --span-layers 4 --maximum-time-step 0.004 --lean
python setup.py --name xy_medium --dx 0.05656854249492380 --output-root campaigns/geometric_xy_100s/spatial --cores 2 --end-time 100 --span 1.0 --span-layers 4 --maximum-time-step 0.004 --lean
python setup.py --name xy_fine --dx 0.04 --output-root campaigns/geometric_xy_100s/spatial --cores 2 --end-time 100 --span 1.0 --span-layers 4 --maximum-time-step 0.004 --output-interval 0.25 --backup-interval 0.25 --lean
python setup.py --name xy_finer --dx 0.02828427124746190 --output-root campaigns/geometric_xy_100s/spatial --cores 2 --end-time 100 --span 1.0 --span-layers 4 --maximum-time-step 0.004 --lean
python setup.py --name z_eight --dx 0.04 --output-root campaigns/geometric_xy_100s/controls --cores 2 --end-time 100 --span 1.0 --span-layers 8 --maximum-time-step 0.004 --lean
python setup.py --name span_two --dx 0.04 --output-root campaigns/geometric_xy_100s/controls --cores 2 --end-time 100 --span 2.0 --span-layers 8 --maximum-time-step 0.004 --lean
python setup.py --name dt_half --dx 0.04 --output-root campaigns/geometric_xy_100s/controls --cores 2 --end-time 100 --span 1.0 --span-layers 4 --maximum-time-step 0.002 --lean
