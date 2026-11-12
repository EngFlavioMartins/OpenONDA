#!/bin/bash
set -e

cd "$(dirname "$0")"

# Remove previous output and simulate every case from t = 0.
./allclean.sh

# Same-mesh references of the coupled study (allrun.sh one level up). Each coupled run is compared
# with the FVM-only run on its cells and step; the step is refined with the cells,
# dt = 0.008 s h / 0.04 m.
#   h = 0.04 m: the baseline and groups T, M, I, the box margins 1.6 and 2.0 D of group D, and the
#               particle spacing of group S;
#   h = 0.08, 0.06 m: the cell sizes of group S;
#   h = 0.03 m: the FVM discretization error at h = 0.04 m.
python setup.py --cell-size 0.08 --name h0.08_dt0.016
python setup.py --cell-size 0.06 --name h0.06_dt0.012
python setup.py --cell-size 0.04 --name h0.04_dt0.008
python setup.py --cell-size 0.03 --name h0.03_dt0.006

# Group D: the coupled boxes of margin 3.2 and 4.0 D reach beyond the fine cells above, so their
# references fill the larger box with cells of 0.04 m. The coupled runs end at 55 s.
python setup.py --cell-size 0.04 --fine-box-margin 3.2 --end-time 55 --name h0.04_dt0.008_fb3.2
python setup.py --cell-size 0.04 --fine-box-margin 4.0 --end-time 55 --name h0.04_dt0.008_fb4.0
