#!/bin/bash
set -e
cd "$(dirname "$0")/.."  # the case directory

# Each run solves a copy of this case in sensitivity/<run>/ up to t = 55 s, where
# the lift-amplitude error has reached its developed level, and resumes its latest
# backup when launched again. The arguments are the FVM box margin [D], the FVM
# steps per exchange and the transfer-region inset [cells].
run() {
    directory="sensitivity/box$1_sub$2_inset$3"
    mkdir -p "$directory"
    cp -r --update=none setup.py assets "$directory"
    (cd "$directory" && python setup.py --box-margin "$1" --fvm-substeps "$2" --transfer-inset "$3" --end-time 55)
}

# Baseline
run 2.4 5 8.75

# FVM box margin
run 1.0 5 8.75
run 1.4 5 8.75
run 1.8 5 8.75
run 2.2 5 8.75
run 2.6 5 8.75

# FVM steps per exchange
run 2.4 2 8.75
run 2.4 4 8.75
run 2.4 6 8.75
run 2.4 8 8.75

# Transfer-region inset
run 2.4 5 4
run 2.4 5 6
run 2.4 5 14
run 2.4 5 20
