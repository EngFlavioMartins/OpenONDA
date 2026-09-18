#!/bin/bash -e
cd -- "$(dirname -- "$0")"

python setup.py --name very_coarse --dx 0.080
python setup.py --name coarse      --dx 0.070
python setup.py --name medium      --dx 0.060
python setup.py --name fine        --dx 0.050
python setup.py --name very_fine   --dx 0.040
