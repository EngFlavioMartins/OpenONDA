#!/bin/bash -e

python setup.py --name very_coarse --dx 0.22
python setup.py --name coarse      --dx 0.20
python setup.py --name medium      --dx 0.18
python setup.py --name fine        --dx 0.16
python setup.py --name very_fine   --dx 0.14
