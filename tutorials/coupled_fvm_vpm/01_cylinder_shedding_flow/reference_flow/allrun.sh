#!/bin/bash -e

./allclean.sh

python setup.py --name very_coarse --dx 0.060
python setup.py --name coarse      --dx 0.050
python setup.py --name medium      --dx 0.040
python setup.py --name fine        --dx 0.030
python setup.py --name very_fine   --dx 0.020
