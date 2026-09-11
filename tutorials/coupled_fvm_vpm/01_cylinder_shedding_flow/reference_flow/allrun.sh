#!/bin/bash -e

./allclean.sh

python setup.py --name coarse    --dx 0.12500
python setup.py --name medium    --dx 0.06250
python setup.py --name fine      --dx 0.03125
python setup.py --name very_fine --dx 0.02000
