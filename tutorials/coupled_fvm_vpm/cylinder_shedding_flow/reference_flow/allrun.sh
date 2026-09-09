#!/bin/bash -e

python setup.py --name coarse --dx 0.125
python setup.py --name medium --dx 0.0625
python setup.py --name fine --dx 0.03125
python setup.py --name very_fine --dx 0.01563
