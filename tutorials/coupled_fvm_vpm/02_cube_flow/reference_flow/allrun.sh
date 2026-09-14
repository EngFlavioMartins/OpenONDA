#!/bin/bash -e
python setup.py --name coarse --dx 0.10
python setup.py --name medium --dx 0.08
python setup.py --name fine   --dx 0.06
python setup.py --name dense  --dx 0.04
