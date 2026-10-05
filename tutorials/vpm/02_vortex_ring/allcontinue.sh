#!/bin/bash
set -e

cd "$(dirname "$0")"

python -m openonda.tutorial_runner . setup --variant dns_direct
python -m openonda.tutorial_runner . setup --variant dns_transposed
python -m openonda.tutorial_runner . setup --variant dns_mixed
python -m openonda.tutorial_runner . setup --variant les_transposed
