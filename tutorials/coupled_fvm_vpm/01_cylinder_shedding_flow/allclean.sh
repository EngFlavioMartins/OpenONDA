#!/bin/bash
set -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples figures
rm -rf study_results/cylinder/default
rm -f ./*.log
