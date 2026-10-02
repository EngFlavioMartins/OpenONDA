#!/bin/bash
set -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples figures study_results __pycache__ assets/__pycache__
rm -f ./*.log
