#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution samples figures constant
rm -rf __pycache__ assets/__pycache__
rm -f ./*.log
