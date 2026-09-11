#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution constant samples figures
rm -rf __pycache__ assets/__pycache__
rm -f ./*.log
