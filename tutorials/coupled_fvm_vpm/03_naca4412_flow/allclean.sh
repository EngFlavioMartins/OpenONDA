#!/bin/bash -e

cd -- "$(dirname -- "$0")"
rm -rf solution constant samples figures .matplotlib __pycache__ assets/__pycache__
rm -f ./*.log
