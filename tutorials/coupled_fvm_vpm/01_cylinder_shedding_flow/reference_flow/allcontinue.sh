#!/bin/bash
set -e

cd "$(dirname "$0")"

# Continue every case of allrun.sh from its latest accepted FVM backup.
./allrun.sh --continue
