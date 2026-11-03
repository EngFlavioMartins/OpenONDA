#!/bin/bash

cd "$(dirname "$0")"

# Remove solver output and figures; keep the mesh, the case inputs and the sensitivity cases in
# solution/sensitivity_analysis/.
rm -rf samples figures
find solution -mindepth 1 -maxdepth 1 ! -name sensitivity_analysis -exec rm -rf {} + 2> /dev/null
exit 0
