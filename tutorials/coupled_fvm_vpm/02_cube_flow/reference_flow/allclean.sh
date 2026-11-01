#!/bin/bash

cd "$(dirname "$0")"

# Remove solver output and figures; keep the meshes and case inputs.
rm -rf solution samples figures
