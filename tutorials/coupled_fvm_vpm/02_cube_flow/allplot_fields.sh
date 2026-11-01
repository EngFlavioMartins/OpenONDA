#!/bin/bash
set -e

cd "$(dirname "$0")"

# Compare mid-plane velocity and vorticity fields.
python -m assets.plot_fields
