#!/usr/bin/env bash
# Clean the tutorial-local Lamb--Oseen output directories.
# Only tutorial-local artifacts are removed; repository-root directories named
# solution, samples, or figures are never touched.

cd "$(dirname "$0")"
rm -rf solution samples figures
rm -f ./*.log ./run_manifest.json
