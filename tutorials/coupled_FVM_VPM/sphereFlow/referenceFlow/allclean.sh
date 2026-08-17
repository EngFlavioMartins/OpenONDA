#!/bin/sh
cd "$(dirname "$0")" || exit 1
rm -rf solution samples __pycache__
echo "Cleaned reference: solution/ samples/ removed."
