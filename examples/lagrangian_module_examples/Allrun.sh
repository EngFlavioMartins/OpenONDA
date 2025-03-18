#!/bin/bash

echo "Running: Lamb_Oseen_Vortex.py"
python Lamb_Oseen_Vortex.py

echo "Running: Vortex_Ring_potential.py"
python Vortex_Ring_potential.py

echo "Running: Vortex_Ring_pseudo2D.py"
python Vortex_Ring_pseudo2D.py

echo "Running: Vortex_Ring_DNS.py"
python Vortex_Ring_DNS.py

echo "Running: Vortex_Ring_LES.py"
python Vortex_Ring_LES.py

echo "Running: Rings_Collision_LES.py"
python Rings_Collision_LES.py

echo "Running: Rings_LeapFrogging_LES.py"
python Rings_LeapFrogging_LES.py
