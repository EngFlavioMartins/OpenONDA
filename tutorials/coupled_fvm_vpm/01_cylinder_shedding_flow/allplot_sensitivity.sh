#!/bin/bash

# Plot the completed coupling sensitivity study.
python -m assets.plot_sensitivity
python -m assets.strength_ratio_history
python -m assets.flow_scales
python -m assets.plot_error_analysis

# Error sources and one-at-a-time sensitivity of the interface (groups D, I, T, V, F).
python -m assets.interface_sensitivity
