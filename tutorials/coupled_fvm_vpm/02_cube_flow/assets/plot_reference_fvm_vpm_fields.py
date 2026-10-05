#!/usr/bin/env python3
"""Compare Reference FVM and VPM velocity fields on the saved z=0 slice."""

from __future__ import annotations


from .plot_coupled_fvm_vpm_fields import main


if __name__ == "__main__":
    main("reference_fvm_vpm_fields")
