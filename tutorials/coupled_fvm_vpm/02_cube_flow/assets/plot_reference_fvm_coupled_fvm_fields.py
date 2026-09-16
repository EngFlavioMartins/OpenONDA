#!/usr/bin/env python3
"""Compare Reference FVM and Coupled FVM fields on the saved z=0 slice."""

from __future__ import annotations

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"

from .plot_coupled_fvm_vpm_fields import main


if __name__ == "__main__":
    main("reference_fvm_coupled_fvm_fields")
