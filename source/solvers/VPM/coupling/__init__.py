"""Orchestration between VPM and the VLM/panel boundary-element solvers.

The actual panel/VLM solver implementations live in ``boundary_elements``.
This package contains only the coupling orchestration: what VPM asks of a
coupled solver during one VPM step, and how the shed particles it returns are
appended to the wake.

``CouplingStepper``
    Advances the coupled panel or VLM solver once per VPM step and appends any
    shed particles it returns, mirroring the solver-facing
    ``advance`` / ``advance_coupled`` interfaces of ``boundary_elements``.
"""

from .stepper import CouplingStepper

__all__ = ["CouplingStepper"]
