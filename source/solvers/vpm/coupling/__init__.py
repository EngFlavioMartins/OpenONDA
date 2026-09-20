"""Orchestration between VPM and the VLM solver.

The VLM implementation lives in ``boundary_elements.vlm``. This package
contains only the coupling orchestration: what VPM asks of VLM during one VPM
step, and how the shed particles it returns are appended to the wake.

``CouplingStepper``
    Advances the coupled VLM solver once per VPM step and appends any shed
    particles it returns.
"""

from .stepper import CouplingStepper

__all__ = ["CouplingStepper"]
