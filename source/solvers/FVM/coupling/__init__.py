"""FVM↔VPM coupling support.

Exposes :class:`OFWInterfaceMixin`, the duck-typed contract the OpenONDA coupler
(``source/coupler/``) calls on its Eulerian backend (historically the OFW
OpenFOAM wrapper).  Mixing it into the FVM ``Solver`` makes the native solver a
drop-in alternative backend with no coupler changes.
"""

from .ofw_interface import OFWInterfaceMixin

__all__ = ["OFWInterfaceMixin"]
