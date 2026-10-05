"""Optional finite Gaussian slab-image mesh components.

Importing this package never imports CuPy or selects a GPU. The components
are selected explicitly through SlipSlabInduction and confer no automatic
image-tail or particle state validation.
"""


def __getattr__(name):
    if name == "GaussianImageFields":
        from .fields import GaussianImageFields

        return GaussianImageFields
    if name == "GaussianMeshParameters":
        from .parameters import GaussianMeshParameters

        return GaussianMeshParameters
    if name == "GaussianSlabSettings":
        from .session import GaussianSlabSettings

        return GaussianSlabSettings
    raise AttributeError(name)


__all__ = ["GaussianImageFields", "GaussianMeshParameters", "GaussianSlabSettings"]
