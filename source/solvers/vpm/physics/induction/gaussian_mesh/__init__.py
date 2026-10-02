"""Optional finite Gaussian slab-image mesh components.

Importing this package never imports CuPy or selects a GPU. The components
are selected explicitly through SlipSlabInduction and confer no automatic
image-tail or health admission.
"""


def __getattr__(name):
    if name == "GaussianImageFields":
        from .fields import GaussianImageFields
        return GaussianImageFields
    if name == "GaussianMeshParameters":
        from .policy import GaussianMeshParameters
        return GaussianMeshParameters
    if name == "GaussianSlabPolicy":
        from .session import GaussianSlabPolicy
        return GaussianSlabPolicy
    raise AttributeError(name)


__all__ = ["GaussianImageFields", "GaussianMeshParameters", "GaussianSlabPolicy"]
