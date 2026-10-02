"""Execute original host-only grid geometry method bodies without Taichi/JIT.

Only source AST function bodies are loaded; numerical modules are not imported.
The upload/sync stand-ins merely retain the original host-produced arrays.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def legacy_host_type(source_root=None):
    root = Path(source_root) if source_root is not None else Path(__file__).resolve().parents[2]
    path = root / "source/solvers/vpm/physics/diffusion/grid.py"
    tree = ast.parse(path.read_text())
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef)
               and item.name == "_GridDiffusionMixin")
    names = {"_prepare_body_mask_current_grid", "_prepare_body_links",
             "_body_interior_at_particles", "_body_blocked_segments",
             "_fold_slab_exterior_z", "_lattice_aligned_bounds",
             "configure_max_grid_extent", "configure_grid_lattice_anchor",
             "_explicit_diffusion_substep_count"}
    # Production now wraps the unchanged predicate body in exact-cache
    # admission. Select that explicit fresh body when present; archived source
    # retains its original direct method. No cache helpers are executed here.
    has_fresh = any(isinstance(item, ast.FunctionDef) and item.name == "_prepare_body_mask_fresh"
                    for item in cls.body)
    if has_fresh:
        names.remove("_prepare_body_mask_current_grid")
        names.add("_prepare_body_mask_fresh")
    methods = [item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name in names]
    if {item.name for item in methods} != names or any(
        item.decorator_list and not (
            item.name == "_explicit_diffusion_substep_count"
            and len(item.decorator_list) == 1
            and isinstance(item.decorator_list[0], ast.Name)
            and item.decorator_list[0].id == "staticmethod"
        ) for item in methods
    ):
        raise ValueError("Host oracle source method contract changed")
    module = ast.Module(body=methods, type_ignores=[])
    namespace = {"np": np, "math": math, "ti": SimpleNamespace(sync=lambda: None),
                 "_GRID_TRANSFER_CHUNK": 65536}
    exec(compile(module, str(path), "exec"), namespace)

    class Host:
        def __init__(self, shape, contains, blocks, slab=None):
            self._body_mask_grid = np.zeros(shape, dtype=np.int32)
            self._body_link_grid = np.zeros(shape, dtype=np.int32)
            self._body_mask_active = True
            self._body_classifier = contains
            self._body_segment_classifier = blocks
            self._body_geometry_revision = "oracle"
            self._body_mask_cache_key = None
            self._slip_slab_bounds = slab
            self._grid_shape = shape
            self._event_observer = SimpleNamespace(record=lambda *_: None)

        def _allocate_grid(self, shape):
            # Configuration-only census needs dimensions, not device storage.
            self._grid_shape = shape

        def _prepare_body_mask_current_grid(self, origin, spacing, nx, ny, nz):
            self._prepare_body_mask_fresh(np.asarray(origin, dtype=np.float32).reshape(3),
                                          spacing, nx, ny, nz)

        @staticmethod
        def _grid_transfer_buffer(*_):
            return np.empty(65536, dtype=np.int32)

        @staticmethod
        def _upload_scalar_chunk_kernel(field, buffer, start, count, ny, nz):
            field.reshape(-1)[start:start + count] = buffer[:count]

    for name in names:
        setattr(Host, name, namespace[name])
    return Host
