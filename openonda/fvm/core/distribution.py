"""Distribution of the global mesh over the ranks of a partitioned run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openonda.core.parallel import collective_phase

from ..io.backup import mesh_digest
from ..mesh import geometry
from ..mesh.partition import localize_mesh_and_geometry
from ..mesh.validation import enforce_quality_thresholds, validate_geometry, validate_topology

if TYPE_CHECKING:
    from ..config.configuration import FVMConfiguration
    from .parallel import ParallelContext

_PARTITION_TAG = 9131


def distribute_mesh(
    parallel: ParallelContext, global_mesh: dict[str, Any], configuration: FVMConfiguration
) -> tuple[dict[str, Any], dict[str, Any], Any, ParallelContext]:
    """Give every rank its partition of the mesh built on rank zero.

    Rank zero validates the global mesh, localizes one partition at a time and
    sends it to its rank; ``global_mesh`` is only read on rank zero. Returns
    the local mesh data, local geometry, the mesh quality report of the global
    mesh (on every rank) and the parallel context that carries the partition.
    """
    comm = parallel.comm
    assert comm is not None
    if any(boundary.velocity_type == "cyclic" for boundary in configuration.boundaries):
        raise NotImplementedError(
            "Partitioned cyclic patches require periodic partition adjacency, which is "
            "not yet implemented"
        )
    quality = None
    global_geo = None
    global_hash = None
    with collective_phase(comm, "partitioned mesh preparation"):
        if parallel.is_root:
            validate_topology(global_mesh)
            global_geo = geometry.compute_mesh_geometry(
                global_mesh,
                gradient_scheme=configuration.numerics.schemes.gradient_scheme,
                compute_lsq=False,
            )
            quality = validate_geometry(global_mesh, global_geo)
            enforce_quality_thresholds(quality, configuration.mesh_quality)
            global_hash = mesh_digest(global_mesh)

    # Scatter would force rank zero to retain one complete localized
    # mesh partition per rank.  Send one partition at a time instead.
    # A localization failure on rank zero is sent to the ranks still
    # waiting, so the phase below ends on every rank before solver
    # construction starts.
    local_partition_data = None
    partition_data = None
    with collective_phase(comm, "mesh partition distribution"):
        if parallel.is_root:
            assert global_geo is not None and global_hash is not None
            # Send worker partitions first and build rank zero last. Drop
            # each sent mesh partition before localizing the next partition so
            # rank zero holds at most one local mesh partition beside the global mesh.
            rank_order = [*range(1, parallel.size), 0]
            delivered: set[int] = set()
            for rank in rank_order:
                try:
                    partition_data = localize_mesh_and_geometry(
                        global_mesh,
                        global_geo,
                        rank,
                        parallel.size,
                        include_visualization_ghosts=configuration.output.ghost_layers == 1,
                    )
                    partition_data[0]["global_mesh_hash"] = global_hash
                except BaseException:
                    for destination in range(1, parallel.size):
                        if destination not in delivered:
                            comm.send((False, None), dest=destination, tag=_PARTITION_TAG)
                    raise
                if rank == 0:
                    local_partition_data = partition_data
                else:
                    comm.send((True, partition_data), dest=rank, tag=_PARTITION_TAG)
                    delivered.add(rank)
                    partition_data = None
        else:
            # After a failure rank zero sends (False, None); the phase
            # then raises rank zero's error on this rank too.
            _, local_partition_data = comm.recv(source=0, tag=_PARTITION_TAG)
    assert local_partition_data is not None
    mesh_data, geo_data, partition = local_partition_data
    mesh_quality = parallel.bcast(quality, root=0)
    parallel = parallel.with_partition(partition)
    mesh_data["_parallel_context"] = parallel
    return mesh_data, geo_data, mesh_quality, parallel
