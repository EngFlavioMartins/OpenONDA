"""Opt-in synchronized component timers for bounded solver qualification.

These nested diagnostic measurements are NOT additive. They wrap only Python
entry points; kernels, physical inputs, output schedules and solver tolerances
are unchanged. Fine GBD measurements synchronize batch-kernel entry points as
well as host phases and therefore must not be interpreted as uninstrumented
production timings. Restore all instance methods on exit.
"""

from contextlib import contextmanager
from functools import wraps
from time import perf_counter

# Keep these at phase/batch granularity. Per-particle wall-correction and segment
# classifier calls would add thousands of synchronizations and distort the work
# being measured. Methods absent from a particular physics backend are skipped.
_GBD_METHODS = (
    "gbd_diffusion",
    "_gbd_diffusion_impl",
    "_compute_grid_bounds",
    "_lattice_aligned_bounds",
    "_ensure_grid_capacity",
    "_allocate_grid",
    "_body_interior_at_particles",
    "_prepare_body_mask_current_grid",
    "_prepare_body_links",
    "_zero_grid_kernel",
    "_m4_scatter_gpu_kernel",
    "_m4_scatter_slip_image_kernel",
    "_lagrange6_scatter_gpu_kernel",
    "_m4_wall_corrections",
    "_reflect_sparse_wall_corrections",
    "_add_sparse_wall_corrections_kernel",
    "_apply_body_mask_current_grid",
    "_advance_gbd_laplacian",
    "_scatter_zone_ids",
    "_scatter_id_field",
    "_scatter_scalar_weighted",
    "_download_active_vec_grid",
    "_upload_active_vec_grid",
    "_upload_active_scalar_grid",
    "_weight_slip_slab_endpoint_nodes",
    "_select_diffusion_threshold",
    "_wall_recovery_labels",
    "_augment_moment_recovery_support",
    "_redistribute_pruned_moments",
    "_build_diffusion_particle_arrays",
)


@contextmanager
def _method_timers(measurements):
    """Share synchronized Python-entry timers and exception-safe restoration."""
    import taichi as ti

    saved = []
    wrapped = set()

    def wrap(owner, method, label, *, before=None):
        key = (id(owner), method)
        if key in wrapped:
            return
        original = getattr(owner, method, None)
        if not callable(original):
            return
        local = method in vars(owner)

        @wraps(original)
        def measured(*args, **kwargs):
            if before is not None:
                before()
            ti.sync()
            start = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                ti.sync()
                row = measurements.setdefault(label, {"calls": 0, "seconds": 0.0})
                row["calls"] += 1
                row["seconds"] += perf_counter() - start

        setattr(owner, method, measured)
        saved.append((owner, method, original, local))
        wrapped.add(key)

    try:
        yield wrap
    finally:
        for owner, method, original, local in reversed(saved):
            if local:
                setattr(owner, method, original)
            else:
                delattr(owner, method)


@contextmanager
def profile_grid_diffusion(physics, measurements):
    """Instrument a private PhysicsEngine without constructing a solver."""
    with _method_timers(measurements) as wrap:
        for method in _GBD_METHODS:
            wrap(physics, method, "gbd." + method)
        yield


@contextmanager
def profile_components(coupler, measurements, *, gbd_detail=True):
    if not coupler._is_master:
        yield
        return
    vpm = coupler.vpm_solver

    def attach_grid_timers():
        if not gbd_detail:
            return
        # PhysicsEngine owns the GBD mixin; its device grid is allocated lazily.
        # Resolve the owner immediately before diffusion, not when entering this
        # context. This also follows replacement/lazy physics properties without
        # forcing device allocation in a context that never invokes diffusion.
        physics = getattr(vpm, "physics", None)
        if physics is None:
            physics = getattr(vpm.stepper, "physics", None)
        if physics is not None:
            for method in _GBD_METHODS:
                wrap(physics, method, "gbd." + method)

    with _method_timers(measurements) as wrap:
        for method in (
            "_advance_particles",
            "_apply_viscous_diffusion",
            "_update_velocity_and_gradients",
            "_apply_grid_diffusion",
        ):
            wrap(
                vpm.stepper,
                method,
                "vpm." + method,
                before=attach_grid_timers if method == "_apply_grid_diffusion" else None,
            )
        wrap(vpm.stage_rhs, "evaluate", "vpm.complete_stage_rhs")
        if callable(getattr(vpm.stage_rhs, "evaluate_induction", None)):
            # Preserve the existing backend's method references. Wrapping
            # evaluate_stage or _images would (correctly) make the reuse
            # comparison_settings decline an unknown modified numerical operator.
            wrap(vpm.stage_rhs, "evaluate_induction", "vpm.complete_induction")
        else:
            # Immutable pre-reuse implementations remain valid controls.
            wrap(vpm.induction, "evaluate_stage", "vpm.complete_induction")
            if hasattr(vpm.induction, "_images"):
                wrap(vpm.induction, "_images", "vpm.image_fields")
        guard = getattr(vpm.stage_rhs.position_guard, "__self__", None)
        if guard is not None and callable(getattr(guard, "_project", None)):
            wrap(guard, "_project", "vpm.wall_projection")
        wrap(vpm.output_manager, "dispatch", "vpm.output_dispatch")
        wrap(coupler, "_transfer_vorticity_to_vpm", "coupler.transfer")
        yield
