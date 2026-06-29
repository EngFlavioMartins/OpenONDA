"""
Taichi GPU-Accelerated Barnes-Hut Treecode for VPM.
====================================================
Fully parallel LBVH (Linear Bounding Volume Hierarchy) construction and
binary-tree traversal using Taichi for GPU acceleration.

The tree is built on-device via:
  1. AABB computation (parallel min/max reduction)
  2. Morton-code encoding (30-bit, 10 bits/axis)
  3. CPU argsort (Phase 1; GPU radix-sort in Phase 2)
  4. GPU Karras radix tree (O(N), fully parallel — one thread per internal node)
  5. Level-synchronous bottom-up multipole moments (one kernel per tree level)

The tree construction (step 4) uses the Karras 2012 algorithm with each internal
node solved independently (no serial stack, no shared scratch, Vulkan-portable):
  - δ(i, j) = common-prefix length of sorted Morton keys (index tie-break on ties)
  - each internal node determines its [first, last] range and split via δ +
    exponential/binary search
  - internal node 0 is the root (covers [0, N-1])

All phases are GPU-resident except the CPU argsort (~0.2 MB at 49k) and a single
device→host read of N is avoided (the multipole level bound is derived from N).

Author:  Flavio A. C. Martins (f.m.martins@tudelft.nl), OpenONDA Team
Date: February 2026

Copyright (C) 2026 Flavio A. C. Martins, OpenONDA
"""

import time

import numpy as np
import taichi as ti

@ti.data_oriented
class TaichiTreecode:
    """
    GPU-accelerated Barnes-Hut treecode using Taichi with LBVH build.

    The tree is a **binary** radix tree (Karras 2012) constructed from
    Morton-coded particle positions.  Traversal uses the same stack-based
    MAC-driven iteration as the previous octree code, adapted to two
    children per internal node.

    Parameters:
        max_particles: Maximum number of particles
        max_nodes: Maximum number of tree nodes (typically 2*max_particles)
        theta: Opening angle for MAC criterion
        max_leaf_size: Maximum particles per leaf node
    """

    def __init__(
        self,
        max_particles: int = 100000,
        max_nodes: int = 200000,
        theta: float = 0.5,
        max_leaf_size: int = 32,
        kernel_type: str = "WINCKELMANS",
    ):
        self.max_particles = max_particles
        self.max_nodes = max_nodes
        self.theta = theta
        self.max_leaf_size = max_leaf_size
        self.theta_sq = theta * theta
        self.kernel_type = kernel_type.upper()

        # ─────────────────────────────────────────────────────────────
        # PARTICLE DATA (copied from input via GPU kernel — no to_numpy)
        # ─────────────────────────────────────────────────────────────
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.circulations = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.radii = ti.field(dtype=ti.f32, shape=max_particles)

        # ─────────────────────────────────────────────────────────────
        # TREE STRUCTURE (binary LBVH)
        # ─────────────────────────────────────────────────────────────
        # Node properties (center/half_size computed from AABB)
        self.node_center = ti.Vector.field(3, dtype=ti.f32, shape=max_nodes)
        self.node_half_size = ti.field(dtype=ti.f32, shape=max_nodes)

        # Multipole moments
        self.node_total_circ = ti.Vector.field(3, dtype=ti.f32, shape=max_nodes)
        self.node_com = ti.Vector.field(3, dtype=ti.f32, shape=max_nodes)
        self.node_avg_radius = ti.field(dtype=ti.f32, shape=max_nodes)

        # Binary tree structure (left/right child, -1 = none)
        self.node_left = ti.field(dtype=ti.i32, shape=max_nodes)
        self.node_right = ti.field(dtype=ti.i32, shape=max_nodes)
        # Parent pointer (root = -1); derived from the child links after build.
        self.node_parent = ti.field(dtype=ti.i32, shape=max_nodes)
        # Depth from the root, used by the level-synchronous multipole pass.
        self.node_depth = ti.field(dtype=ti.i32, shape=max_nodes)
        self.node_is_leaf = ti.field(dtype=ti.i32, shape=max_nodes)
        self.node_particle_start = ti.field(dtype=ti.i32, shape=max_nodes)
        self.node_particle_count = ti.field(dtype=ti.i32, shape=max_nodes)

        # Particle-to-leaf mapping (contiguous sorted indices)
        self.leaf_particles = ti.field(dtype=ti.i32, shape=max_particles)

        # ─────────────────────────────────────────────────────────────
        # LBVH BUILD FIELDS
        # ─────────────────────────────────────────────────────────────
        self.morton_codes = ti.field(dtype=ti.u32, shape=max_particles)
        self.sorted_indices = ti.field(dtype=ti.i32, shape=max_particles)
        # LCP array (for Karras tree; length max_particles for simplicity)
        self._lcp = ti.field(dtype=ti.i32, shape=max_particles)
        # Nearest smaller LCP left/right (for Karras tree construction)
        self._nsl = ti.field(dtype=ti.i32, shape=max_particles)
        self._nsr = ti.field(dtype=ti.i32, shape=max_particles)
        # Temporary stack for serial NSL/NSR computation (GPU)
        self._stack = ti.field(dtype=ti.i32, shape=max_particles)
        # Node particle range in sorted order
        self._node_first = ti.field(dtype=ti.i32, shape=max_nodes)
        self._node_last = ti.field(dtype=ti.i32, shape=max_nodes)
        # Node AABB
        self._node_aabb_min = ti.Vector.field(3, dtype=ti.f32, shape=max_nodes)
        self._node_aabb_max = ti.Vector.field(3, dtype=ti.f32, shape=max_nodes)

        # ─────────────────────────────────────────────────────────────
        # OUTPUT VELOCITIES
        # ─────────────────────────────────────────────────────────────
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)

        # ─────────────────────────────────────────────────────────────
        # OUTPUT VELOCITY GRADIENTS AND STRAIN RATES
        # ─────────────────────────────────────────────────────────────
        self.velocity_gradients = ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_particles)
        self.strain_rates = ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_particles)

        # ─────────────────────────────────────────────────────────────
        # TARGET POINT FIELDS
        # ─────────────────────────────────────────────────────────────
        self.max_targets = max_particles
        self.target_positions = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.target_velocities = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.target_velocity_gradients = ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_particles)
        self.n_targets = ti.field(dtype=ti.i32, shape=())
        self.kernel_type_id = ti.field(dtype=ti.i32, shape=())

        # ─────────────────────────────────────────────────────────────
        # TREE TRAVERSAL STACK
        # ─────────────────────────────────────────────────────────────
        self.max_stack_depth = 48
        self.traversal_stack = ti.field(dtype=ti.i32, shape=(max_particles, 48))
        self.target_traversal_stack = ti.field(dtype=ti.i32, shape=(max_particles, 48))

        # ─────────────────────────────────────────────────────────────
        # COUNTERS
        # ─────────────────────────────────────────────────────────────
        self.n_particles = ti.field(dtype=ti.i32, shape=())
        self.n_nodes = ti.field(dtype=ti.i32, shape=())
        self._root = ti.field(dtype=ti.i32, shape=())
        self._max_depth = ti.field(dtype=ti.i32, shape=())

        # Background velocity
        self.u_inf = ti.Vector.field(3, dtype=ti.f32, shape=())

        # Statistics
        self.build_time = 0.0
        self.eval_time = 0.0
        self.grad_time = 0.0

        # AABB fields (allocated at instance level, not class level)
        self._aabb_min = ti.Vector.field(3, dtype=ti.f32, shape=())
        self._aabb_max = ti.Vector.field(3, dtype=ti.f32, shape=())

        # Constants for gradient kernel
        self.MIN_R_SIGMA_GRADIENT = 0.5
        self.DEFAULT_CUTOFF_RADIUS_FACTOR = 15.0

        self.set_kernel_type(self.kernel_type)

    def set_kernel_type(self, kernel_type: str) -> None:
        normalized = kernel_type.upper()
        if normalized == "GAUSSIAN":
            kernel_id = 0
        elif normalized == "WINCKELMANS":
            kernel_id = 1
        else:
            raise ValueError(f"Unsupported treecode kernel_type: {kernel_type}")
        self.kernel_type = normalized
        self.kernel_type_id[None] = kernel_id

    # BUILD — On-GPU LBVH construction

    def build(self, positions=None, circulations=None, radii=None, N=None,
              force: bool = False) -> None:
        """
        Build LBVH binary tree from particle data.

        Supports two calling conventions:

        1. Field-based (preferred)::
            tree.build(pos_field, strg_field, rad_field, N)

           Copies from Taichi fields directly — **no CPU round-trip**.

        2. NumPy arrays (backward compat)::
            tree.build(pos_np, strg_np, rad_np)

        When called with the same *N* (and no *force*) repeatedly, the build
        is skipped — the tree is already valid.

        Args:
            positions:  Taichi vec3 field *or* NumPy array [N,3] of positions.
            circulations: Taichi vec3 field *or* NumPy array [N,3].
            radii:  Taichi scalar field *or* NumPy array [N].
            N:  Number of active particles (required for field API).
            force:  Accepted for API compatibility; the tree is always rebuilt.
        """
        # The tree is rebuilt on every call.  A previous N-only guard skipped
        # the rebuild whenever the particle *count* was unchanged — but N is
        # constant across timesteps and RK sub-stages while positions/strengths
        # change every call, so that guard froze the tree at its first
        # configuration and silently corrupted every subsequent evaluation.
        # Correct build-once-per-configuration reuse needs a real change signal
        # (a position/strength version token) and is deferred; correctness first.
        t_start = time.perf_counter()

        # ── Detect calling convention ──────────────────────────────────
        using_fields = hasattr(positions, 'shape') and hasattr(positions, '__getitem__') \
                       and not isinstance(positions, np.ndarray)

        if using_fields:
            N_val = N if N is not None else self.n_particles[None]
            if N_val > self.max_particles:
                raise ValueError(
                    f"Too many particles ({N_val}) for treecode capacity ({self.max_particles})"
                )
            self.n_particles[None] = N_val
            self._copy_particle_fields(positions, circulations, radii, N_val)
        else:
            pos_np = positions
            strg_np = circulations
            rad_np = radii
            N_val = len(pos_np)
            if N_val > self.max_particles:
                raise ValueError(
                    f"Too many particles ({N_val}) for treecode capacity ({self.max_particles})"
                )
            self.n_particles[None] = N_val
            self._upload_numpy_particles(pos_np, strg_np, rad_np, N_val)

        # ── Build LBVH ────────────────────────────────────────────────
        self._build_lbvh(N_val)

        self.build_time = time.perf_counter() - t_start

    def _upload_numpy_particles(self, pos_np, strg_np, rad_np, N):
        """Upload NumPy particle arrays to GPU fields."""
        def pad_2d(arr):
            padded = np.zeros((self.max_particles, 3), dtype=np.float32)
            padded[:N] = arr.astype(np.float32)
            return padded
        def pad_1d(arr):
            padded = np.zeros(self.max_particles, dtype=np.float32)
            padded[:N] = arr.astype(np.float32)
            return padded
        self.positions.from_numpy(pad_2d(pos_np))
        self.circulations.from_numpy(pad_2d(strg_np))
        self.radii.from_numpy(pad_1d(rad_np))
        ti.sync()

    @ti.kernel
    def _copy_particle_fields(self, pos: ti.template(), strg: ti.template(),
                               rad: ti.template(), N: ti.i32):
        """Copy particle data from source Taichi fields to treecode fields."""
        for i in range(N):
            self.positions[i] = pos[i]
            self.circulations[i] = strg[i]
            self.radii[i] = rad[i]

    # ── GPU Karras tree build (fully parallel, Vulkan-portable) ──────────
    #
    # Replaces the former LCP + serial-stack NSL/NSR Cartesian-tree build.
    # That build used ``ti.loop_config(serialize=True)`` over a shared ``_stack``
    # field, which Vulkan does not reliably serialize (the monotonic stack then
    # races → degenerate tree → ~0 velocities).  The Karras 2012 construction
    # below assigns each internal node's range and split *independently* — one
    # thread per node, no shared scratch, no serialize — so it is correct on
    # CPU, Vulkan, and CUDA alike.

    @ti.func
    def _clz32(self, x: ti.u32) -> ti.i32:
        """Count leading zeros of a 32-bit word (x may be 0 → 32)."""
        n = 0
        if x == ti.u32(0):
            n = 32
        else:
            v = x
            while (v & ti.u32(0x80000000)) == ti.u32(0):
                v = v << ti.u32(1)
                n += 1
        return n

    @ti.func
    def _delta(self, i: ti.i32, j: ti.i32, N: ti.i32) -> ti.i32:
        """Karras δ: common-prefix length of sorted Morton keys i and j.

        Out-of-range j → -1.  Equal keys fall back to the index tie-break
        (32 + clz(i^j)), which makes the effective keys distinct and bounds the
        tree depth (so duplicate Morton cells cannot create an unbounded chain).
        """
        res = -1
        if j >= 0 and j < N:
            ki = self.morton_codes[self.sorted_indices[i]]
            kj = self.morton_codes[self.sorted_indices[j]]
            x = ki ^ kj
            if x == ti.u32(0):
                res = 32 + self._clz32(ti.u32(i) ^ ti.u32(j))
            else:
                res = self._clz32(x)
        return res

    @ti.kernel
    def _build_karras_parallel_kernel(self, N: ti.i32):
        """Parallel Karras 2012 radix-tree build.

        Internal node ``i`` is stored at field index ``N + i`` and covers a
        contiguous sorted range; leaf ``j`` is stored at field index ``j``.
        Internal node 0 is always the root (covers [0, N-1]).
        """
        # ── init all nodes (parallel) ──
        for idx in range(2 * N - 1):
            self.node_left[idx] = -1
            self.node_right[idx] = -1
            self.node_is_leaf[idx] = 0
            self._node_first[idx] = -1
            self._node_last[idx] = -1
            self.node_particle_start[idx] = -1
            self.node_particle_count[idx] = 0
        # ── leaves (parallel) ──
        for j in range(N):
            self.node_is_leaf[j] = 1
            self.leaf_particles[j] = self.sorted_indices[j]
            self._node_first[j] = j
            self._node_last[j] = j
            self.node_particle_start[j] = j
            self.node_particle_count[j] = 1
        # ── internal nodes: one independent thread per node ──
        for i in range(N - 1):
            # Direction of the range (toward the neighbour with longer prefix).
            dl = self._delta(i, i - 1, N)
            dr = self._delta(i, i + 1, N)
            d = 1
            if dr <= dl:
                d = -1
            delta_min = self._delta(i, i - d, N)
            # Exponential search for a far end of the range.
            l_max = 2
            while self._delta(i, i + l_max * d, N) > delta_min:
                l_max *= 2
            # Binary search for the precise other end.
            l = 0
            t = l_max >> 1
            while t >= 1:
                if self._delta(i, i + (l + t) * d, N) > delta_min:
                    l += t
                t = t >> 1
            j = i + l * d
            first = ti.min(i, j)
            last = ti.max(i, j)
            # Find split: last index of the left subrange.
            node_delta = self._delta(first, last, N)
            split = first
            stride = last - first
            cont = 1
            while cont == 1:
                stride = (stride + 1) >> 1
                newsplit = split + stride
                if newsplit < last:
                    if self._delta(first, newsplit, N) > node_delta:
                        split = newsplit
                if stride <= 1:
                    cont = 0
            # Children: leaf if the sub-range is a single element, else internal.
            left = split if split == first else (N + split)
            right = (split + 1) if (split + 1) == last else (N + split + 1)
            idx = N + i
            self.node_left[idx] = left
            self.node_right[idx] = right
            self._node_first[idx] = first
            self._node_last[idx] = last
            self.node_particle_start[idx] = first
            self.node_particle_count[idx] = last - first + 1
            if i == 0:
                self._root[None] = N  # internal node 0 is the root

    def _build_lbvh(self, N):
        """Internal LBVH build pipeline (all steps after data upload)."""
        if N <= 1:
            # Trivial: single particle, root = that particle
            self.n_nodes[None] = N
            self._root[None] = 0 if N == 0 else 0
            self.node_is_leaf[0] = 1
            self.node_particle_start[0] = 0
            self.node_particle_count[0] = N
            if N > 0:
                self.leaf_particles[0] = 0
                self.node_total_circ[0] = self.circulations[0]
                self.node_com[0] = self.positions[0]
                self.node_center[0] = self.positions[0]
                self.node_half_size[0] = 0.0
                self.node_avg_radius[0] = self.radii[0]
                self.node_left[0] = -1
                self.node_right[0] = -1
                self.node_parent[0] = -1
            return

        # No defensive ``ti.sync()`` between build kernels: Taichi submits them to
        # one device queue in order, and each kernel-launch boundary is itself the
        # ordering guarantee (data-dependent kernels see their predecessor's
        # writes).  The multipole pass is *level-synchronous* (one kernel per tree
        # level, deepest→root) precisely so the bottom-up has no cross-thread
        # read-after-write hazard — the previous design's atomic-climb walk did,
        # and the inter-kernel barriers only masked it by timing.

        # Step 1: AABB (parallel min/max reduction)
        self._compute_aabb_kernel(N)

        # Step 2: Morton codes (30-bit) — reads the AABB written above
        self._compute_morton_codes_kernel(N, self._aabb_min, self._aabb_max)

        # Step 3: sort keys (Phase 1 — CPU argsort; Phase 2 → GPU radix sort).
        # ``to_numpy``/``from_numpy`` are the build's only host↔device barrier.
        morton_np = self.morton_codes.to_numpy()[:N]
        sorted_idx = np.argsort(morton_np, kind='mergesort')
        padded = np.full(self.max_particles, -1, dtype=np.int32)
        padded[:N] = sorted_idx.astype(np.int32)
        self.sorted_indices.from_numpy(padded)

        # Step 4: GPU Karras tree — fully parallel, one thread per internal node
        # (no serialize, no shared scratch → Vulkan-correct).
        self._build_karras_parallel_kernel(N)

        # Step 5: parents + depths, then the level-synchronous bottom-up.
        self._compute_parents_kernel(N)
        self._leaf_multipole_init_kernel(N)
        self._compute_depths_kernel(N)
        # One combine kernel per level, deepest → root.  Each level reads only the
        # already-finalised level below it; the kernel-launch boundary orders them,
        # so the result is identical on CPU / Vulkan / CUDA with no fence and no
        # race.  The loop bound is computed from N alone (no mid-build device→host
        # read of a scalar, which is unreliable on Vulkan): the Karras tree depth
        # is at most ~30 Morton bits + the index tie-break ≈ 2·ceil(log2 N), and
        # combine kernels past the real depth are no-ops.
        max_levels = 2 * int(np.ceil(np.log2(max(N, 2)))) + 34
        for level in range(max_levels, -1, -1):
            self._combine_level_kernel(N, level)

        self.n_nodes[None] = 2 * N - 1

    # ── AABB kernel ──────────────────────────────────────────────────

    @ti.kernel
    def _compute_aabb_kernel(self, N: ti.i32):
        """Parallel min/max reduction over particle positions.

        The static init unrolls to three scalar stores that run before the
        parallel ``range(N)`` loop; the per-particle atomics then reduce
        concurrently across threads (no ``serialize`` needed).
        """
        for k in ti.static(range(3)):
            self._aabb_min[None][k] = 1e10
            self._aabb_max[None][k] = -1e10
        for i in range(N):
            p = self.positions[i]
            for k in ti.static(range(3)):
                ti.atomic_min(self._aabb_min[None][k], p[k])
                ti.atomic_max(self._aabb_max[None][k], p[k])

    # ── Morton-code kernel ────────────────────────────────────────────

    @ti.func
    def _expand_bits(self, v: ti.u32) -> ti.u32:
        """Expand 10-bit integer to 30-bit by inserting two zero bits
        between each original bit (for 3D Morton code)."""
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    @ti.kernel
    def _compute_morton_codes_kernel(self, N: ti.i32,
                                      aabb_min: ti.template(),
                                      aabb_max: ti.template()):
        """Quantize f32 positions to 30-bit Morton codes (10 bits/axis)."""
        inv_range = ti.Vector([0.0, 0.0, 0.0])
        for k in ti.static(range(3)):
            span = aabb_max[None][k] - aabb_min[None][k]
            inv_range[k] = 1023.0 / (span + 1e-30)

        for i in range(N):
            p = self.positions[i]
            qi = ti.Vector([ti.u32(0), ti.u32(0), ti.u32(0)])
            for k in ti.static(range(3)):
                q = ti.u32(ti.floor((p[k] - aabb_min[None][k]) * inv_range[k]))
                qi[k] = ti.min(q, ti.u32(1023))
            code = ti.u32(0)
            code |= self._expand_bits(qi[0]) << 2
            code |= self._expand_bits(qi[1]) << 1
            code |= self._expand_bits(qi[2])
            self.morton_codes[i] = code

    # ── Bottom-up multipoles ──────────────────────────────────────────

    @ti.kernel
    def _compute_parents_kernel(self, N: ti.i32):
        """Derive node_parent from the child links set by the Karras kernel.

        Robust by construction: every node is the child of exactly one internal
        node, so scanning internal nodes and writing parent[child]=idx covers
        all non-root nodes; the root is the only node never written and keeps
        its -1 sentinel.
        """
        for idx in range(2 * N - 1):
            self.node_parent[idx] = -1
        for idx in range(N, 2 * N - 1):
            left = self.node_left[idx]
            right = self.node_right[idx]
            if left >= 0:
                self.node_parent[left] = idx
            if right >= 0:
                self.node_parent[right] = idx

    @ti.kernel
    def _leaf_multipole_init_kernel(self, N: ti.i32):
        """Seed leaf multipoles/AABB directly from particles.

        Leaves carry a single particle, so their multipole (COM = position,
        total_circ = circulation, avg_radius = radius) is exact and their
        geometric extent is zero — node_half_size = 0 makes the MAC always
        accept a leaf, evaluating it via its exact single-particle multipole.
        Setting these explicitly also overwrites any stale values left in the
        node fields from a previous (possibly larger) build.
        """
        for j in range(N):
            p = self.sorted_indices[j]
            self.node_total_circ[j] = self.circulations[p]
            self.node_avg_radius[j] = self.radii[p]
            self.node_com[j] = self.positions[p]
            self.node_center[j] = self.positions[p]
            self.node_half_size[j] = 0.0
            self._node_aabb_min[j] = self.positions[p]
            self._node_aabb_max[j] = self.positions[p]

    @ti.kernel
    def _compute_depths_kernel(self, N: ti.i32):
        """Depth (distance to root) of every node, and the max over the tree.

        One thread per node climbs ``node_parent`` to the root counting hops, so
        no ordering is required.  ``_max_depth`` bounds the level-synchronous
        combine loop; it is the only host read in the multipole pass.
        """
        self._max_depth[None] = 0
        for idx in range(2 * N - 1):
            d = 0
            node = self.node_parent[idx]
            while node >= 0:
                node = self.node_parent[node]
                d += 1
            self.node_depth[idx] = d
            ti.atomic_max(self._max_depth[None], d)

    @ti.kernel
    def _combine_level_kernel(self, N: ti.i32, level: ti.i32):
        """Combine every internal node at ``level`` from its two children.

        Called once per level, deepest first.  A node at ``level`` has children at
        ``level + 1`` that the previous launch already finalised, so this is a
        pure read of the level below — no cross-thread hazard, correct on every
        backend without a fence.  Leaves (idx < N) are seeded separately.
        """
        for idx in range(N, 2 * N - 1):
            if self.node_depth[idx] == level:
                self._combine_node(idx)

    @ti.func
    def _combine_node(self, idx: ti.i32):
        left = self.node_left[idx]
        right = self.node_right[idx]
        if left >= 0 and right >= 0:
            first = ti.min(self._node_first[left], self._node_first[right])
            last = ti.max(self._node_last[left], self._node_last[right])
            self._node_first[idx] = first
            self._node_last[idx] = last
            self.node_particle_start[idx] = first
            self.node_particle_count[idx] = last - first + 1

            total_circ = self.node_total_circ[left] + self.node_total_circ[right]
            self.node_total_circ[idx] = total_circ

            mag_l = ti.sqrt(self.node_total_circ[left].dot(self.node_total_circ[left]))
            mag_r = ti.sqrt(self.node_total_circ[right].dot(self.node_total_circ[right]))
            total_mag = mag_l + mag_r
            com = ti.Vector([0.0, 0.0, 0.0])
            if total_mag > 1e-15:
                com = (self.node_com[left] * mag_l + self.node_com[right] * mag_r) / total_mag
            else:
                com = (self.node_com[left] + self.node_com[right]) * 0.5
            self.node_com[idx] = com

            count_l = max(self.node_particle_count[left], 1)
            count_r = max(self.node_particle_count[right], 1)
            total_count = count_l + count_r
            avg_rad = (self.node_avg_radius[left] * count_l
                       + self.node_avg_radius[right] * count_r) / total_count
            self.node_avg_radius[idx] = avg_rad

            aabb_min = ti.Vector([0.0, 0.0, 0.0])
            aabb_max = ti.Vector([0.0, 0.0, 0.0])
            for k in ti.static(range(3)):
                aabb_min[k] = ti.min(self._node_aabb_min[left][k],
                                     self._node_aabb_min[right][k])
                aabb_max[k] = ti.max(self._node_aabb_max[left][k],
                                     self._node_aabb_max[right][k])
            self._node_aabb_min[idx] = aabb_min
            self._node_aabb_max[idx] = aabb_max

            center = (aabb_min + aabb_max) * 0.5
            self.node_center[idx] = center
            half_size = ti.sqrt((aabb_max - aabb_min).dot(aabb_max - aabb_min)) * 0.5
            self.node_half_size[idx] = max(half_size, 1e-8)

    # KERNEL FUNCTIONS (qf, zeta, erf, skew)

    @ti.func
    def _erf_approx(self, x: ti.f32) -> ti.f32:
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.327591100
        sign = ti.cast(1.0, ti.f32)
        x_abs = x
        if x < 0.0:
            sign = -1.0
            x_abs = -x
        t = 1.0 / (1.0 + p * x_abs)
        y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * ti.exp(-x_abs * x_abs))
        return sign * y

    @ti.func
    def q_kernel(self, r_sigma: ti.f32) -> ti.f32:
        ONE_OVER_FOUR_PI = ti.cast(0.07957747154594767, ti.f32)
        result = ti.cast(0.0, ti.f32)
        if self.kernel_type_id[None] == 0:
            two_over_sqrt_pi = ti.cast(1.1283791671, ti.f32)
            if r_sigma < 1e-4:
                result = (
                    (4.0 / (3.0 * ti.sqrt(ti.acos(-1.0) ** 3))) * (r_sigma**3) * ONE_OVER_FOUR_PI
                )
            else:
                erf_term = self._erf_approx(r_sigma)
                exp_term = two_over_sqrt_pi * r_sigma * ti.exp(-r_sigma * r_sigma)
                result = (erf_term - exp_term) * ONE_OVER_FOUR_PI
        else:
            r2 = r_sigma * r_sigma
            base = r2 + 1.0  # > 0: x**2.5 = x²·√x (no transcendental pow)
            result = (
                r_sigma * r2 * (r2 + 2.5) / (base * base * ti.sqrt(base)) * ONE_OVER_FOUR_PI
            )
        return result

    @ti.func
    def zeta_kernel(self, r_sigma: ti.f32) -> ti.f32:
        ONE_OVER_FOUR_PI = ti.cast(0.07957747154594767, ti.f32)
        result = ti.cast(0.0, ti.f32)
        if self.kernel_type_id[None] == 0:
            one_over_pi_15 = ti.cast(0.179587122125, ti.f32)
            result = one_over_pi_15 * ti.exp(-r_sigma * r_sigma)
        else:
            r2 = r_sigma * r_sigma
            base = r2 + 1.0  # > 0: x**3.5 = x³·√x (no transcendental pow)
            result = 7.5 / (base * base * base * ti.sqrt(base)) * ONE_OVER_FOUR_PI
        return result

    @ti.func
    def skew(self, v: ti.template()) -> ti.Matrix:
        return ti.Matrix([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

    # LEAF SUMMATION FUNCTIONS

    @ti.func
    def _leaf_velocity_sum(
        self, node: int, target_pos: ti.template(), target_rad: ti.f32, self_idx: int
    ) -> ti.math.vec3:
        vel = ti.Vector([0.0, 0.0, 0.0])
        start = self.node_particle_start[node]
        count = self.node_particle_count[node]
        for k in range(count):
            j = self.leaf_particles[start + k]
            if j != self_idx:
                r_vec_j = target_pos - self.positions[j]
                r_mag_j = ti.sqrt(r_vec_j.dot(r_vec_j))
                if r_mag_j > 1e-10:
                    sigma = 0.5 * (target_rad + self.radii[j])
                    q_val = self.q_kernel(r_mag_j / sigma)
                    vel -= q_val * r_vec_j.cross(self.circulations[j]) / (r_mag_j**3)
        return vel

    @ti.func
    def _target_leaf_velocity_sum(self, node: int, target_pos: ti.template()) -> ti.math.vec3:
        vel = ti.Vector([0.0, 0.0, 0.0])
        start = self.node_particle_start[node]
        count = self.node_particle_count[node]
        for k in range(count):
            j = self.leaf_particles[start + k]
            r_vec_j = target_pos - self.positions[j]
            r_mag_j = ti.sqrt(r_vec_j.dot(r_vec_j))
            if r_mag_j > 1e-10:
                sigma = self.radii[j]
                q_val = self.q_kernel(r_mag_j / sigma)
                vel -= q_val * r_vec_j.cross(self.circulations[j]) / (r_mag_j**3)
        return vel

    @ti.func
    def _leaf_gradient_sum(
        self, node: int, target_pos: ti.template(), target_rad: ti.f32,
        self_idx: int, min_r_sigma: ti.f32, max_r_sigma: ti.f32
    ) -> ti.Matrix:
        gradu = ti.Matrix.zero(ti.f32, 3, 3)
        start = self.node_particle_start[node]
        count = self.node_particle_count[node]
        for k in range(count):
            j = self.leaf_particles[start + k]
            if j != self_idx:
                r_vec_j = target_pos - self.positions[j]
                r_mag_j = ti.sqrt(r_vec_j.dot(r_vec_j))
                if r_mag_j > 1e-10:
                    sigma = 0.5 * (target_rad + self.radii[j])
                    r_sigma = r_mag_j / sigma
                    if r_sigma > min_r_sigma and r_sigma < max_r_sigma:
                        q_val = self.q_kernel(r_sigma)
                        zeta_val = self.zeta_kernel(r_sigma) / sigma**3
                        term1 = q_val / r_mag_j**3
                        term2 = 3.0 * q_val / r_mag_j**5 - zeta_val / r_mag_j**2
                        cross_j = r_vec_j.cross(self.circulations[j])
                        gradu += term1 * self.skew(
                            self.circulations[j]
                        ) + term2 * cross_j.outer_product(r_vec_j)
        return gradu

    @ti.func
    def _target_leaf_gradient_sum(
        self, node: int, target_pos: ti.template(), min_r_sigma: ti.f32
    ) -> ti.Matrix:
        gradu = ti.Matrix.zero(ti.f32, 3, 3)
        start = self.node_particle_start[node]
        count = self.node_particle_count[node]
        for k in range(count):
            j = self.leaf_particles[start + k]
            r_vec_j = target_pos - self.positions[j]
            r_mag_j = ti.sqrt(r_vec_j.dot(r_vec_j))
            if r_mag_j > 1e-10:
                sigma = self.radii[j]
                r_sigma = r_mag_j / sigma
                if r_sigma > min_r_sigma:
                    q_val = self.q_kernel(r_sigma)
                    zeta_val = self.zeta_kernel(r_sigma) / sigma**3
                    term1 = q_val / r_mag_j**3
                    term2 = 3.0 * q_val / r_mag_j**5 - zeta_val / r_mag_j**2
                    cross_j = r_vec_j.cross(self.circulations[j])
                    gradu += term1 * self.skew(
                        self.circulations[j]
                    ) + term2 * cross_j.outer_product(r_vec_j)
        return gradu

    # TRAVERSAL — Binary-tree stack-based

    @ti.func
    def _push_children_particle(self, i: int, node: int, stack_ptr: int) -> int:
        """Push children of *node* onto the per-thread particle traversal stack."""
        left = self.node_left[node]
        right = self.node_right[node]
        if right >= 0 and stack_ptr < self.max_stack_depth - 1:
            self.traversal_stack[i, stack_ptr] = right
            stack_ptr += 1
        if left >= 0 and stack_ptr < self.max_stack_depth - 1:
            self.traversal_stack[i, stack_ptr] = left
            stack_ptr += 1
        return stack_ptr

    @ti.func
    def _push_children_target(self, i: int, node: int, stack_ptr: int) -> int:
        left = self.node_left[node]
        right = self.node_right[node]
        if right >= 0 and stack_ptr < self.max_stack_depth - 1:
            self.target_traversal_stack[i, stack_ptr] = right
            stack_ptr += 1
        if left >= 0 and stack_ptr < self.max_stack_depth - 1:
            self.target_traversal_stack[i, stack_ptr] = left
            stack_ptr += 1
        return stack_ptr

    @ti.func
    def _traverse_particle_vel(self, i: int, theta_sq: ti.f32, n_nodes: int) -> ti.math.vec3:
        vel = ti.Vector([0.0, 0.0, 0.0])
        target_pos = self.positions[i]
        target_rad = self.radii[i]
        root = self._root[None]
        self.traversal_stack[i, 0] = root
        stack_ptr = 1
        while stack_ptr > 0:
            stack_ptr -= 1
            node = self.traversal_stack[i, stack_ptr]
            if node < 0 or node >= n_nodes:
                continue
            com = self.node_com[node]
            r_vec = target_pos - com
            r_sq = r_vec.dot(r_vec)
            r_mag = ti.sqrt(r_sq)
            node_size = 2.0 * self.node_half_size[node]
            if r_mag > 1e-8 and (node_size * node_size / r_sq) < theta_sq:
                sigma = 0.5 * (target_rad + self.node_avg_radius[node])
                r_sigma = r_mag / sigma
                q_val = self.q_kernel(r_sigma)
                vel -= q_val * r_vec.cross(self.node_total_circ[node]) / (r_mag * r_mag * r_mag)
            elif self.node_is_leaf[node] == 1:
                vel += self._leaf_velocity_sum(node, target_pos, target_rad, i)
            else:
                stack_ptr = self._push_children_particle(i, node, stack_ptr)
        return vel

    @ti.func
    def _traverse_particle_grad(self, i: int, theta_sq: ti.f32, n_nodes: int) -> ti.Matrix:
        gradu = ti.Matrix.zero(ti.f32, 3, 3)
        target_pos = self.positions[i]
        target_rad = self.radii[i]
        MIN_R_SIGMA = ti.cast(0.5, ti.f32)
        MAX_R_SIGMA = ti.cast(15.0, ti.f32)
        root = self._root[None]
        self.traversal_stack[i, 0] = root
        stack_ptr = 1
        while stack_ptr > 0:
            stack_ptr -= 1
            node = self.traversal_stack[i, stack_ptr]
            if node < 0 or node >= n_nodes:
                continue
            com = self.node_com[node]
            r_vec = target_pos - com
            r_sq = r_vec.dot(r_vec)
            r_mag = ti.sqrt(r_sq)
            node_size = 2.0 * self.node_half_size[node]
            if r_mag > 1e-8 and (node_size * node_size / r_sq) < theta_sq:
                sigma = 0.5 * (target_rad + self.node_avg_radius[node])
                r_sigma = r_mag / sigma
                if r_sigma > MIN_R_SIGMA and r_sigma < MAX_R_SIGMA:
                    q_val = self.q_kernel(r_sigma)
                    zeta_val = self.zeta_kernel(r_sigma) / (sigma * sigma * sigma)
                    total_circ = self.node_total_circ[node]
                    term1 = q_val / (r_mag * r_mag * r_mag)
                    term2 = 3.0 * q_val / (r_mag * r_mag * r_mag * r_mag * r_mag) - zeta_val / (
                        r_mag * r_mag
                    )
                    gradu += term1 * self.skew(total_circ) + term2 * r_vec.cross(
                        total_circ
                    ).outer_product(r_vec)
            elif self.node_is_leaf[node] == 1:
                gradu += self._leaf_gradient_sum(
                    node, target_pos, target_rad, i, MIN_R_SIGMA, MAX_R_SIGMA
                )
            else:
                stack_ptr = self._push_children_particle(i, node, stack_ptr)
        return gradu

    @ti.func
    def _traverse_target_vel(self, i: int, theta_sq: ti.f32, n_nodes: int) -> ti.math.vec3:
        vel = ti.Vector([0.0, 0.0, 0.0])
        target_pos = self.target_positions[i]
        root = self._root[None]
        self.target_traversal_stack[i, 0] = root
        stack_ptr = 1
        while stack_ptr > 0:
            stack_ptr -= 1
            node = self.target_traversal_stack[i, stack_ptr]
            if node < 0 or node >= n_nodes:
                continue
            com = self.node_com[node]
            r_vec = target_pos - com
            r_sq = r_vec.dot(r_vec)
            r_mag = ti.sqrt(r_sq)
            node_size = 2.0 * self.node_half_size[node]
            if r_mag > 1e-8 and (node_size * node_size / r_sq) < theta_sq:
                sigma = self.node_avg_radius[node]
                r_sigma = r_mag / sigma
                q_val = self.q_kernel(r_sigma)
                vel -= q_val * r_vec.cross(self.node_total_circ[node]) / (r_mag * r_mag * r_mag)
            elif self.node_is_leaf[node] == 1:
                vel += self._target_leaf_velocity_sum(node, target_pos)
            else:
                stack_ptr = self._push_children_target(i, node, stack_ptr)
        return vel

    @ti.func
    def _traverse_target_grad(self, i: int, theta_sq: ti.f32, n_nodes: int) -> ti.Matrix:
        gradu = ti.Matrix.zero(ti.f32, 3, 3)
        target_pos = self.target_positions[i]
        MIN_R_SIGMA = ti.cast(0.5, ti.f32)
        root = self._root[None]
        self.target_traversal_stack[i, 0] = root
        stack_ptr = 1
        while stack_ptr > 0:
            stack_ptr -= 1
            node = self.target_traversal_stack[i, stack_ptr]
            if node < 0 or node >= n_nodes:
                continue
            com = self.node_com[node]
            r_vec = target_pos - com
            r_sq = r_vec.dot(r_vec)
            r_mag = ti.sqrt(r_sq)
            node_size = 2.0 * self.node_half_size[node]
            if r_mag > 1e-8 and (node_size * node_size / r_sq) < theta_sq:
                sigma = self.node_avg_radius[node]
                r_sigma = r_mag / sigma
                if r_sigma > MIN_R_SIGMA:
                    q_val = self.q_kernel(r_sigma)
                    zeta_val = self.zeta_kernel(r_sigma) / (sigma * sigma * sigma)
                    total_circ = self.node_total_circ[node]
                    term1 = q_val / (r_mag * r_mag * r_mag)
                    term2 = 3.0 * q_val / (r_mag * r_mag * r_mag * r_mag * r_mag) - zeta_val / (
                        r_mag * r_mag
                    )
                    gradu += term1 * self.skew(total_circ) + term2 * r_vec.cross(
                        total_circ
                    ).outer_product(r_vec)
            elif self.node_is_leaf[node] == 1:
                gradu += self._target_leaf_gradient_sum(node, target_pos, MIN_R_SIGMA)
            else:
                stack_ptr = self._push_children_target(i, node, stack_ptr)
        return gradu

    # COMPUTE KERNELS

    @ti.kernel
    def compute_velocities_kernel(self, theta_sq: ti.f32):
        N = self.n_particles[None]
        n_nodes = self.n_nodes[None]
        for i in range(N):
            self.velocities[i] = (
                self._traverse_particle_vel(i, theta_sq, n_nodes) + self.u_inf[None]
            )

    @ti.kernel
    def compute_velocity_gradients_kernel(self, theta_sq: ti.f32):
        N = self.n_particles[None]
        n_nodes = self.n_nodes[None]
        for i in range(N):
            gradu = self._traverse_particle_grad(i, theta_sq, n_nodes)
            self.velocity_gradients[i] = gradu
            strain = ti.Matrix.zero(ti.f32, 3, 3)
            for p in ti.static(range(3)):
                for q in ti.static(range(3)):
                    strain[p, q] = 0.5 * (gradu[p, q] + gradu[q, p])
            self.strain_rates[i] = strain

    @ti.kernel
    def compute_velocity_and_gradient_kernel(self, theta_sq: ti.f32):
        """Fused single-traversal evaluation of u, ∇u and S.

        The solver needs **both** u (advection) and ∇u (stretching) at the same
        configuration every RK stage.  Walking the tree once and sharing the
        MAC/open decision, node geometry, and the leaf direct sums is strictly
        cheaper than two traversals.  Every per-branch term here is identical to
        ``_traverse_particle_vel`` + ``_traverse_particle_grad`` — velocity is
        ungated, the gradient keeps the MIN<r/σ<MAX gate — so the output is
        bit-identical to the two separate kernels.
        """
        N = self.n_particles[None]
        n_nodes = self.n_nodes[None]
        MIN_R_SIGMA = ti.cast(0.5, ti.f32)
        MAX_R_SIGMA = ti.cast(15.0, ti.f32)
        u_inf = self.u_inf[None]
        root = self._root[None]
        for i in range(N):
            vel = ti.Vector([0.0, 0.0, 0.0])
            gradu = ti.Matrix.zero(ti.f32, 3, 3)
            target_pos = self.positions[i]
            target_rad = self.radii[i]
            self.traversal_stack[i, 0] = root
            stack_ptr = 1
            while stack_ptr > 0:
                stack_ptr -= 1
                node = self.traversal_stack[i, stack_ptr]
                if node < 0 or node >= n_nodes:
                    continue
                com = self.node_com[node]
                r_vec = target_pos - com
                r_sq = r_vec.dot(r_vec)
                r_mag = ti.sqrt(r_sq)
                node_size = 2.0 * self.node_half_size[node]
                if r_mag > 1e-8 and (node_size * node_size / r_sq) < theta_sq:
                    # Far field: one node, both fields share sigma / r_sigma / q.
                    sigma = 0.5 * (target_rad + self.node_avg_radius[node])
                    r_sigma = r_mag / sigma
                    total_circ = self.node_total_circ[node]
                    q_val = self.q_kernel(r_sigma)
                    vel -= q_val * r_vec.cross(total_circ) / (r_mag * r_mag * r_mag)
                    if r_sigma > MIN_R_SIGMA and r_sigma < MAX_R_SIGMA:
                        zeta_val = self.zeta_kernel(r_sigma) / (sigma * sigma * sigma)
                        term1 = q_val / (r_mag * r_mag * r_mag)
                        term2 = 3.0 * q_val / (r_mag * r_mag * r_mag * r_mag * r_mag) - zeta_val / (
                            r_mag * r_mag
                        )
                        gradu += term1 * self.skew(total_circ) + term2 * r_vec.cross(
                            total_circ
                        ).outer_product(r_vec)
                elif self.node_is_leaf[node] == 1:
                    vel += self._leaf_velocity_sum(node, target_pos, target_rad, i)
                    gradu += self._leaf_gradient_sum(
                        node, target_pos, target_rad, i, MIN_R_SIGMA, MAX_R_SIGMA
                    )
                else:
                    stack_ptr = self._push_children_particle(i, node, stack_ptr)
            self.velocities[i] = vel + u_inf
            self.velocity_gradients[i] = gradu
            strain = ti.Matrix.zero(ti.f32, 3, 3)
            for p in ti.static(range(3)):
                for q in ti.static(range(3)):
                    strain[p, q] = 0.5 * (gradu[p, q] + gradu[q, p])
            self.strain_rates[i] = strain

    def compute_velocities_gpu(self, background_velocity: np.ndarray | None = None) -> None:
        """Run the velocity traversal on-device; the result stays in
        ``self.velocities`` (a Taichi field).  No ``to_numpy`` download — callers
        that keep the data on the GPU (e.g. ``base.velocity_self`` via a
        field-to-field copy) use this to avoid a per-step N×3 round-trip."""
        t_start = time.perf_counter()
        if background_velocity is not None:
            self.u_inf[None] = ti.Vector(background_velocity.astype(np.float32).tolist())
        else:
            self.u_inf[None] = ti.Vector([0.0, 0.0, 0.0])
        self.compute_velocities_kernel(self.theta * self.theta)
        ti.sync()
        self.eval_time = time.perf_counter() - t_start

    def compute_velocities(self, background_velocity: np.ndarray | None = None) -> np.ndarray:
        self.compute_velocities_gpu(background_velocity)
        N = self.n_particles[None]
        return self.velocities.to_numpy()[:N]

    def compute_velocity_gradients_gpu(self) -> None:
        """Run the velocity-gradient traversal on-device; results stay in
        ``self.velocity_gradients`` / ``self.strain_rates`` (Taichi fields)."""
        t_start = time.perf_counter()
        self.compute_velocity_gradients_kernel(self.theta * self.theta)
        ti.sync()
        self.grad_time = time.perf_counter() - t_start

    def compute_velocity_gradients(self) -> tuple:
        self.compute_velocity_gradients_gpu()
        N = self.n_particles[None]
        grads = self.velocity_gradients.to_numpy()[:N]
        strains = self.strain_rates.to_numpy()[:N]
        return grads, strains

    def compute_velocity_and_gradient_gpu(self, background_velocity: np.ndarray | None = None) -> None:
        """Fused on-device evaluation of u, ∇u and S in a *single* tree traversal.

        Results stay in ``self.velocities`` / ``self.velocity_gradients`` /
        ``self.strain_rates`` (Taichi fields).  Use this from the solver when both
        the advection velocity and the stretching gradient are needed at the same
        configuration in an RK stage — one build, one traversal, no download."""
        t_start = time.perf_counter()
        if background_velocity is not None:
            self.u_inf[None] = ti.Vector(background_velocity.astype(np.float32).tolist())
        else:
            self.u_inf[None] = ti.Vector([0.0, 0.0, 0.0])
        self.compute_velocity_and_gradient_kernel(self.theta * self.theta)
        ti.sync()
        self.eval_time = time.perf_counter() - t_start

    def compute_velocity_and_gradient(self, background_velocity: np.ndarray | None = None) -> tuple:
        self.compute_velocity_and_gradient_gpu(background_velocity)
        N = self.n_particles[None]
        return (
            self.velocities.to_numpy()[:N],
            self.velocity_gradients.to_numpy()[:N],
            self.strain_rates.to_numpy()[:N],
        )

    # TARGET POINT EVALUATIONS

    @ti.kernel
    def compute_target_velocities_kernel(self, theta_sq: ti.f32, avg_radius: ti.f32):
        M = self.n_targets[None]
        n_nodes = self.n_nodes[None]
        for i in range(M):
            self.target_velocities[i] = (
                self._traverse_target_vel(i, theta_sq, n_nodes) + self.u_inf[None]
            )

    @ti.kernel
    def compute_target_velocity_gradients_kernel(self, theta_sq: ti.f32, avg_radius: ti.f32):
        M = self.n_targets[None]
        n_nodes = self.n_nodes[None]
        for i in range(M):
            self.target_velocity_gradients[i] = self._traverse_target_grad(i, theta_sq, n_nodes)

    def compute_target_velocities(
        self, target_positions: np.ndarray, background_velocity: np.ndarray | None = None
    ) -> np.ndarray:
        M = len(target_positions)
        if M == 0:
            return np.zeros((0, 3), dtype=np.float32)
        if self.max_targets < M:
            raise ValueError(f"Too many targets: {M} > {self.max_targets}")
        self.n_targets[None] = M
        self.target_positions.from_numpy(target_positions.astype(np.float32))
        if background_velocity is not None:
            self.u_inf[None] = ti.Vector(background_velocity.astype(np.float32).tolist())
        else:
            self.u_inf[None] = ti.Vector([0.0, 0.0, 0.0])
        avg_radius = float(self.radii.to_numpy()[: self.n_particles[None]].mean())
        theta_sq = self.theta * self.theta
        self.compute_target_velocities_kernel(theta_sq, avg_radius)
        ti.sync()
        return self.target_velocities.to_numpy()[:M]

    def compute_target_velocity_gradients(self, target_positions: np.ndarray) -> np.ndarray:
        M = len(target_positions)
        if M == 0:
            return np.zeros((0, 3, 3), dtype=np.float32)
        if self.max_targets < M:
            raise ValueError(f"Too many targets: {M} > {self.max_targets}")
        self.n_targets[None] = M
        self.target_positions.from_numpy(target_positions.astype(np.float32))
        avg_radius = float(self.radii.to_numpy()[: self.n_particles[None]].mean())
        theta_sq = self.theta * self.theta
        self.compute_target_velocity_gradients_kernel(theta_sq, avg_radius)
        ti.sync()
        return self.target_velocity_gradients.to_numpy()[:M]

    # INFO

    def info(self) -> str:
        grad_info = f"\n  Grad time: {self.grad_time * 1000:.2f} ms" if self.grad_time > 0 else ""
        return (
            f"TaichiTreecode (GPU/LBVH):\n"
            f"  Particles: {self.n_particles[None]}\n"
            f"  Nodes: {self.n_nodes[None]}\n"
            f"  Opening angle θ: {self.theta}\n"
            f"  Build time: {self.build_time * 1000:.2f} ms\n"
            f"  Eval time: {self.eval_time * 1000:.2f} ms{grad_info}"
        )

# =========================================================
# Convenience function
# =========================================================

def compute_velocities_treecode_gpu(
    positions: np.ndarray,
    circulations: np.ndarray,
    radii: np.ndarray,
    theta: float = 0.5,
    background_velocity: np.ndarray | None = None,
) -> np.ndarray:
    N = len(positions)
    tree = TaichiTreecode(max_particles=N, max_nodes=2 * N, theta=theta)
    tree.build(positions, circulations, radii)
    return tree.compute_velocities(background_velocity)
