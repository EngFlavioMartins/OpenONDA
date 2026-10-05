"""Gaussian query outputs copy to the host without default-pool staging."""

import numpy as np
import pytest

from source.solvers.vpm.physics.induction.gaussian_mesh.coordinates import slab_coordinates
from source.solvers.vpm.physics.induction.gaussian_mesh.fields import GaussianImageFields
from source.solvers.vpm.physics.induction.gaussian_mesh.stencil import cardinal_stencil_gpu
from tests.vpm.test_gaussian_mesh_package import _case

pytestmark = pytest.mark.gpu

# The previous interleaved gather is an independent layout/parity oracle.
_LEGACY_GATHER = r'''
typedef REAL real;
extern "C" __global__ void gather(
    const int count,const int order,const int *first,const real *weights,
    const int ox,const int oy,const int oz,const int fy,const int fz,
    const real *field,const int component,real *output) {
  int target=blockIdx.x*blockDim.x+threadIdx.x;
  if(target>=count) return;
  const int *base=first+3*target;
  const real *w=weights+3*order*target;
  double value=0;
  for(int a=0;a<order;++a) for(int b=0;b<order;++b) for(int c=0;c<order;++c) {
    long long index=((long long)(base[0]+a-ox)*fy+base[1]+b-oy)*fz+base[2]+c-oz;
    value+=(double)w[a]*(double)w[order+b]*(double)w[2*order+c]*(double)field[index];
  }
  output[12*(long long)target+component]=(real)value;
}
'''


def _legacy_result(owner, query):
    cp = owner.cp
    lattice, _, _ = slab_coordinates(query, owner.zmin, owner.zmax, owner.spacing)
    kernel = cp.RawKernel(
        _LEGACY_GATHER.replace("REAL", "float" if owner.dtype.itemsize == 4 else "double"),
        "gather", options=("--std=c++11",), backend="nvrtc",
    )
    with owner._owner.allocation_scope():
        first, weight, _ = cardinal_stencil_gpu(
            lattice, owner.origin, owner.order, owner.shape, dtype=owner.dtype,
            pool=owner.pool, max_points=len(query), max_new_bytes=owner.max_scratch_bytes,
        )
        output = cp.empty((len(query), 12), dtype=owner.dtype)
        for column in range(12):
            kernel(((len(query) + 255) // 256,), (256,), (
                np.int32(len(query)), np.int32(owner.order), first, weight,
                *(np.int32(n) for n in owner.compact_start),
                np.int32(owner.compact_shape[1]), np.int32(owner.compact_shape[2]),
                owner._compact_fields[column], np.int32(column), output,
            ))
        u, j, _ = owner._correction.evaluate(query, owner._prepared_world_images)
        output[:, :3] += u
        output[:, 3:] += j.reshape(-1, 9)
        # Copy the contiguous old allocation rather than its strided views.
        return cp.asnumpy(output)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_contiguous_split_outputs_match_prior_gather_with_partial_batches(monkeypatch, dtype):
    cp = pytest.importorskip("cupy")
    from source.solvers.vpm.physics.induction.gaussian_mesh import fields

    x, gamma, sigma, q, images = _case()
    query = np.tile(q, (11, 1))
    with GaussianImageFields(x, gamma, sigma, q, zmin=0., zmax=.193,
                             tau=.12, spacing=.035, cutoff=.6,
                             dtype=dtype, correction_dtype=dtype) as owner:
        owner.prepare(images)
        expected = _legacy_result(owner, query)
        assert np.any(expected[:, :3]) and np.any(expected[:, 3:])
        monkeypatch.setattr(fields, "query_batch_size", lambda *args, **kwargs: 3)
        pool = cp.get_default_memory_pool()
        default_state = pool.used_bytes(), pool.total_bytes(), pool.get_limit()
        for count in (0, 1, 7, len(query)):
            u, j, report = owner.evaluate_prepared(query[:count])
            assert u.flags.c_contiguous and j.flags.c_contiguous
            assert u.dtype == j.dtype == np.dtype(dtype)
            assert u.shape == (count, 3) and j.shape == (count, 3, 3)
            assert u.nbytes + j.nbytes == count * 12 * np.dtype(dtype).itemsize
            if count:
                assert u.data.mem is j.data.mem
                assert j.data.ptr - u.data.ptr == u.nbytes
            assert report["query_batches"] == (count + 2) // 3
            np.testing.assert_array_equal(cp.asnumpy(u), expected[:count, :3])
            np.testing.assert_array_equal(cp.asnumpy(j), expected[:count, 3:].reshape(count, 3, 3))
            assert (pool.used_bytes(), pool.total_bytes(), pool.get_limit()) == default_state
            del u, j


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_growing_host_transfers_do_not_retain_default_pool_blocks(dtype):
    cp = pytest.importorskip("cupy")
    x, gamma, sigma, q, images = _case()
    pool = cp.get_default_memory_pool()
    allocator = cp.cuda.get_allocator()
    default_state = pool.used_bytes(), pool.total_bytes(), pool.get_limit()
    with GaussianImageFields(x, gamma, sigma, q, zmin=0., zmax=.193,
                             tau=.12, spacing=.035, cutoff=.6, order=4,
                             dtype=dtype, correction_dtype=dtype,
                             max_scratch_bytes=64 * 1024**2,
                             max_correction_bytes=16 * 1024**2,
                             max_plan_bytes=8 * 1024**2,
                             max_total_bytes=80 * 1024**2) as owner:
        owner.prepare(images)
        u, j, _ = owner.evaluate_prepared(q)
        expected = cp.asnumpy(u), cp.asnumpy(j)
        del u, j
        for count in (65537, 65553, 65581):
            query = np.resize(q, (count, 3))
            u, j, _ = owner.evaluate_prepared(query)
            assert u.flags.c_contiguous and j.flags.c_contiguous
            for got, prior in zip((cp.asnumpy(u), cp.asnumpy(j)), expected, strict=True):
                np.testing.assert_array_equal(got, np.resize(prior, got.shape))
            assert cp.cuda.get_allocator() is allocator
            assert (pool.used_bytes(), pool.total_bytes(), pool.get_limit()) == default_state
            del u, j
    assert (pool.used_bytes(), pool.total_bytes(), pool.get_limit()) == default_state
