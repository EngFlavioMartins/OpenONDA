"""Every-request admission runs before hits and invalidates on failures."""

from dataclasses import replace

import numpy as np
import pytest
import taichi as ti

from tests.vpm.test_induction_exact_reuse import Harness


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    owned = ti.lang.impl.get_runtime().prog is None
    if owned:
        ti.init(arch=ti.cpu, cpu_max_num_threads=2, offline_cache=False)
    yield
    if owned:
        ti.reset()


def test_admission_is_called_for_miss_hit_subsets_and_changed_sources():
    h = Harness()
    calls = []
    def admit(**args):
        calls.append((args["stage_time"], args["count"], args["strength_rate_enabled"],
                      args["velocity_gradient_out"]))
    h.reuse.contract_provider = lambda: replace(h.backend.contract(), request_admission=admit)
    try:
        h.run(time=1)
        h.run(time=2, rate=False, gradient=False)
        np.testing.assert_array_equal(h.rate.to_numpy()[:h.count], 0)
        h.radius[0] = .9
        h.run(time=3)
        assert [item[:3] for item in calls] == [(1, 4, True), (2, 4, False), (3, 4, True)]
        assert calls[1][3] is None
        assert h.reuse.statistics.hits == 1 and h.backend.calls == 2
    finally:
        h.reuse.close()


@pytest.mark.parametrize("failure", ["provider", "admission"])
def test_failed_preflight_revokes_existing_result_and_preserves_caller_outputs(failure):
    h = Harness()
    broken = False
    def fail():
        raise RuntimeError("injected preflight")
    def provider():
        if broken and failure == "provider":
            fail()
        return replace(h.backend.contract(), request_admission=lambda **args: fail() if broken else None)
    h.reuse.contract_provider = provider
    try:
        h.run()
        assert h.reuse._valid
        h.velocity.fill(71)
        h.rate.fill(73)
        h.gradient.fill(79)
        broken = True
        with pytest.raises(RuntimeError, match="preflight"):
            h.run()
        assert not h.reuse._valid
        assert h.reuse.statistics.hits == 0
        for output, sentinel in ((h.velocity, 71), (h.rate, 73), (h.gradient, 79)):
            np.testing.assert_array_equal(output.to_numpy(), sentinel)
        broken = False
        h.run()
        assert h.backend.calls == 2
    finally:
        h.reuse.close()


def test_explicit_request_decline_preserves_original_arguments_and_invalidates():
    h = Harness()
    def admission(**args):
        return None if type(args["strength_rate_enabled"]) is bool else False
    h.reuse.contract_provider = lambda: replace(h.backend.contract(), request_admission=admission)
    try:
        h.run()
        assert h.reuse._valid
        h.run(rate=2)
        assert h.backend.seen_flags[-1] == (2, True)
        assert not h.reuse._valid
        assert h.reuse.statistics.bypasses == 1 and h.reuse.statistics.hits == 0
        h.run()
        assert h.backend.calls == 3
    finally:
        h.reuse.close()
