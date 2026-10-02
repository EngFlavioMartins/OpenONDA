"""Pure fake qualification: no Taichi, solver, or MPI imports."""

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def asset():
    path = (
        Path(__file__).resolve().parents[2]
        / "tests/support/cylinder/capture_induction_reuse.py"
    )
    spec = importlib.util.spec_from_file_location("reuse_trace_asset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class Statistics:
    requests: int = 0
    exact_checks: int = 0
    hits: int = 0
    misses: int = 0
    bypasses: int = 0


def fixture(asset):
    calls = []
    cache = SimpleNamespace(
        _valid=True, _count=8, backend=object(), statistics=Statistics(),
    )
    key = ("operator", ("capacity", 8), ("table", b"a" * 5000))
    cache._key = id(cache.backend), asset._canonical(key)

    def provider():
        calls.append("contract")
        return SimpleNamespace(operator_key=key)

    cache.contract_provider = provider

    class RHS:
        def _induction_evaluator(self):
            return cache

        def evaluate_induction(self, state, time, rates):
            evaluator = self._induction_evaluator()
            evaluator.contract_provider()
            evaluator.statistics.requests += 1
            evaluator.statistics.exact_checks += 1
            evaluator.statistics.hits += 1
            if getattr(state, "fail", False):
                raise ValueError("expected")

    rhs = RHS()
    owner = SimpleNamespace(_is_master=True, vpm_solver=SimpleNamespace(stage_rhs=rhs))
    return owner, cache, provider, calls


def test_records_actual_single_contract_call_and_restores_existing_profiler(asset):
    owner, cache, provider, calls = fixture(asset)
    rhs, rows = owner.vpm_solver.stage_rhs, []
    original = rhs.evaluate_induction
    timed = []

    def profiler(*args):
        timed.append(1)
        return original(*args)

    rhs.evaluate_induction = profiler
    with asset.capture_induction_reuse_requests(owner, rows):
        rhs.evaluate_induction(SimpleNamespace(stage_index=0, count=8), 1.0, None)
    assert calls == ["contract"] and timed == [1]
    assert rhs.evaluate_induction is profiler and cache.contract_provider is provider
    assert "_induction_evaluator" not in vars(rhs)
    row = rows[0]
    assert row["operator_key_equal"] and row["previous_valid"]
    assert row["statistics_delta"]["hits"] == 1 and row["contract_calls"] == 1


def test_reports_bounded_named_differences_without_large_payload(asset):
    owner, cache, provider, _ = fixture(asset)
    cache._key = id(cache.backend), asset._canonical(
        ("operator", ("capacity", 4), ("table", b"b" * 5000))
    )
    rows = []
    with asset.capture_induction_reuse_requests(owner, rows):
        owner.vpm_solver.stage_rhs.evaluate_induction(
            SimpleNamespace(stage_index=None, count=8), 2.0, None
        )
    differences = rows[0]["operator_key_differences"]
    assert len(differences) == 2 and "capacity" in differences[0]["path"]
    assert differences[1]["before"]["length"] == 5000
    assert len(repr(rows)) < 2000 and cache.contract_provider is provider


def test_hook_restoration_and_record_on_failure(asset):
    owner, cache, provider, _ = fixture(asset)
    rows, rhs = [], owner.vpm_solver.stage_rhs
    with pytest.raises(ValueError, match="expected"), asset.capture_induction_reuse_requests(owner, rows):
        rhs.evaluate_induction(SimpleNamespace(stage_index=1, count=8, fail=True), 3.0, None)
    assert rows[0]["error"] == "ValueError: expected"
    assert cache.contract_provider is provider
    assert "evaluate_induction" not in vars(rhs) and "_induction_evaluator" not in vars(rhs)


def test_nonowner_noop_and_record_bound(asset):
    with asset.capture_induction_reuse_requests(SimpleNamespace(_is_master=False), []):
        pass
    owner, _, _, calls = fixture(asset)
    rows = []
    with asset.capture_induction_reuse_requests(owner, rows, max_records=1):
        for _ in range(3):
            owner.vpm_solver.stage_rhs.evaluate_induction(
                SimpleNamespace(stage_index=0, count=8), 1.0, None
            )
    assert len(rows) == 1 and len(calls) == 3
