"""Tests for the best-effort Numba disk-cache decorator."""

from __future__ import annotations

import pytest

from source import _numba


def test_cacheable_njit_falls_back_only_when_no_cache_locator_exists(monkeypatch):
    calls = []

    def fake_njit(*args, **kwargs):
        calls.append((args, kwargs))

        def decorate(function):
            if kwargs.get("cache"):
                raise RuntimeError("cannot cache function: no locator available")
            return ("compiled", function)

        return decorate

    monkeypatch.setattr(_numba, "_numba_njit", fake_njit)

    def kernel():
        return None

    compiled = _numba.cacheable_njit(cache=True, fastmath=False)(kernel)

    assert compiled == ("compiled", kernel)
    assert [call[1]["cache"] for call in calls] == [True, False]
    assert all(call[1]["fastmath"] is False for call in calls)


def test_cacheable_njit_preserves_other_numba_errors(monkeypatch):
    def fake_njit(*args, **kwargs):
        def decorate(function):
            raise RuntimeError("unrelated compilation failure")

        return decorate

    monkeypatch.setattr(_numba, "_numba_njit", fake_njit)

    with pytest.raises(RuntimeError, match="unrelated compilation failure"):

        @_numba.cacheable_njit(cache=True)
        def kernel():
            return None
