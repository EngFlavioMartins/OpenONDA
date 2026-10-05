"""Read-only, benchmark-only attribution of exact-induction reuse decisions.

Wrap the actual reuse_conditions-provider invocation, not a numerical backend and not
an extra operator-key evaluation. No source/device field is read by this hook.
"""

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import struct


def _comparison_value(value):
    if type(value) is tuple:
        return "tuple", tuple(_comparison_value(item) for item in value)
    if type(value) is float:
        return "float64", struct.pack("!d", value)
    return type(value).__name__, value


def _summary(value):
    kind, item = value
    if kind == "tuple":
        return {"type": kind, "length": len(item)}
    if kind == "bytes":
        return {"type": kind, "length": len(item), "sha256": hashlib.sha256(item).hexdigest()}
    if kind == "float64":
        return {"type": kind, "value": struct.unpack("!d", item)[0], "bits": item.hex()}
    if kind == "str" and len(item) > 160:
        return {"type": kind, "length": len(item), "prefix": item[:160]}
    return {"type": kind, "value": item}


def _differences(left, right, *, limit=8):
    result = []

    def visit(a, b, path):
        if len(result) >= limit or a == b:
            return
        if a[0] == b[0] == "tuple" and len(a[1]) == len(b[1]):
            for index, (old, new) in enumerate(zip(a[1], b[1], strict=True)):
                label = f"[{index}]"
                if old[0] == new[0] == "tuple" and old[1] and new[1]:
                    first = old[1][0]
                    if first == new[1][0] and first[0] == "str":
                        label += "." + first[1][:80]
                visit(old, new, path + label)
        else:
            result.append({"path": path, "before": _summary(a), "after": _summary(b)})

    visit(left, right, "operator")
    return result


@contextmanager
def capture_induction_reuse_requests(coupler, records, *, max_records=128):
    """Append compact per-request records, restoring every Python hook on exit.

    Nest either inside or outside profile_components; restoration respects the
    previous instance binding. Only the VPM owner records, and the numerical
    backend's methods remain untouched so supported-backend validation is not affected.
    """
    if not coupler._is_master:
        yield
        return
    rhs = coupler.vpm_solver.stage_rhs
    names = ("evaluate_induction", "_induction_evaluator")
    original = {name: getattr(rhs, name) for name in names}
    instance = {name: (name in vars(rhs), vars(rhs).get(name)) for name in names}
    providers, active = [], []

    def evaluator():
        cache = original["_induction_evaluator"]()
        if not hasattr(cache, "conditions_provider"):
            return cache
        if active:
            active[-1]["cache"] = cache
            active[-1]["before"] = asdict(cache.statistics)
        if any(owner is cache for owner, _, _ in providers):
            return cache
        provider = cache.conditions_provider

        def forwarded():
            reuse_conditions = provider() if provider is not None else None
            if active:
                row = active[-1]["row"]
                row["evaluation_calls"] += 1
                row.update(
                    previous_valid=bool(cache._valid),
                    previous_count=int(cache._count),
                    conditions_present=reuse_conditions is not None,
                )
                if (
                    reuse_conditions is not None
                    and getattr(reuse_conditions, "operator_key", None) is not None
                ):
                    old = cache._key
                    current = id(cache.backend), _comparison_value(reuse_conditions.operator_key)
                    row["operator_key_equal"] = old == current
                    row["same_backend_settings"] = old is not None and old[0] == current[0]
                    row["operator_key_differences"] = (
                        [] if old is None else _differences(old[1], current[1])
                    )
            return reuse_conditions

        cache.conditions_provider = forwarded
        providers.append((cache, provider, forwarded))
        return cache

    def evaluate(stage_state, stage_time, stage_rates):
        row = {
            "request": len(records) + 1,
            "stage_index": stage_state.stage_index,
            "stage_time": float(stage_time),
            "count": int(stage_state.count),
            "evaluation_calls": 0,
        }
        frame = {"row": row}
        active.append(frame)
        try:
            return original["evaluate_induction"](stage_state, stage_time, stage_rates)
        except BaseException as error:
            row["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            active.pop()
            if "cache" in frame:
                after = asdict(frame["cache"].statistics)
                row["statistics_delta"] = {
                    key: after[key] - frame["before"][key]
                    for key in ("requests", "exact_checks", "hits", "misses", "bypasses")
                }
            if len(records) < max_records:
                records.append(row)

    rhs._induction_evaluator = evaluator
    rhs.evaluate_induction = evaluate
    try:
        yield
    finally:
        for owner, provider, proxy in reversed(providers):
            if owner.conditions_provider is proxy:
                owner.conditions_provider = provider
        for name, (existed, value) in instance.items():
            if existed:
                setattr(rhs, name, value)
            else:
                delattr(rhs, name)


__all__ = ["capture_induction_reuse_requests"]
