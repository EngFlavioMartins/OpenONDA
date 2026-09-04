#!/usr/bin/env python3
"""Require complete documentation and type signatures on ``openonda`` exports."""

from __future__ import annotations

import importlib
import inspect
import types
import typing

PUBLIC_MODULES = (
    "openonda",
    "openonda.coupler",
    "openonda.fvm",
    "openonda.fvm.mesher",
    "openonda.runtime",
    "openonda.tutorials",
    "openonda.vpm",
)


def _signature_findings(
    value: object,
    *,
    drop_first: bool = False,
    check_return: bool = True,
) -> list[str]:
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError) as error:
        return [f"signature unavailable: {error}"]

    findings: list[str] = []
    parameters = list(signature.parameters.values())
    if drop_first and parameters and parameters[0].name in {"self", "cls"}:
        parameters = parameters[1:]
    for parameter in parameters:
        if parameter.annotation is inspect.Parameter.empty:
            findings.append(f"parameter {parameter.name} lacks annotation")
        elif parameter.annotation is typing.Any or str(parameter.annotation) in {
            "typing.Any",
            "Any",
        }:
            findings.append(f"parameter {parameter.name} uses Any")
    if check_return:
        if signature.return_annotation is inspect.Signature.empty:
            findings.append("return lacks annotation")
        elif signature.return_annotation is typing.Any or str(signature.return_annotation) in {
            "typing.Any",
            "Any",
        }:
            findings.append("return uses Any")
    return findings


def audit_public_api() -> list[str]:
    """Return actionable findings for every unique public exported symbol."""
    findings: list[str] = []
    seen: set[tuple[int, str]] = set()
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        for name in getattr(module, "__all__", ()):
            value = getattr(module, name)
            key = (id(value), name)
            if key in seen:
                continue
            seen.add(key)
            qualified_name = f"{module_name}.{name}"
            if inspect.isclass(value):
                issues = [] if inspect.getdoc(value) else ["class lacks docstring"]
                issues.extend(_signature_findings(value, check_return=False))
                if issues:
                    findings.append(f"class {qualified_name}: " + "; ".join(issues))
                for method_name, raw_member in value.__dict__.items():
                    if method_name.startswith("_"):
                        continue
                    member = (
                        raw_member.__func__
                        if isinstance(raw_member, staticmethod | classmethod)
                        else raw_member
                    )
                    if not isinstance(member, types.FunctionType | types.BuiltinFunctionType):
                        continue
                    method_issues = [] if inspect.getdoc(member) else ["method lacks docstring"]
                    method_issues.extend(_signature_findings(member, drop_first=True))
                    if method_issues:
                        findings.append(
                            f"method {qualified_name}.{method_name}: " + "; ".join(method_issues)
                        )
            elif inspect.isfunction(value):
                issues = [] if inspect.getdoc(value) else ["function lacks docstring"]
                issues.extend(_signature_findings(value))
                if issues:
                    findings.append(f"function {qualified_name}: " + "; ".join(issues))
    return findings


def main() -> int:
    """Run the public API gate and report all violations together."""
    findings = audit_public_api()
    if findings:
        print("Incomplete public API contracts:")
        print("\n".join(findings))
        return 1
    print("Public API completeness scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
