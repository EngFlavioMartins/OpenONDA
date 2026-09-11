#!/usr/bin/env python3
"""Validate cube-flow plotting provenance and exact sample-time alignment."""

from __future__ import annotations

if not __package__:
    from pathlib import Path as _CasePath
    from openonda.tutorial_runner import case_package

    __package__ = case_package(_CasePath(__file__).resolve().parents[1]) + ".assets"


from . import _plotutil as util


def main() -> None:
    result = util.validate_plot_inputs()
    print(
        "Validated plotting inputs: "
        f"profiles through t={result['latest_profile_time']:.6g} s, "
        f"fields through t={result['latest_field_time']:.6g} s, "
        f"forces through t={result['latest_force_time']:.6g} s"
    )


if __name__ == "__main__":
    main()
