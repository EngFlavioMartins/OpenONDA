"""Static contract for immutable Taichi kernel arguments."""

from __future__ import annotations

import ast
from pathlib import Path


def _assigned_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Tuple | ast.List):
        names: set[str] = set()
        for element in target.elts:
            names.update(_assigned_names(element))
        return names
    return set()


def test_taichi_kernels_do_not_rebind_arguments():
    source_root = Path(__file__).resolve().parents[2] / "source" / "solvers" / "vpm"
    violations: list[str] = []

    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef):
                continue
            if not any(
                ast.unparse(decorator) == "ti.kernel" for decorator in function.decorator_list
            ):
                continue
            argument_names = {
                argument.arg
                for argument in (
                    *function.args.posonlyargs,
                    *function.args.args,
                    *function.args.kwonlyargs,
                )
                if argument.arg != "self"
            }
            for node in ast.walk(function):
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr):
                    targets = [node.target]
                else:
                    continue
                rebound = set().union(*(_assigned_names(target) for target in targets))
                for name in sorted(rebound & argument_names):
                    relative_path = path.relative_to(source_root.parent.parent.parent)
                    violations.append(
                        f"{relative_path}:{node.lineno}: {function.name} rebinds {name}"
                    )

    assert not violations, "\n".join(violations)
