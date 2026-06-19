# Contributing to OpenONDA

Thank you for helping make OpenONDA a clean, efficient, and accessible CFD solver!
This guide explains how to set up your development environment, run code-quality
checks, and follow the project's conventions.

---

## 1. Getting Started

### 1.1 Install in development mode

```bash
git clone https://github.com/EngFlavioMartins/OpenONDA.git
cd OpenONDA
pip install -e ".[dev]"
```

This installs OpenONDA **editable** (changes to `source/` are reflected
immediately) plus all linting, type-checking, and test tools.

### 1.2 Optional: pre-commit hooks

```bash
pre-commit install        # runs checks automatically on every commit
pre-commit run --all-files  # run once manually over the whole repo
```

Pre-commit is **optional** — CI will run the same checks on every Pull Request.

---

## 2. Running Code-Quality Checks Locally

Use the helper script before pushing:

```bash
./scripts/code_quality.sh        # check only
./scripts/code_quality.sh --fix  # auto-fix where possible (ruff import sorting, etc.)
```

Or run tools individually:

| Tool | Purpose | Command | Blocking in CI? |
|------|---------|---------|-----------------|
| **ruff** | Lint + format | `ruff check source/` | Yes |
| **mypy** | Type checking | `mypy source/coupler/` | Yes (coupler only) |
| **pytest** | Tests | `TI_BACKEND=cpu pytest -m "not slow and not gpu"` | Yes |
| **vulture** | Dead code | `vulture source/` | No (informational) |
| **complexipy** | Complexity | `complexipy source/` | No (informational) |
| **tach** | Architecture boundaries | `tach check` | No (informational) |
| **interrogate** | Docstring coverage | `interrogate source/ --fail-under 85` | No (informational) |

### What to do when a tool complains

- **ruff** — most issues are auto-fixable: `ruff check --fix source/`
- **mypy** — if you see `Item "None" of ... has no attribute`, add a guard:
  ```python
  if x is None:
      return default
  # now mypy knows x is not None
  ```
- **vulture** — if it flags a symbol you actually need, add it to `ignore_names`
  in `pyproject.toml` under `[tool.vulture]`.

---

## 3. Architecture Overview

```
source/
├── solvers/
│   ├── VPM/          # Vortex Particle Method (Taichi, GPU/CPU)
│   ├── FVM/          # Finite Volume Method (pure Python)
│   └── OFW/          # OpenFOAM C++ wrapper (Cython bridge)
├── coupler/          # Hybrid FVM-VPM coupling
│   ├── config/       # Coupler configuration types
│   ├── core/         # Main solver orchestration
│   ├── diagnostics/  # Injection & conservation checks
│   └── to_review/    # Experimental coupling algorithms
└── version.py
```

### Import rules (enforced by `tach`)

- `VPM` may **not** import `FVM` internals directly.
- `FVM` may **not** import `VPM` internals directly.
- `coupler` is the **only** package allowed to import both solvers.
- `OFW` depends on `FVM` (it is the OpenFOAM bridge).

---

## 4. Taichi Best Practices

OpenONDA's VPM solver uses [Taichi](https://docs.taichi-lang.org/) for
high-performance kernels on GPU and CPU.

### 4.1 `ti.sync()` — when do you need it?

Taichi kernels are **asynchronous** on GPU. If you launch a kernel and
immediately read the result back in Python, use `ti.sync()` to be safe:

```python
# Good
my_kernel(field)
ti.sync()
result = field.to_numpy()

# Also acceptable (modern Taichi)
my_kernel(field)
result = field.to_numpy()   # implicit sync for this field
```

The main solver loops already contain `ti.sync()` at the correct boundaries.
If you add a new physics stage, mirror the existing pattern in
`source/solvers/VPM/core/solver.py`.

### 4.2 Avoid memory leaks

Taichi fields are **not garbage-collected**. Never create fields inside a loop:

```python
# BAD — leaks GPU memory
for step in range(1000):
    temp = ti.field(ti.f32, shape=n)   # leaks!
    my_kernel(temp)

# GOOD — reuse a cached field
self._temp_field = ti.field(ti.f32, shape=max_n)
for step in range(1000):
    my_kernel(self._temp_field)
```

See `source/solvers/VPM/physics/evaluation.py` for examples of cached result
fields.

### 4.3 Race conditions

Inside a `ti.kernel`, parallel `for` loops run on all threads simultaneously.

- **Safe**: each thread writes to a different index:
  ```python
  for i in range(n):
      result[i] = compute(i)   # OK
  ```

- **Unsafe without atomic**: multiple threads write to the same scalar:
  ```python
  total = 0.0
  for i in range(n):
      total += value[i]        # RACE CONDITION!
  ```

  Fix with `ti.atomic_add`:
  ```python
  total = 0.0
  for i in range(n):
      ti.atomic_add(total, value[i])   # OK
  ```

> **Note**: Taichi automatically promotes simple scalar `+=` reductions
> (e.g., `res += a[i] * b[i]`) to safe reductions, so you rarely need
> explicit `ti.atomic_add` for dot-products or sums.

---

## 5. Common Issues for New Contributors

### "mypy says my variable can be None"

Add an explicit guard or `assert`:

```python
# Before (mypy error)
vs = self.viscous_scheme
return vs.scheme

# After (clean)
vs = self.viscous_scheme
if vs is None:
    return {}
return {"scheme": vs.scheme}
```

### "ruff says imports are unsorted"

```bash
ruff check --fix source/
```

### "tach says I can't import from FVM in VPM"

Move shared code to `source/solvers/VPM/` (if VPM-specific) or create a
general utility. If you truly need cross-solver access, do it inside
the `coupler/` package.

### Tests fail with "no GPU"

All CI tests run on CPU. Locally:

```bash
TI_BACKEND=cpu pytest tests/vpm/ -m "not slow and not gpu"
```

---

## 6. C/C++ Code (OpenFOAM Wrapper)

The `source/solvers/OFW/` directory contains Cython and C++ code.
If you modify `foamSolverWrapper.cpp`, please format it with:

```bash
# Requires clang-format (install via your package manager)
clang-format -i source/solvers/OFW/foamSolverWrapper.cpp
```

Style is configured in `.clang-format` (LLVM-based, 4-space indent, 100-col limit).

---

## 7. Need Help?

- Open a [GitHub Discussion](https://github.com/EngFlavioMartins/OpenONDA/discussions)
- File an [Issue](https://github.com/EngFlavioMartins/OpenONDA/issues)
- Check the User Manual in `docs/User_Manual.md`
