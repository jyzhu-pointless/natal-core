# AGENTS.md

> This is the English version. See also: [Chinese version](./AGENTS.md)
>
> When either version is updated, the other must be updated synchronously.

## Language

Use Chinese by default. Only reply in English when the user explicitly asks in English.

**All English in code and documentation must use American spelling**: identifiers, docstrings, comments, commit messages all follow this. Common conversions: `-ise/-isation` → `-ize/-ization` (initialize/materialize/normalize), `-our` → `-or` (behavior/color), `-re` → `-er` (center), `modelling` → `modeling`. Note exceptions: words that legitimately end with `-ise` in American English (comprise, exercise, raise, noise, wise, etc.) are left as is.

## Specification References

The following files define this project's coding, documentation, and testing standards, in order of priority:

1. `docstring_spec.md`
2. `quality_checks_spec.md`
3. `docstring_spec_cn.md` (Chinese explanation)
4. `quality_checks_spec_cn.md` (Chinese explanation)

In case of conflict, the English versions take precedence.

## Behavioral Guidelines

- Any proposal, plan, or non‑trivial modification must first be explained to the user and approved before execution. Do not implement without permission.
- Prefer writing comments. Comments should explain WHY (design intent, constraints, non‑obvious logic), not WHAT (the code itself already shows that).
- Do not create documentation files (*.md) unless explicitly requested by the user.
- Do not over‑abstract. Do not design for hypothetical needs.
- After modifications, provide a detailed written explanation of what you changed, why, and the resulting effect. Avoid vague statements like "I changed this function"; instead be specific, e.g., "I changed the `foo` function's parameter from `x` to `y` to support the new use case."
- **【Important!】In written explanations, use plain, accessible language as much as possible. If you introduce specialized software engineering terms (other than concepts already established in this project's architecture), you must explain them in detail.** Examples:
  - Not: "The preset parameter is managed by the Configurator's deferred mechanism and is also modified by the Configurator at runtime."
  - But: "Genetic preset parameters are special—they cannot be written directly into the configuration during construction; they must wait until the Population object is actually created. So the Configurator temporarily stores them (deferred) and applies them all at once when `build()` is executed. Runtime modifications to preset parameters also go through the Configurator's `update()` entry point."
- **Prefer specialized tools**: Read/Glob/Grep/Edit/Write are more reliable than shell commands (they are not blocked by the sandbox, have stable output format, and are more user‑friendly). Use Bash only for batch operations, pipe combinations, or when specialized tools cannot accomplish the task.
- **Do not commit / push proactively**: Do not execute `git commit`, `git push`, or any form of commit operation unless explicitly requested by the user.
- **Do not modify `.gitignore`**: Do not change `.gitignore` unless explicitly requested.

## Gate Checks

After every change, the following commands must be run:

```bash
pytest                          # Run all tests
pyright                         # Type check (strict mode)
ruff check src demos             # Lint check
ruff check src demos --fix       # Lint auto‑fix
python scripts/check_rust.py    # Rust: fmt + clippy + check (rust‑analyzer optional diagnostics)
python scripts/generate_init_pyi.py  # Regenerate stubs after public API changes
```

The virtual environment is already activated, so you can run the commands directly.

Before committing, you must pass the **Python triple gate and the Rust hard gate**: `pytest` + `pyright` + `ruff check src demos`, and `python scripts/check_rust.py` (where `cargo fmt --check`, `cargo clippy -- -D warnings`, and `cargo check --all-targets` must all succeed; rust‑analyzer diagnostics are optional and do not block). Do not suppress or bypass them.

### Review Process

After each code modification, the following sequence must be executed in order. **Do not substitute self‑review for these steps**:

1. **`@tester`** — Generate rigorous tests for new or changed code according to the `numerical-verification` and `adversarial-review` standards. Must cover five test categories:
   - **Negative contract tests** (assert that deleted interfaces are inaccessible)
   - **Ownership tests** (assert returned ndarray/list/dict are copies or read‑only)
   - **State transition tests** (restore→run, finish→snapshot, import→run, clear→record)
   - **Axis combination tests** (Cartesian enumeration of configuration axes)
   - **Error path tests** (invalid inputs raise the correct exception and leave state unchanged)
   Each assertion must prove a numerical invariant.
2. **Run `python scripts/generate_init_pyi.py`** — Regenerate stubs after public API changes so subsequent `pyright` checks use the latest stubs.
3. **`@evaluator`** — Perform an **adversarial** review. The reviewer takes an attacker's stance, actively seeking evidence that the code does not conform to the spec, rather than merely verifying that gates pass. Must:
   - Build a **ledger** from the spec: three lists: must‑exist, must‑not‑exist, and invariants
   - For every must‑not‑exist item, do a **full‑repo search** (src/tests/demos/docs/stub) to confirm it is inaccessible
   - For every invariant, execute **attacks**: ownership attacks (check ndarray references/write‑protection), state‑machine attacks (restore→run, finish→snapshot), axis‑combination attacks (Cartesian enumeration)
   - **Actually run** all demo scripts and documented code examples
   - Run `pytest` / `pyright` / `ruff` / `python scripts/check_rust.py` gates, loading the `code‑review`, `numerical‑verification`, and `adversarial‑review` skills
   - Decision rule (mechanical): `APPROVED` ⇔ zero hard‑blockers
4. **If the change involves public API (signatures, parameters, defaults, module renames, etc.), the main agent invokes `@docs`** to synchronize documentation and example code in `docs/zh/` and `docs/en/`.
5. **The main agent must not run `pytest`, `pyright`, `ruff`, `python scripts/check_rust.py` themselves and claim "review passed."** The results of these commands must be independently verified by the evaluator and reported in a structured report.

A modification is considered complete only after the evaluator gives an `APPROVED` verdict.

#### Mandatory Rejection Criteria for Evaluator

When any of the following occurs, the evaluator **must** return `REJECTED`. **Hard‑blockers are not softened by severity** – one missing annotation for `Any` is treated the same as a semantic break.

- **Test failures**: `pytest` shows any FAILED.
- **Rust hard gate failure**: `cargo fmt --check`, `cargo clippy -- -D warnings`, or `cargo check --all-targets` returns non‑zero (executed via `python scripts/check_rust.py`). rust‑analyzer optional diagnostics are not a reason for rejection.
- **Insufficient coverage**: line coverage for new modules or new code in existing modules is **< 95%**.
- **Hard‑blocker exists**: any one of the following categories triggers REJECTED:
  - **Must‑not‑exist violation**: interfaces that the spec requires to be deleted are still accessible (including via `getattr`, `__init__` re‑exports)
  - **Invariant break**: ownership leak (ndarray references / non‑write‑protected), state‑machine error (tick out of sync after restore), axis‑combination crash
  - **Demo/doc crash**: demo scripts or documented code examples actually error out when run
  - **Unannotated `Any` / `object`**
  - **Unannotated `# type: ignore`**
  - **`cast(Any, …)`**
  - **Non‑Google‑style docstring section** or **missing type annotation for parameter / return / attribute**
- **Missing negative contract tests**: no `assert‑not‑exists` tests for interfaces marked as deleted in the spec.
- **Missing ownership tests**: public methods that return containers (ndarray/list/dict) lack read‑only / copy verification in tests.
- **State transitions not covered**: critical lifecycle sequences (restore→run, finish→snapshot, import→run, clear→record) are not tested.

### Fix Strategy

- **Modified files**: all pyright / ruff / pytest / cargo fmt / cargo clippy / cargo check errors must be fixed.
- **Files affected by the change**: errors in other files caused by signature or import changes must also be fixed.
- **Pre‑existing issues in untouched files**: point them out and analyze; fixing is recommended but not required in the current commit.
- **`cast(Any, …)` is forbidden**. Do not use it to bypass type checking. Its presence triggers REJECTED.
- **Do not abuse `Any` or `object`**: parameter, return, and variable type annotations must point to concrete types; do not lazily use `Any` or `object`. Adding imports for the needed types is worthwhile. `Any` is only acceptable when there is a specific, documented reason (e.g., `Callable[..., Any]` for "any callable").
- **`cast(T, x)`** is allowed only when static analysis cannot possibly prove `x: T` (e.g., narrowing an `Optional` after a guard). Prefer type‑narrowing assertions or refactoring.
- **`# type: ignore`** is a last resort. Every ignore must be accompanied by a short comment. Missing comment triggers REJECTED.

### Test Coverage

- **New modules**: ≥95% line coverage.
- **New code in existing modules**: ≥95% line coverage.
- **Deterministic simulations** (`stochastic=False`): exact numeric assertions.
- **Stochastic simulations**: require statistical validation (multiple runs, confidence intervals, or distribution tests). A single run that passes is not sufficient.
- Prefer using pytest‑collected tests over script‑based smoke tests.

### Docstring Specification

- Use only Google‑style sections (`Args:`, `Returns:`, `Raises:`, etc.). Do not invent new section names.
- Docstring content must be in **English**.
- All parameters, returns, and attributes must be explicitly type‑annotated (prefer using annotations).

### Change Description

After every modification, include the following four items:
1. Changed files
2. Behavioral changes
3. Verification commands executed
4. Residual risks or follow‑up items (if any)
