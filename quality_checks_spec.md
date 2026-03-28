# Quality Checks Specification (Unit Tests, Pyright, Ruff)

This specification complements `docstring_spec.md` and defines the mandatory quality checks for code changes.
It standardizes unit testing, static type checking, and linting so that changes are safe, consistent, and reviewable.

---

## 1. Scope

This specification applies to all Python code changes in this repository, especially under `src/` and `demos/`.
It applies equally to human contributors and AI Agents.

Required quality gates:
- Unit tests (`pytest`)
- Type checking (`pyright` in strict mode)
- Lint checks (`ruff`)

---

## 1.1 Solo Development and Lightweight Collaboration

This repository is primarily maintained by a single developer with occasional external contributions.
The workflow is intentionally lightweight, but quality gates remain strict:
- Review steps may be lightweight, but quality checks are never optional.
- Even without external reviewers, run all commands in Section 2 before merge.
- Keep changes small and focused to reduce rollback risk.
- Each change should include a short validation summary (tests, typing, lint).

---

## 1.2 AI Agent Applicability

AI Agent contributions must meet the exact same quality bar as human contributions, plus traceability:
- Include changed-file scope, behavior summary, and executed validation commands.
- No style-only changes without validation.
- No new suppression (`# type: ignore`, Ruff ignores) without explicit rationale.
- If API behavior or contracts change, update tests and docs in the same change.

---

## 2. Required Commands

Run the following commands before creating or merging a change:

```bash
pytest
pyright
ruff check src demos
```

Recommended autofix command for lint issues:

```bash
ruff check src demos --fix
```

If a change affects packaging or import exposure, also regenerate stubs:

```bash
python scripts/generate_init_pyi.py
```

---

## 3. Unit Testing Rules (pytest)

### 3.1 Test placement and naming

- Place tests under `tests/`.
- Test file names must follow `test_*.py`.
- Test function names should describe behavior, e.g., `test_builder_rejects_invalid_age_structure`.

### 3.2 Test design requirements

- Prefer behavior-driven tests that validate externally observable outcomes.
- Keep tests deterministic; avoid time-dependent or random behavior without fixed seeds.
- Use the Arrange-Act-Assert structure for readability.
- Prefer one behavioral assertion group per test.

### 3.3 Fixtures and shared setup

- Reuse fixtures in `tests/conftest.py` instead of duplicating setup logic.
- Keep fixtures small and focused.
- Do not hide critical test assumptions inside deep fixture chains.

### 3.4 What to test for each change

Every non-trivial code change must add or update tests that cover:
- Happy path behavior
- Boundary conditions
- Invalid inputs and error paths (when relevant)
- Regression scenario for the bug being fixed (if applicable)

### 3.5 Prohibited testing practices

- Do not skip tests without a clear reason.
- Do not leave temporary debug assertions or print-based checks.
- Do not merge a feature with only manual verification when automated tests are feasible.

---

## 4. Type Checking Rules (Pyright)

### 4.1 Strict mode requirement

Pyright must pass with repository configuration:
- `typeCheckingMode = "strict"`
- `include = ["src"]`
- `ignore = ["tests/**", "**/__pycache__", "**/*.pyc"]`

### 4.2 Typing requirements

- Add explicit type annotations for public APIs, function signatures, and important internal values.
- Avoid implicit `Any` in new or modified code.
- Use precise types (`Sequence[int]`, `Mapping[str, float]`, `NDArray[...]`, etc.) instead of overly broad types where practical.

### 4.3 Handling type errors

- Prefer fixing root causes over local suppression.
- Use `cast(...)` only when a runtime invariant is known and cannot be expressed directly.
- `# type: ignore` is allowed only as a last resort and must include a short, specific reason.

### 4.4 Backward compatibility and typing

- Keep runtime behavior unchanged when introducing type-only refactors.
- If a typing change tightens API contracts, update tests and docs in the same change.

---

## 5. Lint Rules (Ruff)

### 5.1 Baseline configuration

Ruff checks follow repository configuration in `pyproject.toml`, including:
- `line-length = 88`
- Rule families: `E`, `W`, `F`, `I`, `B`, `C4`, `UP`
- Configured exceptions in `ignore` and `per-file-ignores`

### 5.2 Scope and execution

- Lint checks must pass for changed files under `src/` and `demos/`.
- Keep imports sorted according to Ruff/isort rules.
- Resolve lint findings by improving code clarity, not by broad suppression.

### 5.3 Allowed exceptions

- Respect repository-level exceptions that are already justified in `pyproject.toml`.
- Do not add new ignores unless they are narrowly scoped and documented in the change.

### 5.4 Prohibited lint handling

- Do not disable entire rule categories for convenience.
- Do not keep dead code, unreachable branches, or unused imports in committed changes.

---

## 6. Pull Request Quality Checklist

A change is ready to merge only when all of the following are true:
- Unit tests pass (`pytest`).
- Type checks pass (`pyright`).
- Lint checks pass (`ruff check src demos`).
- Added or modified behavior is covered by tests.
- New public APIs are typed and documented consistently with `docstring_spec.md`.
- For AI-generated changes, the change note includes summary, validation results, and residual risks (if any).

---

## 7. Quick Triage Guidance

If checks fail, resolve in this order:
1. Fix correctness and failing tests first.
2. Fix type issues that indicate real contract mismatches.
3. Fix lint issues and import/order cleanup.
4. Re-run the full command set before submission.

This order reduces rework and avoids masking behavioral regressions with style-only edits.

---

## 8. Minimum Delivery Flow (Human and AI Agent)

Follow this minimum flow for every change:
1. Implement one focused change objective.
2. Run `pytest`, `pyright`, and `ruff check src demos`.
3. If any check fails, fix in the order defined in Section 7 and rerun all checks.
4. Update related docs and docstrings for public API changes.
5. Record in the change note:
	- What changed.
	- How it was validated.
	- Known limitations or follow-up items (if any).
