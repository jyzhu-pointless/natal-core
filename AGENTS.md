# NATAL Core Agent Rules

This file defines default instructions for AI Agents working in this repository.
These rules are always-on at repository scope.

## 1) Source of truth

Follow these documents in descending priority:
1. docstring_spec.md
2. quality_checks_spec.md
3. docstring_spec_cn.md (Chinese explanation)
4. quality_checks_spec_cn.md (Chinese explanation)

If rules conflict, prefer the English specification files.

## 2) Mandatory quality gates before proposing completion

The virtual environment is auto-activated — run commands directly, do not prepend `source .venv/bin/activate`.

```bash
pytest
pyright
ruff check src demos
```

After those commands complete, review `docs/` for any needed documentation updates.

If API exports changed, also run:
- python scripts/generate_init_pyi.py

### Fix-everything policy

- **Modified files**: Fix ALL pyright / ruff / pytest failures, regardless of whether they pre-existed.
- **Other files affected by the change** (e.g., signature or import changes): must be fixed too.
- **Pre-existing issues in untouched files**: explicitly note and analyse them. Fixing is encouraged but not strictly required for the current commit.
- **`cast(Any, …)` is forbidden**. Never bypass type checking this way.
- **`Any` in function parameter lists is forbidden** unless accompanied by a concrete, documented justification.
- **`cast(T, x)`** may be used only when static analysis cannot prove `x: T` at all (e.g., narrowing `Optional` after an explicit guard) and the error is otherwise unavoidable. Prefer type-narrowing assertions or restructuring before `cast`.
- **`# type: ignore`** is a last resort. Every ignore must include a short, specific reason on the same line.

### Test coverage

- **New modules**: ≥95% line coverage.
- **New code in existing modules**: ≥95% line coverage.
- **Deterministic simulations** (`stochastic=False`): exact numerical assertions on counts, frequencies, or derived statistics.
- **Stochastic simulations**: statistical validation — multiple runs with confidence intervals or distributional checks. A single passing run is insufficient.

## 3) Docstring requirements

- Use Google style sections only.
- Keep docstring text in English.
- Ensure all parameters/returns/attributes are explicitly typed (annotation preferred).
- Do not invent section headers.

## 4) Testing requirements

- Add or update tests for all non-trivial behavior changes.
- Cover happy path, boundaries, and error paths when relevant.
- Keep tests deterministic (fixed seeds for random behavior).
- Prefer pytest-collected tests over script-style smoke tests.

## 5) Typing and linting requirements

- Do not introduce new implicit Any in modified code.
- Do not add # type: ignore unless absolutely necessary with a clear reason.
- Do not add broad Ruff ignores; keep exceptions minimal and documented.

## 6) Change note requirements (human and AI)

Every change summary must include:
- Files changed
- Behavior changes
- Validation commands executed
- Residual risks or follow-up items (if any)

## 7) Working mode for this repository

- Repository is mainly solo-maintained with occasional external contributions.
- Keep changes small and focused.
- Lightweight review is acceptable, but quality gates are never optional.
