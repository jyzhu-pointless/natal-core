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

For code changes, run and pass all of the following:
- pytest
- pyright
- ruff check src demos

Activate the repository virtual environment before running `pyright` so it uses
the project-installed dependencies and configuration.

After those commands complete, review `docs/` for any needed documentation updates and apply them before finalizing the change.

If API exports changed, also run:
- python scripts/generate_init_pyi.py

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
