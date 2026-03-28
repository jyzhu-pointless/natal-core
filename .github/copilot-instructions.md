# Copilot Workspace Instructions for NATAL Core

Apply these rules for all tasks in this repository.

## Rule Priority

Use this priority order when rules overlap:
1. docstring_spec.md
2. quality_checks_spec.md
3. docstring_spec_cn.md
4. quality_checks_spec_cn.md
5. AGENTS.md

## Mandatory Validation

Before finalizing any code change, run and pass:
- pytest
- pyright
- ruff check src demos

If public exports changed, also run:
- python scripts/generate_init_pyi.py

## Docstring Rules

- Follow Google style section names only.
- Keep docstring content in English.
- Ensure typed parameters, returns, and attributes.

## Testing Rules

- Add or update tests for non-trivial behavior changes.
- Include boundary and error-path coverage when relevant.
- Prefer pytest-collected tests over script-style smoke tests.
- Keep tests deterministic.

## Type and Lint Rules

- Avoid new implicit Any in modified code.
- Do not add # type: ignore unless necessary and justified.
- Do not add broad Ruff ignore rules.

## Response Requirements

For implementation tasks, include:
- Changed files summary
- Behavior impact summary
- Validation commands executed
- Residual risks or follow-up items if any

## Working Style

- Keep changes small and focused.
- Preserve existing behavior unless the task explicitly asks for behavior changes.
- Prefer minimal-scope refactors.
