# Quality Checks Specification

> [Chinese version](./quality_checks_spec_cn.md). Keep both versions synchronized; English takes precedence.

## Scope and Responsibility

This specification owns quality policy for Python and Rust code changes, including executable examples. [AGENTS.md](./AGENTS.md) defines authorization, risk classification, and reviewer assignment. [docstring_spec.md](./docstring_spec.md) defines docstring and annotation format. Skills reference these policies instead of introducing separate thresholds.

Prose, comment, or layout changes that do not affect runtime or example code need accuracy, bilingual consistency, link, and relevant format checks rather than full code gates. Substantial rule or skill changes also need the independent scenario exercises described in AGENTS.md. Standalone syntax/style illustrations in specifications that do not call project code, describe project APIs, or express scientific models need document checks and snippet validation, not full project gates; executable project API/model examples remain code changes.

## Validation Timing and Commands

During development, run targeted checks that answer the current question. Before delivering a code change, run all four final gates against the completed artifacts:

```bash
pytest
pyright
ruff check src demos
python scripts/check_rust.py
```

This includes Rust-only changes. The Rust script's `cargo fmt --check`, `cargo clippy -- -D warnings`, and `cargo check --all-targets` are hard gates; optional rust-analyzer diagnostics do not block delivery. Use the repository's environment and configuration; if a command is unavailable, report the missing prerequisite rather than assume success.

The main agent may run checks. For high-risk changes, the evaluator independently executes final full gates; an identical preliminary full run by the author is unnecessary. After further substantive changes, rerun affected checks and review; repeat full gates when the changes invalidate prior full results or impact is unclear.

Apply `ruff check src demos --fix` only for relevant lint findings, and inspect its diff for unrelated changes. Do not run autofix as an unconditional gate.

For public signatures, defaults, export paths, or other changes to the stub surface, run `python scripts/generate_init_pyi.py` before final type checking. Complete necessary documentation and example updates before final review.

Run affected demos and runnable documentation examples with their required setup. Execute all demos and runnable examples for broad API migrations, shared infrastructure changes affecting their execution, or release validation. Distinguish illustrative fragments from runnable examples; report unavailable prerequisites and unexecuted relevant examples.

## Tests and Applicability

Prefer pytest-collected behavior tests under `tests/test_*.py`, small reusable fixtures, and one coherent behavioral assertion group per test. Each test must verify an explicit behavior contract or mathematical property and identify a concrete error it would catch. Existing tests may satisfy a change if their applicability is demonstrated; do not add redundant tests to meet a quota.

Cover the changed behavior, relevant boundaries, and the regression scenario for a fix. Select additional categories by their triggering conditions:

| Category | Required when | Evidence |
|---|---|---|
| Negative contract | The change removes or forbids an interface | Assert it is inaccessible, including affected exports |
| Ownership | An API promises copies, read-only results, or other isolation | Attempt mutation and verify the promised isolation |
| State transitions | The change involves or affects lifecycle behavior | Exercise relevant sequences such as restore→run, finish→snapshot, import→run, or clear→record |
| Configuration combinations | Several configuration options jointly affect the change | Identify valid combinations and interactions; test critical combinations |
| Error paths | Input constraints or failure handling change | Check the exception and the documented post-failure state |

A non-applicable category needs a short reason, not an invented test. Ownership follows the API contract: an explicitly shared mutable view is not automatically a defect, and a read-only flag alone does not prove isolation. For error paths, require unchanged state when the contract promises atomic failure; otherwise verify documented recovery behavior.

Exhaust small critical combination sets. For large sets, justify representative combinations, boundaries, and generated tests; do not claim exhaustive coverage after sampling.

Non-numerical assertions on interface absence, exception types, identity, mutability, or state are valid. Numerical tests must use independently justified expected values, error bounds, or mathematical invariants. Exact integer results should be exact; floating-point tolerances must have a numerical rationale. A run that merely avoids crashing does not verify numerical correctness.

For stochastic output claims, use statistical validation with justified sample size and tolerances. State the quantity tested (mean, variance, or distribution), assumptions, seed strategy for reproducibility, and how false failures are controlled. There is no universal minimum run count. A deterministic regression or per-sample bound may test a specific property but does not establish distributional correctness. See `numerical-verification` for methods.

## Coverage

- New modules require at least 95% line coverage.
- New executable code in existing modules requires at least 95% line coverage. Measure executed new lines against all executable new lines in the change; the whole module average cannot substitute.
- Report the coverage command, baseline/diff used for new-line measurement, measurement scope, and uncovered lines. For example, `pytest --cov=src/natal --cov-report=term-missing --cov-report=json` supplies Python line data; map it to the change for existing modules.
- Use appropriate language-specific coverage tooling for Rust executable changes; Python coverage does not establish Rust coverage.
- If coverage cannot be reliably measured, report it as unconfirmed and treat required measurement as blocked. Never claim the threshold from an estimate.
- Coverage does not replace contract or numerical checks. Do not hide executable lines with exclusions to reach the threshold.

## Types, Docstrings, and Lint

Use the current repository Pyright strict configuration and Ruff configuration; do not copy potentially stale settings into skills. Annotate function parameters, returns, and public attributes as specified in the docstring specification. Obvious local variable types may be inferred; annotate when inference is ambiguous or the type expresses a meaningful constraint.

Use precise types. `Any` and `object` require a specific documented reason when used as broad annotations; `object` may correctly describe an unknown value that is narrowed before use. Do not replace precise types with broad ones to silence errors. `cast(Any, …)` is forbidden. Use `cast(T, x)` only when a justified runtime invariant cannot be established by static analysis; prefer narrowing or restructuring. Every `# type: ignore` is a last resort and needs a brief, specific reason.

Type-only changes must preserve runtime behavior. Contract tightening requires corresponding tests and documentation. Respect existing justified lint exceptions; new suppressions must be narrow and explained. Do not disable rule families, leave dead code, or expand ignores to make checks pass.

Required type, docstring, and lint violations still need repair within the repair scope. Report their actual impact separately from scientific or runtime defects; architectural preferences and readability suggestions are not automatic blockers.

## Repair Scope and Baseline Failures

Fix all gate errors in modified code files and failures in other files caused by the change. This applies to pytest, Pyright, Ruff, and Rust gates. Do not broaden a task to repair unrelated files solely to obtain a green repository.

A failure in an untouched, unaffected file may be recorded as pre-existing only with evidence such as reproduction before the change, or on the recorded base revision under comparable conditions. A familiar message or unchanged filename is insufficient. Do not reset or overwrite user changes to obtain a baseline.

Record the failing command, diagnostic, baseline revision or captured state, environment comparison, and why the current change does not affect it. Confirmed unrelated baseline failures need not block delivery; still report the full gate as failed with an accepted baseline finding, never as passed. Modified files and failures caused by the change do not qualify for this exception.

If origin cannot be determined, investigate. A demonstrated in-scope defect requires repair; missing evidence or unavailable infrastructure that prevents required verification leaves the task blocked. Do not silently relabel failures, skip tests, suppress diagnostics, or change thresholds.

## Review Evidence and Verdicts

The main agent supplies basic tests and self-validation. The independent evaluator checks existing coverage and may add or strengthen tests during review; no separate tester stage is required. Preserve the implementation/review boundary in AGENTS.md: the evaluator edits tests and reports defects, while the main agent repairs product code.

For a behavior-testable blocker, prefer handing off an executed failing regression test with the confirmed requirement, test location, reproduction command, expected outcome, and observed failure. Confirm that it fails for the claimed behavior rather than a broken fixture or missing dependency. Other blockers may use static diagnostics or concrete evidence; do not force every finding into a test. Unexecuted tests are proposed checks, not reproduced failures.

Resolve essential contract ambiguity before treating a disputed expectation as a repair target. The main agent may challenge an incorrect test with evidence for evaluator correction but must not weaken, skip, or remove valid tests to make results pass. After repair, the evaluator independently reruns the repair targets and affected checks; final gate timing still follows the validation section. Issue a verdict against the final implementation and tests, including tests added during review.

Record four things: current requirements and evidence; issues introduced or affected by the change; confirmed unrelated baseline issues; and required checks that could not be completed.

Each requirement/check is `PASS`, `FAIL`, `NOT CHECKED`, or `N/A` with a reason where needed. Baseline failures retain their failing command result and link to the baseline evidence.

| Verdict | Condition |
|---|---|
| `APPROVED` | No in-scope blocker and all required validation completed; any unrelated baseline exception has evidence and is disclosed |
| `REJECTED` | An in-scope defect, required standards violation, missing applicable test, or measured coverage below 95% requires repair |
| `BLOCKED` | Missing tools, environment, evidence, or independent review prevents required verification, with no already established rejection reason |

If both known defects and missing checks exist, return `REJECTED` and list the blocked checks too. Unchecked requirements cannot be counted as passing. An approval is scoped to the reviewed artifacts and does not claim that accepted baseline failures disappeared.

Prioritize correctness and failing tests, then contract/type issues, then formatting and lint. The delivery report states changed files, behavior and rationale, executed commands and results, reviewer identity when applicable, and remaining limitations.
