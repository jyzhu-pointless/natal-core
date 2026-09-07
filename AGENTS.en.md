# AGENTS.md

> This is the English version. See also: [Chinese version](./AGENTS.md)
>
> Update both versions together. The English version takes precedence if meanings conflict.

## Language and Sources of Rules

Reply in Chinese unless the user explicitly asks in English. Use American spelling in English code and documentation, and write docstrings in English.

This file defines authorization, risk classification, and agent collaboration. [quality_checks_spec.md](./quality_checks_spec.md) owns quality requirements; [docstring_spec.md](./docstring_spec.md) owns docstring and type annotation format. Chinese counterparts explain the same rules; the English specification governs each topic. Skills provide methods rather than duplicate gate or coverage policies.

## Authorization and Scope

- An explicit request to implement, fix, or refactor a goal authorizes investigation, implementation, testing, and necessary repairs within that scope without approval for each step. Existing authorization remains valid.
- Explain and obtain approval before expanding scope, changing public API or scientific model semantics outside the authorization, adding production dependencies, or performing destructive operations. Do not ask again for changes already explicitly authorized.
- When several implementations are viable, choose a simple, verifiable approach consistent with the existing architecture. Ask only when missing information, user preferences, or important tradeoffs affect the decision; continue independent work while waiting.
- Do not commit, push, modify `.gitignore`, or create Markdown documentation files without an explicit user request. Necessary synchronization of existing documentation is part of the authorized task.
- Preserve existing user changes. Do not expand repair scope, suppress errors, or alter check configuration to manufacture passing results.

## Implementation and Communication

- Do not abstract for hypothetical needs. Comments explain intent, constraints, and non-obvious logic rather than repeat the code.
- Explain what changed, why, and its effect in plain language. Explain technical terms not already established in the project.
- Use tools available in the current environment that fit the task; prefer `rg` for search and allow shell for batch operations. Tool names do not prescribe a fixed workflow.

## Risk Classification

State the classification and a short reason in the initial work update. Classification needs no separate approval; explain and upgrade it when investigation reveals greater risk. Judge behavioral impact, not line count.

| Class | Criteria | Required validation and collaboration |
|---|---|---|
| Documentation and formatting | Prose, comments, or layout only; no runtime or example code changes | Check accuracy, bilingual consistency, links, and relevant formatting; no full code gates |
| Local code change | Clear impact, with none of the high-risk concerns below | Targeted tests and full gates before delivery; the main agent may complete it |
| High-risk change | Scientific formulas, random distributions, state restoration, mutable data sharing, public API contracts, Python/Rust data exchange, or unclear code impact | Delegate a tester, obtain independent evaluator review, and run final full gates |

Text-only rule or skill changes use documentation checks; add independent scenario exercises when they materially change agent decisions. Classify executable project API or model example changes by code risk and run affected examples. Standalone syntax or style illustrations in specifications that do not call project code, describe project APIs, or express scientific models use document checks and validation of the snippets themselves, without full project gates. A small formula change remains high-risk.

## Validation Timing and Roles

Use this order: implementation and targeted validation → necessary test strengthening → documentation and stub synchronization → final validation and any required independent review. Tests may precede or accompany implementation.

- **tester**: Delegation is required for high-risk code changes. Find missing scenarios, verify existing coverage, and strengthen tests where needed. Cite adequate existing tests rather than add tests to satisfy a quota. Use `numerical-verification` for numerical tests and `adversarial-review` for contract attacks when relevant.
- **docs**: Public API changes require synchronized `docs/zh/`, `docs/en/`, and related examples before final review. Delegate according to workload; the main agent may do the work.
- **evaluator**: Independently review final code, tests, stubs, and documentation and independently run final full gates for high-risk code changes. Delegate local changes when requested or when review uncertainty warrants it. Use `adversarial-review`; load `numerical-verification` only for numerical checks. Do not automatically load or launch the global `code-review` workflow.

These names describe responsibilities implemented with the environment's subagent tools and available concurrency. Without delegation capability, mark high-risk work as “independent review incomplete”; self-review cannot substitute for approval.

The main agent may run any validation command but must distinguish self-test results from independent review. When the evaluator runs final full gates, the main agent need not duplicate them beforehand. After substantive post-review edits, revalidate and review affected areas; rerun full gates if prior full results are invalidated or the impact is unclear.

## Completion

[The quality specification](./quality_checks_spec.md) is the single source for test applicability, coverage, gate commands, baseline failure evidence, and verdicts.

- High-risk code changes require an evaluator's `APPROVED`; do not claim completion while required validation is missing.
- The main agent may deliver local code changes against the same quality standard without mandatory independent approval.
- Deliver documentation changes after applicable checks; resolve blockers from required independent scenario exercises for rule changes first.
- Report changed files, behavior and rationale, commands actually executed and their results, and residual risks or follow-ups. Identify self-tests versus independent review and never describe confirmed baseline failures as “all checks passed.”
