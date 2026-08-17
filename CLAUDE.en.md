# CLAUDE.md (English)

> This is the English version. See also: [中文版 (Chinese version)](./CLAUDE.md)
>
> When one version is updated, the other must be updated in lockstep.

## Language

默认使用中文回答。仅在用户明确使用英文提问时用英文回复。

(This English version of CLAUDE.md exists for internationalization convenience — the project's working language remains Chinese.)

## Spec References

The following files define the coding, documentation, and testing standards for this project,
in priority order:

1. `docstring_spec.md`
2. `quality_checks_spec.md`
3. `docstring_spec_cn.md` (Chinese explanation)
4. `quality_checks_spec_cn.md` (Chinese explanation)

When rules conflict, the English specification files take precedence.

## Behavioral Guidelines

Follow the Karpathy Coding Guidelines (Claude Code command: `/andrej-karpathy-skills:karpathy-guidelines`):

1. **Think before coding** — Stop and ask when uncertain; don't guess. If a simpler approach exists, say so.
2. **Simplicity first** — Write the minimum code needed to solve the problem. No abstractions for single-use cases. No error handling for impossible scenarios.
3. **Surgical changes** — Touch only what you must. Don't reformat adjacent code. Don't refactor on the side.
4. **Goal-driven execution** — Turn tasks into verifiable goals. For multi-step work, state a plan before executing.

Project-specific rules:

- Any plan, proposal, or non-trivial change must be presented to the user and approved before execution. Do not implement without approval.
- Prefer writing comments. Comments should explain WHY (design intent, constraints, non-obvious logic), not WHAT (the code already says that).
- Do not create documentation files (*.md) unless the user explicitly asks for them.
- No premature abstraction. Don't design for hypothetical future needs.
- After making changes, explain in detail what you changed, why you changed it, and what effect it has. Avoid vague statements like "I changed this function" — instead say specifically what was changed (e.g., "I changed the parameter of `foo` from `x` to `y` to support the new use case").
- Use plain, accessible language in explanations. If you introduce a specialized software engineering term (beyond concepts already established in this project's architecture), you must explain it clearly. Example:
  - Don't say: "Preset parameters are managed by Configurator's deferred mechanism, and runtime changes also go through Configurator."
  - Instead say: "Genetic presets have an unusual requirement — their parameters can't be written directly into the config during construction. The Configurator stores them temporarily (deferred) and applies them all at once when `build()` executes. Runtime changes to preset parameters also go through the Configurator's `update()` entry point."

- **Use multiple-choice questions for clarifying decisions**: When the user needs to choose among viable alternatives (design trade-offs, naming, parameter values), use the AskUserQuestion tool to present 2–4 concrete options for direct selection. This avoids open-ended back-and-forth communication.
- **Maintain the Tasks list**: For multi-step work, use TaskCreate/TaskUpdate to track progress. Keep task statuses up to date (pending → in_progress → completed) and don't leave zombie tasks behind. Trivial single-step operations don't need a task.
- **Prefer dedicated tools over Bash**: Read/Glob/Grep/Edit/Write are more reliable than shell commands — they aren't blocked by the sandbox, produce stable output formats, and render better. Use Bash only for batch operations, pipe compositions, or tasks that dedicated tools cannot handle.
- **Never commit or push without explicit permission**: Do not execute `git commit`, `git push`, or any form of submission unless the user explicitly requests it.
- **Never modify `.gitignore`**: Do not alter `.gitignore` files unless the user explicitly requests it.

## Validation Gates

After every change, run the following commands:

```bash
pytest                          # Run all tests
pyright                         # Type checking (strict mode)
ruff check src demos             # Lint check
ruff check src demos --fix       # Lint auto-fix
python scripts/check_rust.py    # Rust: fmt + clippy + check (optional rust-analyzer diagnostics)
python scripts/generate_init_pyi.py  # Regenerate stub after public API changes
```

The virtual environment is auto-activated — run commands directly.

The Python gates and the Rust hard gates must pass before committing: `pytest` + `pyright` + `ruff check src demos`, plus `python scripts/check_rust.py` (any failure in its `cargo fmt --check`, `cargo clippy -- -D warnings`, or `cargo check --all-targets` fails the gate; rust-analyzer diagnostics are optional and non-blocking). No suppression, no workarounds.

### Review Workflow

After every code change, you MUST follow this sequence. **Self-review is prohibited**:

1. **`@tester`** — Generate strict tests following `numerical-verification` and `adversarial-review` standards. Must cover five test categories: **negative contract tests** (assert deleted interfaces are unreachable), **ownership tests** (assert returned ndarray/list/dict is a copy or read-only), **state-transition tests** (restore→run, finish→snapshot, import→run, clear→record), **axis-combination tests** (Cartesian product of configuration axes), **error-path tests** (invalid input raises correct exception with state unchanged). Every assertion must prove a numerical invariant.
2. **Run `python scripts/generate_init_pyi.py`** — Regenerate stubs after public API changes so subsequent pyright checks work against the latest stubs.
3. **`@evaluator`** — Perform **adversarial** review. The reviewer takes an attacker's stance, actively searching for evidence the code violates the spec, rather than merely verifying that gates pass. Must:
   - Build a **ledger** from the spec: three lists — must-exist, must-not-exist, invariants
   - Do a **full-repo search** (src/tests/demos/docs/stub) for every must-not-exist item, confirming they are unreachable
   - **Attack** every invariant: ownership attacks (check ndarray references/write protection), state-machine attacks (restore→run, finish→snapshot), axis-combination attacks (Cartesian product enumeration)
   - **Execute** every demo script and documented code example
   - Run `pytest` / `pyright` / `ruff` / `python scripts/check_rust.py` gates, load `code-review`, `numerical-verification`, `adversarial-review` skills
   - Use a **mechanical** verdict: `APPROVED` ⇔ zero hard-blockers
4. **If the change involves public API (signatures, parameters, defaults, module renames, etc.), the primary agent invokes `@docs`** to sync documentation and code examples in `docs/zh/` and `docs/en/`.
5. **The primary agent MUST NOT run `pytest`, `pyright`, `ruff`, or `python scripts/check_rust.py` on its own and claim "review passed."** These results must be independently verified by the evaluator, which produces a structured report.

The change is only considered complete when the evaluator returns an `APPROVED` verdict.

#### Evaluator Mandatory Rejection Criteria

The evaluator **MUST** return `REJECTED` when any of the following conditions are met. **Severity does not soften a hard-blocker**: a missing comment on `Any` and a semantic breakage are treated identically.

- **Test failures**: Any `FAILED` in `pytest`.
- **Rust hard gate failure**: Any non-zero `cargo fmt --check`, `cargo clippy -- -D warnings`, or `cargo check --all-targets` (run through `python scripts/check_rust.py`). Optional rust-analyzer diagnostics do not constitute rejection.
- **Insufficient coverage**: Line coverage for new modules or new code in existing modules **< 95%**.
- **Hard-blocker present**: REJECTED on any one of the following:
  - **Must-not-exist violation**: spec-required deletion still accessible (including `getattr`, `__init__` re-export)
  - **Invariant broken**: ownership leak (ndarray reference/non-write-protected), state-machine error (tick desync after restore), axis-combination crash
  - **Demo/doc crash**: demo script or documented code example fails at runtime
  - **Uncommented `Any` / `object`**
  - **Uncommented `# type: ignore`**
  - **`cast(Any, …)`**
  - **Non-Google-style docstring section** or **missing type annotations on parameters/returns/attributes**
- **Missing negative contract**: spec-flagged deletion without a corresponding assert-not-exists test.
- **Missing ownership test**: public method returning a container (ndarray/list/dict) without a corresponding read-only/copy verification test.
- **Uncovered state transitions**: key lifecycle sequences such as restore→run, finish→snapshot, import→run, clear→record are untested.

### Fix Policy

- **Modified files**: All pyright / ruff / pytest / cargo fmt / cargo clippy / cargo check failures must be fixed. "Pre-existing failures" do not exist — if tests pass on `main` and fail on the branch, the change introduced a regression and must be fixed immediately.
- **Files affected by the change**: If a signature or import change causes failures elsewhere, those must be fixed too.
- **Pre-existing issues in untouched files**: Note and analyze them; fixing is encouraged but not required for the current commit.
- **`cast(Any, …)` is forbidden**. Never use it to bypass type checking. Any occurrence triggers REJECTION.
- **Do not abuse `Any` or `object`**: parameter, return, and variable type annotations must use specific types — don't take shortcuts with `Any` or `object`. Adding new imports for type annotations is worth it. `Any` is acceptable only with a concrete, documented justification (e.g., `Callable[..., Any]` to mean "any callable").
- **`cast(T, x)`** may be used only when static analysis cannot prove `x: T` at all (e.g., narrowing an `Optional` after a guard). Prefer type-narrowing assertions or restructuring first.
- **`# type: ignore`** is a last resort. Every ignore must include a short, specific reason on the same line. Missing comment triggers REJECTION.

### Test Coverage

- **New modules**: ≥95% line coverage.
- **New code in existing modules**: ≥95% line coverage.
- **Deterministic simulations** (`stochastic=False`): exact numerical assertions on counts, frequencies, or derived statistics.
- **Stochastic simulations**: statistical validation required — multiple runs, confidence intervals, or distributional checks. A single passing run is not sufficient.
- **Prefer pytest-collected tests** over script-style smoke tests.

### Docstring Requirements

- Use Google-style sections only (`Args:`, `Returns:`, `Raises:`, etc.). Do not invent new section names.
- Docstring text must be in **English**.
- All parameters, returns, and attributes must be explicitly typed (annotations preferred).

### Change Notes

Every change summary must include:
1. Files changed
2. Behavior changes
3. Validation commands executed
4. Residual risks or follow-up items (if any)

## Architecture

The `natal` package uses **lazy loading** — the top-level `__init__.py` statically parses literal `__all__` declarations from public subpackages via AST and builds a name-to-module index; `__getattr__` imports the owning subpackage only on first access.

### Core Modules

| Layer | Subpackages | Responsibility |
|---|---|---|
| Genetics & Indexing | `genetics/`, `patterns/`, `registry/` | Define Species / Chromosome / Locus and genetic entities; parse genotype patterns; map ZTypes / GTypes to integer engine indices |
| Configuration & Data | `configurator/`, `data/` | Fluent construction and runtime updates through Configurator; PopulationConfig / DiscretePopulationConfig and their State array containers |
| Population Models | `population/`, `spatial/` | Age-structured and discrete-generation populations; multi-deme containers, spatial topology, and migration configuration |
| Genetic Effects | `modifiers/`, `presets/`, `fitness/` | Gamete/zygote conversion rules; presets such as HomingDrive, ToxinAntidoteDrive, and Wolbachia; fitness-patch parsing and writes |
| Hook System | `hooks/` | Compile declarative operations and `@hook` functions for prioritized first / early / late / finish events |
| Engine & Acceleration | `engine/`, `numba/` | Age-structured, discrete-generation, and spatial lifecycles; migration kernels; Numba switching, compatibility, and dynamic wrapper generation |
| Output & UI | `output/`, `ui/` | Observation filtering, history recording, readable-format translation, and interactive visualization |

### Data Flow

```
Species → Configurator → IndexRegistry + PopulationConfig / DiscretePopulationConfig
  → Presets, modifiers, and fitness → optional ZType / GType compression
  → Population + PopulationState / DiscretePopulationState

Standard per tick: first Hook → Reproduction → early Hook → Density regulation + Survival
  → late Hook → Aging → tick + 1

Discrete-generation extreme-speed: first Hook → fused Wright-Fisher tick → tick + 1
  (no early / late Hooks)

Spatial models: per-deme lifecycle → Migration → optional Observation / History recording
```

The `finish` Hook is outside the per-tick lifecycle and runs when a population finishes its simulation.

### Key Design Decisions

- **Engine data and mutable parameters are separated**: Config and State are both NamedTuples. Scalar metadata cannot be replaced in place, while array contents remain mutable; nine ecological parameters use 0-d ndarrays so Configurator and Hooks can update them through `config.field[()] = value`.
- **Flat type indexing**: IndexRegistry directly maintains indices for `ZType = (Genotype, slab)` and `GType = (HaploidGenotype, glab)`. At build time, reachability-based compression can prune both index spaces and their related config arrays.
- **One configuration entry point**: `AgeStructuredPopulation.setup(...)`, `DiscreteGenerationPopulation.setup(...)`, and `pop.update()` use the same fluent Configurator API. During construction, ConfigContext applies presets and modifiers before a Population exists and rebuilds genetic maps and the offspring tensor when needed; `.fitness()` instead writes directly to Config.
- **Dual-path Hooks**: Declarative operations compile into contiguous arrays and offset tables (CSR), while custom functions register through `@hook`; both paths can be interleaved by priority. The function signature is `hook(state, config, deme_id) -> int`.
- **Numba-first, Python-fallback**: `njit_switch` enables Numba by default and can switch to Python implementations for debugging. On the Numba path, spatial wrappers run deme lifecycles in parallel and then perform migration.
- **Stable-identity codegen cache**: Hook and lifecycle wrappers derive stable hashes from callable `module:qualname` identities, dispatch structure, and selectors. Generated source and Numba artifacts are stored in `.numba_cache/`; identical combinations use stable paths and reuse compiled results across runs when the cache remains valid.
