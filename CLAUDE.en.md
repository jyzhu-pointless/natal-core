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

## Validation Gates

After every change, run the following commands:

```bash
pytest                          # Run all tests
pyright                         # Type checking (strict mode)
ruff check src demos             # Lint check
ruff check src demos --fix       # Lint auto-fix
python scripts/generate_init_pyi.py  # Regenerate stub after public API changes
```

The virtual environment is auto-activated — run commands directly.

All three must pass before committing: `pytest` + `pyright` + `ruff check src demos`. No suppression, no workarounds.

### Fix Policy

- **Modified files**: All pyright / ruff / pytest failures must be fixed, regardless of whether they pre-existed.
- **Files affected by the change**: If a signature or import change causes failures elsewhere, those must be fixed too.
- **Pre-existing issues in untouched files**: Note and analyze them; fixing is encouraged but not required for the current commit.
- **`cast(Any, …)` is forbidden**. Never use it to bypass type checking.
- **Do not abuse `Any` or `object`**: parameter, return, and variable type annotations must use specific types — don't take shortcuts with `Any` or `object`. Adding new imports for type annotations is worth it. `Any` is acceptable only with a concrete, documented justification (e.g., `Callable[..., Any]` to mean "any callable").
- **`cast(T, x)`** may be used only when static analysis cannot prove `x: T` at all (e.g., narrowing an `Optional` after a guard). Prefer type-narrowing assertions or restructuring first.
- **`# type: ignore`** is a last resort. Every ignore must include a short, specific reason on the same line.

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

The `natal` package uses **lazy loading** — `__init__.py` parses `__all__` via AST to build a name-to-module index, and `__getattr__` imports on first access. Startup is nearly instant.

### Core Modules

| Layer | Modules | Responsibility |
|---|---|---|
| Genetic Structure | `genetic_structures.py`, `genetic_entities.py` | Immutable Species / Chromosome / Locus blueprint; Gene / Genotype / Haplotype entities |
| Config & State | `population_config.py`, `population_state.py` | PopulationConfig (static NamedTuple, 9 ecological params as 0-d ndarray) + PopulationState (mutable arrays) |
| Population Models | `discrete_generation_population.py`, `age_structured_population.py`, `spatial_population.py` | Wright-Fisher / age-structured / spatial multi-deme simulation |
| Builder | `population_builder.py`, `configurator.py` | Fluent builder API + Configurator for runtime modification |
| Hook System | `hooks/` | Event-driven intervention (init/first/early/late/finish), declarative + Python function styles |
| Engine | `engine/` | Numba-accelerated simulation loop + codegen for dynamic wrapper generation |
| Presets | `genetic_presets.py` | HomingDrive / ToxinAntidoteDrive, encapsulating modifier + fitness |
| Parameter Registry | `parameters.py` | ParamDescriptor registry + Numba setter codegen |

### Data Flow

```
Species → IndexRegistry → PopulationConfig + PopulationState
  → Per tick: Hooks → Reproduction → Competition → Survival → Observation
  → Spatial models: per-deme simulation → migration
```

### Key Design Decisions

- **0-d ndarray**: 9 ecological parameters (K, eggs, sex_ratio, etc.) are mutable in-place via `config.field[()] = v` within hooks.
- **Hook signature**: `hook(state, config, deme_id) → int` — config is passed directly into hooks.
- **Numba-first, Python-fallback**: The core simulation runs with or without Numba; `njit_switch` auto-degrades.
- **Codegen caching**: Generated kernel wrappers are cached with content-addressed hashing so repeated simulations reuse compiled modules.
