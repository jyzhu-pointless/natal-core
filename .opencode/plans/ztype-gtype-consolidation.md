# ZType/GType Consolidation Plan

## Context

PR #14 consolidates ZType/GType terminology following a 5-member adversarial audit. The codebase currently has dead code, redundant canonicalization, inconsistent naming, and missing symmetry between ztype/gtype APIs. This plan executes the consensus findings from all 5 auditors.

**Key facts:**
- `Genotype.__new__` already canonicalizes when `species.unordered` is True (genetic_entities.py:617)
- 11 call sites have redundant `if species.unordered: gt = species.unordered_genotype(maternal, paternal)` patterns — pure overhead
- Three resolver methods in `index_registry.py` survive only because `modifiers.py` needs flexible key parsing
- 1037 tests currently pass; all phases must maintain this count

**Guiding principles:**
1. No call site uses `Genotype` where `ZType` is correct, or `HaploidGenotype`+`glab` where `GType` is correct
2. Internal APIs use `ztype`/`gtype` terminology; `Genotype`/`HaploidGenotype` only in user-facing API docstrings
3. Gamete and Zygote sides strictly symmetric
4. No downstream unordered checks — `Species` canonicalizes at construction

---

## Task Dependency Graph

| Task | Depends On | Reason |
|------|------------|--------|
| A1: Delete dead functions from index_registry.py | None | Pure deletion, no dependencies |
| A2: Delete dead type_def.py exports | None | Pure deletion, independent |
| A3: Delete _resolve_genotype_key from base_population.py | None | Pure deletion, independent |
| A4: Update tests for Phase A deletions | A1, A2, A3 | Must know what was deleted before fixing tests |
| B1: Move resolve_genotype_index logic to modifiers.py as _resolve_gidx | A1 | Must complete after dead code removal to avoid merge conflicts |
| B2: Replace resolve_comp_idx with gtype_index in _apply_comp_map | A1 | Depends on dead deletions for clean context |
| B3: Fold resolve_hg_glab_part into _parse_zygote_key | A1 | Depends on dead deletions for clean context |
| B4: Delete orphaned helpers from index_registry.py (_is_unordered, _as_pair) | B1, B2, B3 | Only dead after all resolver callers are removed |
| B5: Update tests for Phase B | B1, B2, B3, B4 | Must verify after all resolver consolidation |
| C1: Add gtype_indices_for to IndexRegistry | None | New method, no dependencies |
| C2: Add tests for gtype_indices_for | C1 | Tests depend on new method |
| D1: Remove redundant unordered_genotype calls in age_structured_population.py | None | Independent file edits |
| D2: Remove redundant unordered_genotype calls in discrete_generation_population.py | None | Independent file edits |
| D3: Remove redundant unordered_genotype calls in population_builder.py | None | Independent file edits |
| E1: Rename n_gen → n_ztypes in age_structured_simulator.py | None | Independent file edit |
| E2: Rename g → n_ztypes in discrete_generation_simulator.py | None | Independent file edit |
| E3: Rename n_genotypes → n_ztypes in migration/adjacency.py | None | Independent file edit |
| E4: Rename n_genotypes → n_ztypes in migration/kernel.py | None | Independent file edit |
| E5: Rename n_genotypes → n_ztypes in modifiers.py | None | Independent file edit |
| E6: Rename n_genotypes → n_ztypes in state_translation.py | None | Independent file edit |
| E7: Rename n_genotypes → n_ztypes in population_state.py docstrings | None | Independent file edit |

---

## Parallel Execution Graph

### Wave 1 (Start immediately — no dependencies):
```
├── A1: Delete dead functions from index_registry.py
├── A2: Delete dead type_def.py exports
├── A3: Delete _resolve_genotype_key from base_population.py
├── C1: Add gtype_indices_for to IndexRegistry
├── D1: Remove redundant unordered_genotype in age_structured_population.py
├── D2: Remove redundant unordered_genotype in discrete_generation_population.py
├── D3: Remove redundant unordered_genotype in population_builder.py
├── E1–E7: All naming fixes (7 parallel edits)
```

### Wave 2 (After Wave 1 completes):
```
├── A4: Update tests for Phase A deletions (depends: A1, A2, A3)
├── C2: Add tests for gtype_indices_for (depends: C1)
```

### Wave 3 (After Wave 2 + Wave 1 gate passes):
```
├── B1: Move resolve_genotype_index logic to modifiers.py (depends: A1)
├── B2: Replace resolve_comp_idx with gtype_index (depends: A1)
├── B3: Fold resolve_hg_glab_part into _parse_zygote_key (depends: A1)
```

### Wave 4 (After Wave 3 completes):
```
├── B4: Delete orphaned helpers from index_registry.py (depends: B1, B2, B3)
```

### Wave 5 (After Wave 4 completes):
```
├── B5: Update tests for Phase B (depends: B4)
```

**Critical Path:** A1 → B1 → B4 → B5
**Estimated Parallel Speedup:** ~60% faster than sequential (naming fixes + Phase D run in parallel with Phase A deletions)

---

## Tasks

### Task A1: Delete dead functions from index_registry.py

**Description:** Remove 8 dead methods, 10 no-op property setters, and 2 dead property getters from `index_registry.py`.

**Delegation Recommendation:**
- Category: `quick` — straightforward deletions, single file
- Skills: [`programming`] — for Python editing conventions

**Skills Evaluation:**
- INCLUDED `programming`: Python file edits with type discipline
- OMITTED `quality-checker`: Deferred to verification gate after wave

**Depends On:** None
**Blocks:** A4, B1, B2, B3

**Acceptance Criteria:**
- Functions removed: `haplo_index()`, `gamete_label_index()`, `somatic_label_index()`, `_ensure_genotype_index()`, `_ensure_haplo_index()`, `_ensure_glab_index()`, `_ensure_haplo_registered()`, `num_gamete_labels()`, `num_somatic_labels()`
- 10 no-op property setters removed (identified as `@some_property.setter` with no side effects, zero callers each)
- Dead property getters removed if their setters are also dead and no external callers exist
- `ruff check src/index_registry.py` passes
- `pyright` passes on the file

---

### Task A2: Delete dead type_def.py exports

**Description:** Remove `GenotypeIndex`, `IndividualType`, `HaploidGenotypeIndex`, `GameteType`, `GlabIndex` and any associated helper functions that become orphaned. Only delete types with **zero imports outside type_def.py**.

**Delegation Recommendation:**
- Category: `quick` — simple deletions after import verification
- Skills: [`programming`]

**Skills Evaluation:**
- INCLUDED `programming`: Python file edits
- OMITTED others: No complexity

**Depends On:** None
**Blocks:** A4

**Acceptance Criteria:**
- All 5 types deleted from type_def.py (only if confirmed zero external imports)
- Any helper functions that become dead after type removal also deleted
- `pyright` passes

---

### Task A3: Delete _resolve_genotype_key from base_population.py

**Description:** Remove `_resolve_genotype_key()` method (line 420-425). Verified zero callers. This method is buggy for ordered species because it unconditionally calls `unordered_genotype()` without checking `species.unordered`.

**Delegation Recommendation:**
- Category: `quick` — single method deletion
- Skills: [`programming`]

**Skills Evaluation:**
- INCLUDED `programming`: Python file edits
- OMITTED others: Trivial scope

**Depends On:** None
**Blocks:** A4

**Acceptance Criteria:**
- Method removed from `base_population.py`
- `pyright` passes

---

### Task A4: Update tests for Phase A deletions

**Description:** Remove or update test functions that reference deleted APIs. Tests to modify in `test_index_registry.py`:
- `test_haplo_index_lookup` (L121-126) → delete test
- `test_num_gamete_labels_empty` (L140-142) → delete test
- `test_num_gamete_labels_after_registration` (L144-148) → delete test
- `test_gamete_label_index_lookup` (L157-162) → delete test
- `test_genotype_and_haplo_indices_independent` (L166-178) → rewrite to use `gtype_index` instead of `haplo_index`
- `test_compress_haplotypes` (L220-236) → rewrite to use `gtype_index` instead of `haplo_index`

**Delegation Recommendation:**
- Category: `quick` — test-only changes
- Skills: [`programming`]

**Skills Evaluation:**
- INCLUDED `programming`: Python test edits

**Depends On:** A1, A2, A3
**Blocks:** Phase A verification gate

**Acceptance Criteria:**
- All 6 affected tests pass or are deleted
- No test references deleted APIs
- `pytest tests/test_index_registry.py -x` passes

---

### Task B1: Move resolve_genotype_index logic to modifiers.py

**Description:** Create a local `_resolve_gidx(diploid_genotypes, gk, index_registry)` function in `modifiers.py` that:
1. Resolves a flexible genotype key → genotype index (0..G-1) using int/object/string matching
2. Immediately expands the result to ZType indices using `index_registry.ztype_indices_for(gt)` when `expand_to_ztypes` is True
3. Replaces all 6 call sites of `resolve_genotype_index()` + `_expand_gidx()` in `tensor_modifier` (gamete) and `_normalize_zygote_val`

**Delegation Recommendation:**
- Category: `deep` — needs careful refactoring of the resolver logic
- Skills: [`programming`]

**Skills Evaluation:**
- INCLUDED `programming`: Python logic refactoring
- OMITTED others: No visual/frontend/doc work

**Depends On:** A1
**Blocks:** B4

**Acceptance Criteria:**
- `_resolve_gidx` function exists in `modifiers.py`
- All 6 call sites in `modifiers.py` use `_resolve_gidx` instead of `index_registry.resolve_genotype_index` + `_expand_gidx`
- `pyright` passes
- Existing modifier tests pass unchanged (behavior preserved)

---

### Task B2: Replace resolve_comp_idx with gtype_index lookup

**Description:** In `_apply_comp_map` (modifiers.py:405-436), replace `index_registry.resolve_comp_idx(haploid_genotypes, n_glabs, comp_key, strict=False)` with a direct lookup using:
1. For bare `int` keys: return as-is (backward compat for compressed integer keys in modifier fixtures)
2. For `(hg_part, glab_part)` tuples: resolve using `haploid_genotypes.index()` + `index_registry.glab_to_index.get()` then compute `hg_idx * n_glabs + glab_idx`
3. For `HaploidGenotype` keys: `haploid_genotypes.index() * n_glabs`
4. For `str` keys: `haplo.to_string()` match + `gtype_index()` fallback

**Delegation Recommendation:**
- Category: `deep` — moderate logic refactoring with backward compat requirement
- Skills: [`programming`]

**Skills Evaluation:**
- INCLUDED `programming`: Python logic refactoring

**Depends On:** A1
**Blocks:** B4

**Acceptance Criteria:**
- `resolve_comp_idx` no longer called from `_apply_comp_map`
- Existing modifier tests pass
- `pyright` passes

---

### Task B3: Fold resolve_hg_glab_part into _parse_zygote_key

**Description:** In `_parse_zygote_key` (modifiers.py:439-472), replace calls to `index_registry.resolve_hg_glab_part()` with inline logic using existing helpers (`_as_pair`, `haploid_genotypes`, `index_registry.glab_to_index`). The function already imports `_as_pair` and has access to all needed data structures.

**Delegation Recommendation:**
- Category: `deep` — refactoring with backward compat
- Skills: [`programming`]

**Skills Evaluation:**
- INCLUDED `programming`: Python logic refactoring

**Depends On:** A1
**Blocks:** B4

**Acceptance Criteria:**
- `resolve_hg_glab_part` no longer called from `_parse_zygote_key`
- Existing zygote modifier tests pass
- `pyright` passes

---

### Task B4: Delete orphaned helpers from index_registry.py

**Description:** After B1/B2/B3 remove all callers, delete from `index_registry.py`:
- `resolve_genotype_index()` (line 584-642)
- `resolve_hg_glab_part()` (line 644-746)
- `resolve_comp_idx()` (line 748-852)
- `_is_unordered()` (line 864-870) — only called by `resolve_genotype_index`
- `_as_pair()` (line 855-861) — three copies exist (also in modifiers.py and genetic_presets.py); remove only the `index_registry.py` copy after its callers are gone

**Delegation Recommendation:**
- Category: `quick` — simple deletions
- Skills: [`programming`]

**Skills Evaluation:**
- INCLUDED `programming`: Python file edits

**Depends On:** B1, B2, B3
**Blocks:** B5

**Acceptance Criteria:**
- All 5 helpers deleted
- `ruff check src/index_registry.py` passes
- `pyright` passes

---

### Task B5: Update tests for Phase B

**Description:**
1. Update `tests/test_modifiers.py` — the test comments reference `resolve_hg_glab_part` (lines 41, 57); update comments to reflect new inline logic
2. No new test failures from Phase B refactoring
3. Verify that all modifier tests still pass with consolidated resolver logic

**Delegation Recommendation:**
- Category: `quick` — comment updates + verification
- Skills: [`programming`]

**Skills Evaluation:**
- INCLUDED `programming`: Python test edits

**Depends On:** B4
**Blocks:** Phase B verification gate

**Acceptance Criteria:**
- Test comments updated
- All modifier tests pass
- `pytest tests/test_modifiers.py -x` passes

---

### Task C1: Add gtype_indices_for to IndexRegistry

**Description:** Add `gtype_indices_for(haplo: HaploidGenotype) -> list[int]` method to `IndexRegistry`, symmetric to existing `ztype_indices_for(genotype: Genotype)`. Scans `_index_to_gtype` to find all GType indices for a given haploid genotype.

```python
def gtype_indices_for(self, haplo: HaploidGenotype) -> list[int]:
    """Return all GType indices for a given HaploidGenotype object.

    Scans ``_index_to_gtype`` — analogous to ``ztype_indices_for``
    for the gamete side.
    """
    return [i for i, (hg, _) in enumerate(self._index_to_gtype) if hg == haplo]
```

**Delegation Recommendation:**
- Category: `quick` — single new method
- Skills: [`programming`]

**Skills Evaluation:**
- INCLUDED `programming`: Python method addition
- OMITTED others: Trivial scope

**Depends On:** None
**Blocks:** C2

**Acceptance Criteria:**
- Method added to `IndexRegistry` class
- Type annotations correct
- `pyright` passes

---

### Task C2: Add tests for gtype_indices_for

**Description:** Add test cases in `test_index_registry.py` covering:
1. Single haploid → single GType index returned
2. Haploid with multiple gamete labels → all GType indices returned
3. Haploid not registered → empty list
4. Symmetry with `ztype_indices_for`

**Delegation Recommendation:**
- Category: `quick` — straightforward tests
- Skills: [`programming`]

**Skills Evaluation:**
- INCLUDED `programming`: Python test writing

**Depends On:** C1
**Blocks:** Phase C gate

**Acceptance Criteria:**
- 4 test cases pass
- `pytest tests/test_index_registry.py -k gtype_indices -x` passes
- `pyright` passes

---

### Task D1: Remove redundant unordered_genotype in age_structured_population.py

**Description:** In `_distribute_initial_population` and `_distribute_initial_sperm_storage`, remove the `if species.unordered: gt = species.unordered_genotype(gt.maternal, gt.paternal)` calls at lines 261-270, 333-342, 358-367. Keep only the `@slab` logic and pattern parsing.

**Delegation Recommendation:**
- Category: `deep` — careful line-level edits to preserve surrounding logic
- Skills: [`programming`]

**Skills Evaluation:**
- INCLUDED `programming`: Python file edits

**Depends On:** None
**Blocks:** Phase D-gate

**Acceptance Criteria:**
- Lines 261-270, 333-342, 358-367 simplified (unordered call removed)
- `@slab` and pattern parsing preserved
- `pyright` passes

---

### Task D2: Remove redundant unordered_genotype in discrete_generation_population.py

**Description:** Same as D1 but for `discrete_generation_population.py` lines 333-342.

**Delegation Recommendation:**
- Category: `quick` — single block
- Skills: [`programming`]

**Skills Evaluation:**
- INCLUDED `programming`: Python file edit

**Depends On:** None
**Blocks:** Phase D-gate

**Acceptance Criteria:**
- Lines 333-342 simplified
- `pyright` passes

---

### Task D3: Remove redundant unordered_genotype in population_builder.py

**Description:** Remove redundant `if species.unordered:` guards and `unordered_genotype()` calls at 6 locations in `resolve_age_structured_initial_individual_count`, `resolve_age_structured_initial_sperm_storage`, `resolve_discrete_initial_individual_count` methods. Lines: 785-787, 795-805, 860-869, 885-894, 1006-1008, 1016-1025.

**Delegation Recommendation:**
- Category: `deep` — 6 locations, careful variable preservation
- Skills: [`programming`]

**Skills Evaluation:**
- INCLUDED `programming`: Python file edits

**Depends On:** None
**Blocks:** Phase D-gate

**Acceptance Criteria:**
- All 6 locations simplified
- `pyright` passes

---

### Task E1–E7: Naming fixes (7 parallel edits)

**Description:** Rename internal variables only (no NamedTuple fields, no public API):

| Task | File | Old Name | New Name |
|------|------|----------|----------|
| E1 | `engine/age_structured_simulator.py:80` | `n_gen` | `n_ztypes` |
| E2 | `engine/discrete_generation_simulator.py:40` | `g` | `n_ztypes` |
| E3 | `engine/migration/adjacency.py:67` | `n_genotypes` | `n_ztypes` |
| E4 | `engine/migration/kernel.py:235` | `n_genotypes` | `n_ztypes` |
| E5 | `modifiers.py:207` | `n_genotypes` (local var) | `n_ztypes` |
| E6 | `state_translation.py` | `n_genotypes` throughout | `n_ztypes` |
| E7 | `population_state.py` | `n_genotypes` in docstrings | `n_ztypes` |

**Delegation Recommendation:**
- Category: `quick` — rename-only, single variable per file
- Skills: [`programming`] (x7)

**Skills Evaluation:**
- INCLUDED `programming`: Variable rename in Python files
- OMITTED others: Trivial scope per task

**Depends On:** None (all independent)
**Blocks:** Phase E gate

**Acceptance Criteria (per task):**
- Variable renamed everywhere in the file
- `ruff check <file>` passes
- `pyright` passes on the file

**Scope boundary — DO NOT rename:**
- `zygotes_to_gametes_map` / `gametes_to_zygotes_map` (~15 call sites)
- `n_haploid_genotypes` → `n_gtypes` (PopulationConfig NamedTuple field)
- `female_genotype_compatibility` → `female_ztype_compatibility` (PopulationConfig field)
- `n_genotypes` in `population_state.py` property names (only docstrings)

---

## Commit Strategy

Each phase should be a single atomic commit (or two if tests are in separate commit):

1. `refactor: remove dead code (Phase A)` — A1 + A2 + A3 + A4
2. `refactor: consolidate resolver methods into modifiers.py (Phase B)` — B1 + B2 + B3 + B4 + B5
3. `feat: add gtype_indices_for symmetry (Phase C)` — C1 + C2
4. `refactor: remove redundant unordered_genotype calls (Phase D)` — D1 + D2 + D3
5. `refactor: rename internal variables to ztype/gtype terminology (Phase E)` — E1–E7

**Gate between each commit:** `ruff check src demos && pyright && pytest -x`

---

## Verification Gates

After each phase:
```bash
ruff check src demos && pyright && pytest -x
```

**Expectation:** 1037 tests pass after every phase. No reduction in test count except:
- Phase A: ~5 tests deleted (dead API coverage), expect ~1032 remaining
- All other phases: test count unchanged

---

## Success Criteria

1. All dead code removed from `index_registry.py`, `base_population.py`, `type_def.py`
2. `resolve_genotype_index`, `resolve_comp_idx`, `resolve_hg_glab_part` callers consolidated from `index_registry.py` into `modifiers.py`
3. `gtype_indices_for` added with tests, symmetric with `ztype_indices_for`
4. 11 redundant `unordered_genotype` call sites removed
5. 7 internal variables renamed to `n_ztypes`
6. `ruff check src demos` — zero errors
7. `pyright` — zero errors
8. `pytest` — all tests pass
9. No regression in modifier behavior or population initialization

---

## Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Phase B resolver consolidation breaks modifier edge cases | Medium | Preserve backward compat for bare int keys; gate with `pytest tests/test_modifiers.py` |
| Phase D unordered removal breaks ordered species | Low | `Genotype.__new__` only canonicalizes when `species.unordered`; ordered species never hit the removed code paths |
| Test fixture references to deleted `haplo_index` in tests outside test_index_registry.py | Low | Grep for `haplo_index` references across all test files before final gate |

---

## Scope Boundaries

**IN SCOPE:**
- Dead code removal in `index_registry.py`, `base_population.py`, `type_def.py`
- Resolver consolidation into `modifiers.py`
- Adding `gtype_indices_for`
- Removing redundant `unordered_genotype` calls at 11 sites
- Internal variable renames to `n_ztypes`

**DEFERRED (not in this plan):**
- Renaming `zygotes_to_gametes_map` / `gametes_to_zygotes_map` (~15 sites, too much breakage)
- Renaming `n_haploid_genotypes` → `n_gtypes` (PopulationConfig NamedTuple, public API)
- Renaming `female_genotype_compatibility` (PopulationConfig field, public API)
- Any user-facing docstring changes beyond `population_state.py`
- Any changes to `configurator.py`, `genetic_presets.py`, `observation.py`, `ui/` files
