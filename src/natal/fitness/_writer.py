"""Fitness DSL writer for the Configurator API.

Resolves genotype-pattern selectors to ztype indices and writes into
config fitness arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import NDArray

from natal.data import DiscretePopulationConfig, PopulationConfig
from natal.genetics import Species
from natal.registry.index import IndexRegistry

if TYPE_CHECKING:
    from natal.genetics import Genotype


def _get_fitness_array(
    config: PopulationConfig | DiscretePopulationConfig,
    field_name: str,
) -> NDArray[np.float64]:
    """Map a fitness field name to the corresponding config array.

    Args:
        config: The PopulationConfig or DiscretePopulationConfig.
        field_name: One of ``"viability"``, ``"fecundity"``,
            ``"sexual_selection"``, or ``"zygote_viability"``.

    Returns:
        The corresponding fitness ndarray from *config*.

    Raises:
        ValueError: If *field_name* is not recognised.
    """
    if field_name == "viability":
        return config.viability_fitness  # type: ignore[return-value]
    if field_name == "fecundity":
        return config.fecundity_fitness  # type: ignore[return-value]
    if field_name == "sexual_selection":
        return config.sexual_selection_fitness  # type: ignore[return-value]
    if field_name == "zygote_viability":
        return config.zygote_viability_fitness  # type: ignore[return-value]
    raise ValueError(f"Unknown fitness field: {field_name}")


def write_fitness_field(
    config: PopulationConfig | DiscretePopulationConfig,
    field_name: str,
    patch: Mapping[str, float | Mapping[str, float]],
    mode: str,
    *,
    species: Species,
    registry: IndexRegistry,
    all_genotypes: list[Genotype],
) -> None:
    """Resolve genotype-pattern strings and write into a fitness tensor.

    *field_name* is one of ``"viability"``, ``"fecundity"``,
    ``"sexual_selection"``, or ``"zygote_viability"``.  *patch* is a
    dict mapping genotype selectors to fitness values, with optional
    sex-keyed or genotype-keyed nesting.

    The function detects the format of *patch* and dispatches to one of
    four branches.  See inline comments for the detection rules.

    Supported formats::

        {genotype: val}                                        # scalar → both sexes, all ages
        {genotype: {"female": val, "male": val}}               # per-selector sex-keyed
        {genotype: {0: val, 1: val}}                           # per-selector age-keyed
        {genotype: {"female": {0: val}}}                       # per-selector sex+age keyed
        {"female": {genotype: val}, "male": {...}}             # top-level sex-keyed
        {female_g: {male_g: val}}                              # sexual_selection pair format
    """
    # ══════════════════════════════════════════════════════════════════════
    # BRANCH 1: top-level sex-keyed
    #   {"female": {genotype: val}, "male": {genotype: val}}
    #
    # Detection: ALL top-level keys are "female" or "male".
    # Action: iterate sex→genotype_dict, delegate each to _write_fitness_field_flat.
    # ══════════════════════════════════════════════════════════════════════
    if patch and all(k in ("female", "male") for k in patch):
        # ---- guard: every value must itself be a dict {genotype: val} ----
        if not all(isinstance(v, Mapping) for v in patch.values()):
            raise TypeError(
                "sex-keyed fitness dict values must be genotype→value mappings"
            )
        sex_patch: Mapping[str, Mapping[str | tuple[Genotype | str, str], float]]
        sex_patch = patch  # type: ignore[assignment]  # Mapping key invariance; narrowed by branch guard
        # ---- write female slice, then male slice ----
        for sex_key, geno_dict in sex_patch.items():
            sex_idx = 0 if sex_key == "female" else 1
            _write_fitness_field_flat(
                config, field_name, geno_dict, mode,
                sex_idx=sex_idx,
                species=species, registry=registry,
                all_genotypes=all_genotypes,
            )
        return

    # ══════════════════════════════════════════════════════════════════════
    # BRANCH 2: sexual_selection — nested female→male pair format
    #   {female_g: {male_g: val}}
    #
    # Detection: field is "sexual_selection" AND any value is a Mapping.
    # Action: resolve female & male selectors independently,
    #         then write into each [f_idx, m_idx] cell.
    # ══════════════════════════════════════════════════════════════════════
    if field_name == "sexual_selection":
        arr = config.sexual_selection_fitness          # shape: (g, g) — [female_idx, male_idx]
        has_nested = any(isinstance(v, Mapping) for v in patch.values())
        if has_nested:
            for female_selector, male_map in patch.items():           # outer: female genotype key
                if not isinstance(male_map, Mapping):                 # guard: must be {male_g: val}
                    raise TypeError(
                        "Mixed scalar/nested format in sexual_selection. "
                        "When using nested female→male pairs, all values "
                        "must be dicts mapping male selectors to values."
                    )
                for male_selector, value in male_map.items():         # inner: male genotype key → float
                    # ---- resolve both selectors to genotype indices ----
                    matched_f = species.resolve_genotype_selectors(
                        selector=female_selector,
                        all_genotypes=all_genotypes,
                        context="sexual_selection (female)",
                    )
                    matched_m = species.resolve_genotype_selectors(
                        selector=male_selector,
                        all_genotypes=all_genotypes,
                        context="sexual_selection (male)",
                    )
                    # ---- write every female×male combination ----
                    for f_geno in matched_f:
                        for f_z in registry.ztype_indices_for(f_geno):
                            for m_geno in matched_m:
                                for m_z in registry.ztype_indices_for(m_geno):
                                    val = float(value)
                                    if mode == "replace":
                                        arr[f_z, m_z] = val
                                    else:
                                        arr[f_z, m_z] *= val
            return

        # ═══════════════════════════════════════════════════════════════
        # BRANCH 3: sexual_selection — flat male-keyed format
        #   {male_g: val}
        #
        # Detection: field is "sexual_selection" AND no nested values.
        # Action: resolve male selector, write value to ALL female rows
        #         of the matched male column (arr[:, m_idx]).
        # ═══════════════════════════════════════════════════════════════
        for male_selector, value in patch.items():
            if isinstance(value, Mapping):                            # guard: mixed scalar/nested
                raise TypeError(
                    "Mixed scalar/nested format in sexual_selection. "
                    "When using nested female→male pairs, all values "
                    "must be dicts."
                )
            # ---- resolve the male genotype to an index ----
            matched_m = species.resolve_genotype_selectors(
                selector=male_selector,
                all_genotypes=all_genotypes,
                context="sexual_selection (male)",
            )
            for m_geno in matched_m:
                for m_z in registry.ztype_indices_for(m_geno):
                    val = float(value)
                    if mode == "replace":
                        arr[:, m_z] = val        # broadcast: all females × this male
                    else:
                        arr[:, m_z] *= val
        return

    # ══════════════════════════════════════════════════════════════════════
    # BRANCH 4: per-selector resolution (viability / fecundity / zygote)
    #   {genotype: val}
    #   {genotype: {"female": val, "male": val}}           — sex-keyed
    #   {genotype: {0: val, 1: val}}                       — age-keyed
    #   {genotype: {"female": {0: val}}}                   — sex+age keyed
    #
    # Detection: everything not caught by branches 1-3.
    # Each selector value may be:
    #   - scalar → apply to both sexes, all ages
    #   - Mapping → inspect the first key to decide the format
    # ══════════════════════════════════════════════════════════════════════
    for selector, value in patch.items():
        if isinstance(value, Mapping):
            # Inspect the first key to determine the nesting structure.
            first_key = next(iter(value.keys()))
            if isinstance(first_key, int) and not isinstance(first_key, bool):  # type: ignore[unnecessary-isinstance] — bool ⊂ int in Python
                # ---- age-keyed: {genotype: {0: val, 1: val}} ----
                for age_key, age_val in value.items():          # type: ignore[var-unknown]  # Mapping values are Any without explicit type params
                    if age_val is None:                         # type: ignore[unnecessary-comparison]  # user may pass {age: None} to skip
                        continue
                    age = int(age_key)
                    for sex_idx in (0, 1):
                        _write_fitness_field_flat(
                            config, field_name,
                            {selector: float(age_val)}, mode,
                            sex_idx=sex_idx, age_idx=age,
                            species=species, registry=registry,
                            all_genotypes=all_genotypes,
                        )
            elif first_key in ("female", "male"):
                # ---- sex-keyed: {genotype: {"female": val, "male": val}} ----
                for sex_key, sex_val in value.items():
                    sex_idx = 0 if sex_key == "female" else 1
                    if isinstance(sex_val, Mapping):
                        # ---- sex+age keyed: {genotype: {"female": {0: val}}} ----
                        for age_key, age_val in sex_val.items():          # type: ignore[var-unknown]  # Mapping values are Any without explicit type params
                            if age_val is None:
                                continue
                            _write_fitness_field_flat(
                                config, field_name,
                                {selector: float(age_val)}, mode,  # type: ignore[arg-type]  # age_val is Unknown from unparameterized Mapping
                                sex_idx=sex_idx, age_idx=int(age_key),  # type: ignore[arg-type]  # age_key is Unknown from unparameterized Mapping
                                species=species, registry=registry,
                                all_genotypes=all_genotypes,
                            )
                    else:
                        # ---- simple sex-keyed (existing behavior) ----
                        _write_fitness_field_flat(
                            config, field_name,
                            {selector: float(sex_val)}, mode,
                            sex_idx=sex_idx,
                            species=species, registry=registry,
                            all_genotypes=all_genotypes,
                        )
            else:
                raise TypeError(
                    f"Unrecognised key in fitness value dict: {first_key!r}. "
                    f"Expected 'female'/'male' (sex-keyed) or int (age-keyed)."
                )
        else:
            # ---- scalar format: {genotype: val} → apply to both sexes, all ages ----
            for sex_idx in (0, 1):
                _write_fitness_field_flat(
                    config, field_name,
                    {selector: float(value)}, mode,
                    sex_idx=sex_idx,
                    species=species, registry=registry,
                    all_genotypes=all_genotypes,
                )


def _write_fitness_field_flat(
    config: PopulationConfig | DiscretePopulationConfig,
    field_name: str,
    patch: Mapping[str | tuple[Genotype | str, str], float],
    mode: str,
    *,
    sex_idx: int,
    species: Species,
    registry: IndexRegistry,
    all_genotypes: list[Genotype],
    age_idx: int | None = None,
) -> None:
    """Write a flat (per-ZType) fitness patch into the correct config array.

    The target array shape depends on *field_name*:

    - ``"viability"`` → ``(n_sexes, n_ages, n_ztypes)`` — writes ``[sex_idx, default_age, zidx]``
    - ``"fecundity"`` → ``(n_sexes, n_ztypes)`` — no age axis
    - ``"sexual_selection"`` → ``(n_ztypes, n_ztypes)`` — no age axis
    - ``"zygote_viability"`` → ``(n_sexes, n_ztypes)`` — no age axis

    When *age_idx* is ``None`` (the default), the write targets the
    last juvenile age (``new_adult_age - 1``) — viability fitness
    normally represents larval / juvenile survival, not adult fitness.
    ``fecundity`` and ``zygote_viability`` have no age axis so
    *age_idx* is ignored for them.

    Args:
        config: The PopulationConfig or DiscretePopulationConfig to modify.
        field_name: One of ``"viability"``, ``"fecundity"``,
            ``"sexual_selection"``, or ``"zygote_viability"``.
        patch: A flat ``{genotype_selector: value}`` mapping.
        mode: ``"replace"`` (overwrite) or ``"multiply"`` (scale existing).
        sex_idx: Index for the sex axis (0 = female, 1 = male).
        species: The Species for genotype-selector resolution.
        registry: The IndexRegistry mapping genotypes to indices.
        all_genotypes: List of all genotype objects for selector matching.
        age_idx: Age index for the write (defaults to ``new_adult_age - 1``).
    """
    # Default to last juvenile age: viability typically affects
    # larvae/juveniles, not adults.  DiscretePopulationConfig has
    # no ``new_adult_age`` field (always 2 ages, adult at age 1).
    resolved_age: int = age_idx if age_idx is not None else (
        getattr(config, "new_adult_age", 1) - 1
    )

    # Resolve valid slab labels for tuple selectors with @slab suffix.
    raw_slabs: list[str] = getattr(species, "somatic_labels", None) or ["default"]

    for selector, value in patch.items():
        # ── tuple syntax: (Genotype, "slab_label") ──
        if isinstance(selector, tuple):
            if len(selector) != 2:
                raise TypeError(
                    f"fitness tuple selector must have 2 elements "
                    f"(genotype_key, slab_label), got {len(selector)}"
                )
            _genotype_key, _slab = selector
            if _slab not in raw_slabs:
                raise ValueError(
                    f"Unknown slab label '{_slab}'. "
                    f"Available slabs: {raw_slabs}"
                )

            if isinstance(_genotype_key, Genotype):
                matched = [_genotype_key]
            else:
                matched = species.resolve_genotype_selectors(
                    selector=_genotype_key,
                    all_genotypes=all_genotypes,
                    context=f"fitness.{field_name}",
                )

            for genotype in matched:
                age_slice = slice(resolved_age, resolved_age + 1)
                zidx = registry.ztype_index(genotype, _slab)

                arr = _get_fitness_array(config, field_name)
                if field_name == "viability":
                    if mode == "replace":
                        arr[sex_idx, age_slice, zidx] = float(value)
                    else:
                        arr[sex_idx, age_slice, zidx] *= float(value)
                elif field_name == "fecundity":
                    if mode == "replace":
                        arr[sex_idx, zidx] = float(value)
                    else:
                        arr[sex_idx, zidx] *= float(value)
                elif field_name == "sexual_selection":
                    if mode == "replace":
                        if sex_idx == 0:
                            arr[zidx, :] = float(value)
                        else:
                            arr[:, zidx] = float(value)
                    else:
                        if sex_idx == 0:
                            arr[zidx, :] *= float(value)
                        else:
                            arr[:, zidx] *= float(value)
                elif field_name == "zygote_viability":
                    if mode == "replace":
                        arr[sex_idx, zidx] = float(value)
                    else:
                        arr[sex_idx, zidx] *= float(value)
            continue
        # ── end tuple branch ──

        from natal.patterns import LabPattern, ZygoteTypePattern

        selector_str = str(selector)
        pattern = ZygoteTypePattern.parse(selector_str, species)
        z_indices = registry.resolve_ztype_indices(pattern)

        # For | patterns (not ::), also try :: for unordered matching.
        # Ordered | may only partially match (e.g. *|A → AA but not Aa).
        # Only promote for unordered species (consistent with
        # genetic_structures.Species._resolve_single_genotype_selector).
        if species.unordered and "|" in selector_str and "::" not in selector_str:
            try:
                unordered_str = selector_str.replace("|", "::", 1)
                unordered_pattern = ZygoteTypePattern.parse(unordered_str, species)
                unordered_indices = registry.resolve_ztype_indices(unordered_pattern)
                if len(unordered_indices) >= len(z_indices):
                    z_indices = unordered_indices
            except Exception:
                pass

        if not z_indices:
            # Check for invalid slab first — give a specific error
            if "@" in selector_str:
                _, s_str = selector_str.rsplit("@", 1)
                lab = LabPattern.parse(s_str)
                matching_slabs = [s for s in raw_slabs if lab.matches(s)]
                if not matching_slabs:
                    raise ValueError(
                        f"No slab matches '{s_str}' in fitness.{field_name} "
                        f"selector '{selector_str}'.  Available: {raw_slabs}"
                    )

        if not z_indices:
            raise ValueError(
                f"No zygote type matches '{selector_str}' in fitness.{field_name}"
            )

        age_slice = slice(resolved_age, resolved_age + 1)

        arr = _get_fitness_array(config, field_name)
        for zidx in z_indices:
            if field_name == "viability":
                if mode == "replace":
                    arr[sex_idx, age_slice, zidx] = float(value)
                else:
                    arr[sex_idx, age_slice, zidx] *= float(value)
            elif field_name == "fecundity":
                if mode == "replace":
                    arr[sex_idx, zidx] = float(value)
                else:
                    arr[sex_idx, zidx] *= float(value)
            elif field_name == "sexual_selection":
                if mode == "replace":
                    if sex_idx == 0:
                        arr[zidx, :] = float(value)
                    else:
                        arr[:, zidx] = float(value)
                else:
                    if sex_idx == 0:
                        arr[zidx, :] *= float(value)
                    else:
                        arr[:, zidx] *= float(value)
            elif field_name == "zygote_viability":
                if mode == "replace":
                    arr[sex_idx, zidx] = float(value)
                else:
                    arr[sex_idx, zidx] *= float(value)
