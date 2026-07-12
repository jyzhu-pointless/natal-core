"""Modifier and preset management mixin for BasePopulation.

Extracted from :mod:`natal.population.base` to reduce the
BasePopulation ABC to its core lifecycle contract.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Sequence,
    Tuple,
)

from natal.population._mixins._hooks import HookManagerMixin

if TYPE_CHECKING:
    from natal.genetics import Species
    from natal.modifiers.module import GameteModifier, ZygoteModifier
    from natal.presets import GeneticPreset


class ModifierPresetMixin(HookManagerMixin):
    """Mixin providing modifier and preset management.

    Builds on HookManagerMixin since modifier compilation triggers
    hook rebuilds.

    Expects the host class (BasePopulation) to define these attributes:
    ``_config``, ``_registry``, ``_presets``, ``_species``,
    ``_gamete_modifiers``, ``_zygote_modifiers``, ``_manual_gamete``,
    ``_manual_zygote``, ``_index_registry``.
    """

    # Declared here so pyright knows these come from the host class.
    _config: Any  # type: ignore[assignment]
    _registry: Any  # type: ignore[assignment]
    _presets: list[Any]  # type: ignore[assignment]
    _species: Any  # type: ignore[assignment]
    _gamete_modifiers: list[tuple[int, Optional[str], Any]]  # type: ignore[assignment]
    _zygote_modifiers: list[tuple[int, Optional[str], Any]]  # type: ignore[assignment]
    _manual_gamete: list[tuple[int, Optional[str], Any]]  # type: ignore[assignment]
    _manual_zygote: list[tuple[int, Optional[str], Any]]  # type: ignore[assignment]
    _index_registry: Any  # type: ignore[assignment]

    # ========================================================================
    # Modifier management
    # ========================================================================
    def _next_modifier_id(self, modifiers: Sequence[Tuple[int, Optional[str], Any]]) -> int:
        """Return the next auto-assigned modifier id."""
        # Keep compatibility with legacy in-memory lists that may contain None ids.
        ids = [mid for mid, _, _ in modifiers]
        return (max(ids) + 1) if ids else 0

    def _resolve_modifier_id(self, modifier_id: Optional[int], modifiers: Sequence[Tuple[int, Optional[str], Any]]) -> int:
        """Normalize optional modifier_id into a concrete integer id."""
        if modifier_id is not None:
            return int(modifier_id)
        return self._next_modifier_id(modifiers)

    def reapply_preset_fitness(self) -> None:
        """Reset fitness tensors to 1.0 and re-apply all preset fitness patches.

        Called after structural changes to presets (addition, removal, or
        reconfiguration).  Only preset-derived fitness is restored — any
        fitness values set directly via ``pop.update().fitness()`` will be
        overwritten, because there is currently no manual-fitness storage
        analogous to ``_manual_gamete`` / ``_manual_zygote``.
        """
        from natal.fitness import apply_preset_fitness_patch

        if self._config is None:
            return
        self._config.viability_fitness.fill(1.0)
        self._config.fecundity_fitness.fill(1.0)
        self._config.sexual_selection_fitness.fill(1.0)
        self._config.zygote_viability_fitness.fill(1.0)
        for preset in sorted(self._presets, key=lambda p: p.priority):
            preset.bind_species(self._species)
            patch = preset.fitness_patch()
            if patch:
                apply_preset_fitness_patch(self, patch)  # type: ignore[arg-type]

    def refresh_modifiers(self) -> None:
        """Rebuild derived modifier lists and maps from _presets + _manual_*.

        Presets are applied in priority order, then manual modifiers are
        appended.  Modifier maps (zygotes_to_gametes_map,
        gametes_to_zygotes_map, offspring_tensor) are rebuilt from the
        combined list.

        .. note::

            This method does **not** touch fitness tensors.  Callers that
            need a full fitness rebuild should also call
            :meth:`reapply_preset_fitness`.
        """
        self._gamete_modifiers.clear()
        self._zygote_modifiers.clear()
        for preset in sorted(self._presets, key=lambda p: p.priority):
            preset.bind_species(self._species)
            if gm := preset.gamete_modifier(self):
                self._gamete_modifiers.append((
                    self._next_modifier_id(self._gamete_modifiers),
                    f"{preset.name}/gamete", gm,
                ))
            if zm := preset.zygote_modifier(self):
                self._zygote_modifiers.append((
                    self._next_modifier_id(self._zygote_modifiers),
                    f"{preset.name}/zygote", zm,
                ))
        self._gamete_modifiers.extend(self._manual_gamete)
        self._zygote_modifiers.extend(self._manual_zygote)
        self.refresh_modifier_maps()

    def refresh_modifier_maps(self) -> None:
        """Rebuild the three modifier maps from current modifier lists.

        Recomputes:
        - ``zygotes_to_gametes_map``: mapping from diploid genotype indices
          to haploid gamete probability distributions (one per sex).
        - ``gametes_to_zygotes_map``: mapping from paired haploid gametes back
          to diploid offspring genotype indices.
        - ``offspring_tensor``: precomputed 4-D tensor combining both maps
          for efficient Numba-based reproduction.

        The maps are stored in ``_config`` via ``_replace``, which creates a
        shallow copy of the config with updated fields.

        .. note::

            This method is called automatically by :meth:`refresh_modifiers`
            and by individual ``add_gamete_modifier`` / ``add_zygote_modifier``
            when ``refresh=True``.
        """
        from natal.data._engine import initialize_gamete_map, initialize_zygote_map
        from natal.engine.simulation.age_structured import (
            compute_offspring_probability_tensor,
        )
        from natal.modifiers.module import build_modifier_wrappers

        if self._config is None or self._registry is None:
            return

        haploid_genotypes = self._registry.index_to_haplo
        diploid_genotypes = self._registry.index_to_genotype
        if not haploid_genotypes or not diploid_genotypes:
            return

        n_glabs = int(self._config.n_glabs)
        n_slabs = int(self._config.n_slabs)

        # Step 1: Build wrapper callables from the combined modifier
        # lists (preset-derived + manually added).  Each wrapper is a
        # callable that accepts genotype indices and returns modified
        # probability vectors.
        gamete_funcs, zygote_funcs = build_modifier_wrappers(
            gamete_modifiers=self._gamete_modifiers,
            zygote_modifiers=self._zygote_modifiers,
            population=self,
            registry=self._index_registry,
        )

        # Step 2: Build the gametogenesis map.  For each diploid genotype
        # and sex, produce a probability distribution over the resulting
        # haploid gametes per gamete label.
        zygotes_to_gametes_map = initialize_gamete_map(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=n_glabs,
            n_slabs=n_slabs,
            gamete_modifiers=gamete_funcs,
        )

        # Step 3: Build the fusion map.  For each pair of haploid gametes
        # (one maternal, one paternal), determine the resulting diploid
        # offspring genotype index.
        gametes_to_zygotes_map = initialize_zygote_map(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=n_glabs,
            n_slabs=n_slabs,
            zygote_modifiers=zygote_funcs,
        )

        # Step 5: Compute the full offspring probability tensor by
        # convolving the maternal and paternal gametogenesis maps through
        # the fusion map.  The result is a 4-D array indexed by
        # (maternal_genotype, paternal_genotype, gamete_label, offspring_genotype).
        n_g = int(self._config.n_ztypes)
        n_hg = int(self._config.n_gtypes)
        offspring_tensor = compute_offspring_probability_tensor(
            meiosis_f=zygotes_to_gametes_map[0],
            meiosis_m=zygotes_to_gametes_map[1],
            haplo_to_genotype_map=gametes_to_zygotes_map,
            n_ztypes=n_g,
            n_gtypes=n_hg,
        )

        # Step 6: Persist all three maps into the config via shallow copy.
        self._config = self._config._replace(
            zygotes_to_gametes_map=zygotes_to_gametes_map,
            gametes_to_zygotes_map=gametes_to_zygotes_map,
            offspring_tensor=offspring_tensor,
            n_ztypes=n_g,
            n_gtypes=n_hg,
            n_glabs=n_glabs,
        )

    def add_gamete_modifier(
        self,
        modifier: GameteModifier,
        name: Optional[str] = None,
        modifier_id: Optional[int] = None,
        refresh: bool = True,
    ) -> None:
        """Register a gamete-level modifier.

        Args:
            modifier: A ``GameteModifier`` callable or object.
            name: Optional human-readable name for debugging.
            modifier_id: Optional numeric priority used for ordering.
            refresh: If True (default), immediately rebuild modifier maps.
                Set to False when adding multiple modifiers in a batch;
                call :meth:`refresh_modifiers` or
                :meth:`refresh_modifier_maps` afterward to apply all at once.
        """
        resolved_id = self._resolve_modifier_id(modifier_id, self._manual_gamete)
        self._manual_gamete.append((resolved_id, name, modifier))
        self._manual_gamete.sort(key=lambda x: x[0])
        self._gamete_modifiers.append((resolved_id, name, modifier))
        self._gamete_modifiers.sort(key=lambda x: x[0])
        if refresh:
            self.refresh_modifier_maps()

    def add_zygote_modifier(
        self,
        modifier: ZygoteModifier,
        name: Optional[str] = None,
        modifier_id: Optional[int] = None,
        refresh: bool = True,
    ) -> None:
        """Register a zygote-level modifier.

        Args:
            modifier: A ``ZygoteModifier`` callable or object.
            name: Optional human-readable name for debugging.
            modifier_id: Optional numeric priority used for ordering.
            refresh: If True (default), immediately rebuild modifier maps.
                Set to False when adding multiple modifiers in a batch;
                call :meth:`refresh_modifiers` or
                :meth:`refresh_modifier_maps` afterward to apply all at once.
        """
        resolved_id = self._resolve_modifier_id(modifier_id, self._manual_zygote)
        self._manual_zygote.append((resolved_id, name, modifier))
        self._manual_zygote.sort(key=lambda x: x[0])
        self._zygote_modifiers.append((resolved_id, name, modifier))
        self._zygote_modifiers.sort(key=lambda x: x[0])
        if refresh:
            self.refresh_modifier_maps()

    def add_preset(self, preset: GeneticPreset) -> None:
        """Add a preset to this population.

        Args:
            preset: A GeneticPreset instance (e.g., HomingDrive or custom preset).
        """
        self._presets.append(preset)

    def apply_preset(self, preset: GeneticPreset) -> None:
        """Apply a genetic preset to this population.

        This is the preferred API for registering presets. The preset's
        gamete modifiers, zygote modifiers, and fitness effects are
        registered in the correct order.

        Args:
            preset: A GeneticPreset instance (e.g., HomingDrive or custom preset).

        Examples:
            >>> from natal.presets import HomingDrive
            >>> drive = HomingDrive(
            ...     name="MyDrive",
            ...     drive_allele="Drive",
            ...     target_allele="WT",
            ...     drive_conversion_rate=0.95
            ... )
            >>> population.apply_preset(drive)

        See Also:
:class:`natal.presets.GeneticPreset` - Base class for creating custom presets
:class:`natal.presets.HomingDrive` - Built-in gene drive preset
        """
        self.add_preset(preset)
        self.refresh_modifiers()
        self.reapply_preset_fitness()

    @classmethod
    def builder(cls, species: Species) -> Any:
        """Create a builder for this population type.

        This is the recommended way to construct populations with presets.

        Args:
            species: Genetic architecture for the population.

        Returns:
            A builder instance for this population type.

        Examples:
            >>> pop = (AgeStructuredPopulation.builder(species)
            ...     .set_age_structure(n_ages=10)
            ...     .add_preset(HomingModificationDrive(...))
            ...     .build())
        """
        raise NotImplementedError(f"{cls.__name__} must implement builder()")

    def register_gamete_labels(self, labels: Optional[Sequence[str]]) -> None:
        """
        Register gamete labels in the IndexRegistry.

        Args:
            labels: Sequence of string labels to register. Labels must be
                unique in the provided sequence. Existing labels are ignored.
        """
        if not hasattr(self, "_index_registry") or self._index_registry is None:
            raise RuntimeError("IndexRegistry not initialized; cannot register gamete labels")

        if labels is None:
            return

        # Normalize and validate input
        try:
            seq = list(labels)
        except Exception as e:
            raise TypeError("labels must be a sequence of strings") from e

        # Ensure provided labels are unique
        if len(set(seq)) != len(seq):
            raise ValueError("labels must be unique")

        # Register each string label if not already present
        for lab in seq:
            if lab not in self._index_registry.glab_labels:
                self._index_registry.glab_labels.append(lab)
