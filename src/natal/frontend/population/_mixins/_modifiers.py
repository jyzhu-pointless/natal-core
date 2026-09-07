"""Modifier and preset management mixin for BasePopulation.

Extracted from :mod:`natal.frontend.population.base` to reduce the
BasePopulation ABC to its core lifecycle contract.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from natal.frontend.population._mixins._hooks import HookManagerMixin

if TYPE_CHECKING:
    from natal.frontend.population.base import BasePopulation

if TYPE_CHECKING:
    from natal.frontend.genetics import Species
    from natal.frontend.modifiers.module import GameteModifier, ZygoteModifier
    from natal.frontend.presets import GeneticPreset


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
        from natal.frontend.fitness import apply_preset_fitness_patch

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

    def refresh_modifiers(self, rebuild_maps: bool = True) -> None:
        """Rebuild derived modifier lists and maps from _presets + _manual_*.

        Presets are applied in priority order, then manual modifiers are
        appended.  Modifier maps (zygotes_to_gametes_map,
        gametes_to_zygotes_map, offspring_tensor) are rebuilt from the
        combined list.

        Args:
            rebuild_maps: If ``True`` (default), also call
                :meth:`refresh_modifier_maps`.  Set to ``False`` when the
                caller plans to batch multiple modifier registrations and
                will call :meth:`refresh_modifier_maps` once afterward.
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
        if rebuild_maps:
            self.refresh_modifier_maps()

    def refresh_modifier_maps(self) -> None:
        """Rebuild the three modifier maps from current modifier lists.

        Recomputes ``zygotes_to_gametes_map``,
        ``gametes_to_zygotes_map``, and the derived ``offspring_tensor``
        through the unified compiler
        (:func:`natal.frontend.genetics.compile.compile_modifier_maps`)
        — the same spelling the build path uses, so the two entry
        points cannot drift (pinned bit-for-bit by the parity safety
        net).

        .. note::

            This method is called automatically by :meth:`refresh_modifiers`
            and by individual ``add_gamete_modifier`` /
            ``add_zygote_modifier`` when ``refresh=True``.
        """
        from natal.frontend.data._engine import (
            initialize_gamete_map,
            initialize_zygote_map,
        )
        from natal.frontend.genetics.compile import compile_modifier_maps

        if self._config is None or self._registry is None:
            return

        haploid_genotypes = self._registry.index_to_haplo
        diploid_genotypes = self._registry.index_to_genotype
        if not haploid_genotypes or not diploid_genotypes:
            return

        n_glabs = int(self._config.n_glabs)
        n_slabs = int(self._config.n_slabs)

        # Step 1: full Mendelian baselines, projected onto the registry's
        # active flat axes.  Compression may retain arbitrary
        # (genotype, slab) and (haplotype, glab) entries, so neither
        # active axis is necessarily a Cartesian product.
        full_z2g = initialize_gamete_map(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=n_glabs,
            n_slabs=n_slabs,
        )
        full_g2z = initialize_zygote_map(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=n_glabs,
            n_slabs=n_slabs,
            unordered=self._species.unordered,
        )
        full_ztype_index = {
            (genotype, slab): genotype_idx * n_slabs + slab_idx
            for genotype_idx, genotype in enumerate(diploid_genotypes)
            for slab_idx, slab in enumerate(self._registry.slab_labels)
        }
        full_gtype_index = {
            (haplotype, glab): haplotype_idx * n_glabs + glab_idx
            for haplotype_idx, haplotype in enumerate(haploid_genotypes)
            for glab_idx, glab in enumerate(self._registry.glab_labels)
        }
        active_ztypes = [
            full_ztype_index[ztype] for ztype in self._registry.index_to_ztype
        ]
        active_gtypes = [
            full_gtype_index[gtype] for gtype in self._registry.index_to_gtype
        ]
        projected_z2g = full_z2g[:, active_ztypes, :][:, :, active_gtypes]
        projected_g2z = full_g2z[active_gtypes, :, :][:, active_gtypes, :][
            :, :, active_ztypes
        ]

        # Step 2: the unified compiler applies the modifier recipes on
        # the projected axes and derives the offspring tensor.
        # The mixin always composes into a BasePopulation; the runtime
        # refresh passes the live host so recipe factories can read it.
        z2g, g2z, offspring_tensor = compile_modifier_maps(
            projected_z2g,
            projected_g2z,
            gamete_modifiers=self._gamete_modifiers,
            zygote_modifiers=self._zygote_modifiers,
            registry=self._index_registry,
            population=cast("BasePopulation[Any]", self),
        )

        # Step 3: Persist all three maps into the config via shallow copy.
        n_g = int(z2g.shape[1])
        n_hg = int(z2g.shape[2])
        self._config = self._config._replace(
            zygotes_to_gametes_map=z2g,
            gametes_to_zygotes_map=g2z,
            offspring_tensor=offspring_tensor,
            n_ztypes=n_g,
            n_gtypes=n_hg,
            n_glabs=n_glabs,
        )
        # Modifier maps are session structure for the Rust bridge: the maps
        # (and dimension counters) changed, so the session must be rebuilt.
        self._rust_dirty.update({"meiosis_map", "offspring_tensor", "__hooks__"})

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

        Registration is idempotent by object identity: if the exact same
        preset instance is already in ``_presets``, this is a no-op.
        This prevents double-registration when ``presets(drive)`` is
        called twice — the alternative (appending twice) would cause
        ``refresh_modifiers()`` to build two copies of the preset's
        gamete/zygote modifier, double-applying its effect in the
        offspring tensor.

        Args:
            preset: A GeneticPreset instance (e.g., HomingDrive or custom preset).
        """
        if not any(p is preset for p in self._presets):
            self._presets.append(preset)

    def apply_preset(self, preset: GeneticPreset) -> None:
        """Apply a genetic preset to this population.

        This is the preferred API for registering presets. The preset's
        gamete modifiers, zygote modifiers, and fitness effects are
        registered in the correct order.

        Args:
            preset: A GeneticPreset instance (e.g., HomingDrive or custom preset).

        Examples:
            >>> from natal.frontend.presets import HomingDrive
            >>> drive = HomingDrive(
            ...     name="MyDrive",
            ...     drive_allele="Drive",
            ...     target_allele="WT",
            ...     drive_conversion_rate=0.95
            ... )
            >>> population.apply_preset(drive)

        See Also:
:class:`natal.frontend.presets.GeneticPreset` - Base class for creating custom presets
:class:`natal.frontend.presets.HomingDrive` - Built-in gene drive preset
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
