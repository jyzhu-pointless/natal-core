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
    ``_config``, ``_presets``, ``_species``,
    ``_gamete_modifiers``, ``_zygote_modifiers``, ``_manual_gamete``,
    ``_manual_zygote``, ``_index_registry``.
    """

    # Declared here so pyright knows these come from the host class.
    _config: Any  # type: ignore[assignment]
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
        overwritten. This explicit reset clears stored manual-fitness patches.
        """
        import numpy as np

        from natal.frontend.configurator import Configurator

        if self._config is None:
            return
        compiler = Configurator.for_population(cast("BasePopulation[Any]", self))
        candidate = compiler._genetic_candidate()  # pyright: ignore[reportPrivateUsage]  # compile an isolated reset declaration.
        candidate._fitness_base = tuple(np.ones_like(array) for array in candidate._fitness_base)  # pyright: ignore[reportPrivateUsage]
        candidate._fitness_steps = []  # pyright: ignore[reportPrivateUsage]
        candidate._compile_specification()  # pyright: ignore[reportPrivateUsage]
        compiler._commit_genetic_candidate(candidate)  # pyright: ignore[reportPrivateUsage]

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
        from natal.frontend.configurator import Configurator

        compiler = Configurator.for_population(cast("BasePopulation[Any]", self))
        candidate = compiler._genetic_candidate()  # pyright: ignore[reportPrivateUsage]  # all runtime recipes use the isolated build compiler.
        candidate._compile_specification(preserve_fitness=True)  # pyright: ignore[reportPrivateUsage]  # map-only updates preserve current native fitness.
        if rebuild_maps:
            compiler._commit_genetic_candidate(candidate)  # pyright: ignore[reportPrivateUsage]
        else:
            self._gamete_modifiers = list(candidate.gamete_modifiers)
            self._zygote_modifiers = list(candidate.zygote_modifiers)

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
        if self._config is None or self._index_registry is None:
            return
        if not self._index_registry.index_to_haplo or not self._index_registry.index_to_genotype:
            return
        from natal.frontend.configurator import Configurator
        from natal.frontend.configurator._registry_builder import rebuild_config_maps

        compiler = Configurator.for_population(cast("BasePopulation[Any]", self))
        candidate = compiler._genetic_candidate()  # pyright: ignore[reportPrivateUsage]  # isolate all user modifier effects before publication.
        candidate._config, _applied = rebuild_config_maps(  # pyright: ignore[reportPrivateUsage]  # the isolated candidate owns its working draft.
            candidate.species,
            candidate._config,  # pyright: ignore[reportPrivateUsage]
            candidate.registry,
            gamete_modifiers=self._gamete_modifiers,
            zygote_modifiers=self._zygote_modifiers,
            compress=False,
            host=candidate,
        )
        compiler._commit_genetic_candidate(candidate)  # pyright: ignore[reportPrivateUsage]

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
        from natal.frontend.configurator import Configurator

        compiler = Configurator.for_population(cast("BasePopulation[Any]", self))
        candidate = compiler._genetic_candidate()  # pyright: ignore[reportPrivateUsage]  # registration is published only after compilation succeeds.
        declarations = candidate._manual_gamete  # pyright: ignore[reportPrivateUsage]
        resolved_id = self._resolve_modifier_id(modifier_id, declarations)
        declarations.append((resolved_id, name, modifier))
        declarations.sort(key=lambda item: item[0])
        if refresh:
            candidate._compile_specification(preserve_fitness=True)  # pyright: ignore[reportPrivateUsage]
            compiler._commit_genetic_candidate(candidate)  # pyright: ignore[reportPrivateUsage]
        else:
            self._manual_gamete = list(declarations)
            self._gamete_modifiers = [*self._gamete_modifiers, (resolved_id, name, modifier)]

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
        from natal.frontend.configurator import Configurator

        compiler = Configurator.for_population(cast("BasePopulation[Any]", self))
        candidate = compiler._genetic_candidate()  # pyright: ignore[reportPrivateUsage]  # registration is published only after compilation succeeds.
        declarations = candidate._manual_zygote  # pyright: ignore[reportPrivateUsage]
        resolved_id = self._resolve_modifier_id(modifier_id, declarations)
        declarations.append((resolved_id, name, modifier))
        declarations.sort(key=lambda item: item[0])
        if refresh:
            candidate._compile_specification(preserve_fitness=True)  # pyright: ignore[reportPrivateUsage]
            compiler._commit_genetic_candidate(candidate)  # pyright: ignore[reportPrivateUsage]
        else:
            self._manual_zygote = list(declarations)
            self._zygote_modifiers = [*self._zygote_modifiers, (resolved_id, name, modifier)]

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
        from natal.frontend.configurator import Configurator

        Configurator.for_population(cast("BasePopulation[Any]", self)).presets(preset)

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
