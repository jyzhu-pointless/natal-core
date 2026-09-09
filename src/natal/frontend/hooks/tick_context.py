"""Event-scoped Python hook views backed by an owned Rust transaction.

State and parameter edits become visible together when the callback succeeds.
An exception discards the candidate, including random draws. The controlled RNG
uses the owning Rust stream, and retained parameter, update, and RNG handles
reject access after the callback returns. Public population snapshots remain
isolated from the writable callback candidate.
"""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from natal.frontend.hooks._transaction import (
    EventTransaction,
    GuardedConfigurator,
    HookRng,
)
from natal.frontend.hooks.types import RESULT_STOP

if TYPE_CHECKING:
    from natal.frontend.data import ModelDraft
    from natal.frontend.genetics import Species
    from natal.frontend.population._params_view import ParamsView
    from natal.frontend.population.base import BasePopulation
    from natal.frontend.registry.index import IndexRegistry

__all__ = ["BlueprintView", "TickContext", "TickMetrics"]


class BlueprintView:
    """Read-only blueprint: dimensions, name catalogs, and switches."""

    def __init__(
        self,
        *,
        n_sexes: int,
        n_ages: int,
        n_ztypes: int,
        discrete: bool,
        stochastic: bool,
        continuous_sampling: bool,
        extreme_speed_mode: int,
        ztype_names: Tuple[str, ...],
        gtype_names: Tuple[str, ...],
    ) -> None:
        """Bind the immutable blueprint fields."""
        self._n_sexes = n_sexes
        self._n_ages = n_ages
        self._n_ztypes = n_ztypes
        self._discrete = discrete
        self._stochastic = stochastic
        self._continuous_sampling = continuous_sampling
        self._extreme_speed_mode = extreme_speed_mode
        self._ztype_names = ztype_names
        self._gtype_names = gtype_names

    @property
    def n_sexes(self) -> int:
        """Number of sexes (state axis 0)."""
        return self._n_sexes

    @property
    def n_ages(self) -> int:
        """Number of age classes (state axis 1)."""
        return self._n_ages

    @property
    def n_ztypes(self) -> int:
        """Number of zygote types after slab expansion (state axis 2)."""
        return self._n_ztypes

    @property
    def discrete(self) -> bool:
        """Whether the population uses the discrete-generation model."""
        return self._discrete

    @property
    def stochastic(self) -> bool:
        """Whether stochastic sampling is enabled."""
        return self._stochastic

    @property
    def continuous_sampling(self) -> bool:
        """Whether continuous (Beta/Dirichlet) sampling is enabled."""
        return self._continuous_sampling

    @property
    def extreme_speed_mode(self) -> int:
        """Wright-Fisher fused-tick mode (0 = staged lifecycle)."""
        return self._extreme_speed_mode

    @property
    def ztype_names(self) -> Tuple[str, ...]:
        """Zygote-type name catalog indexed by ztype id."""
        return self._ztype_names

    @property
    def gtype_names(self) -> Tuple[str, ...]:
        """Genotype name catalog indexed by gtype id."""
        return self._gtype_names


class TickMetrics:
    """On-demand population metrics projected through the blueprint catalogs.

    Every property recomputes from the live state arrays — nothing is
    cached, so values always reflect the current tick.
    """

    def __init__(
        self,
        state: Any,
        blueprint: BlueprintView,
        species: Species,
        registry: IndexRegistry,
        config: Callable[[], ModelDraft],
    ) -> None:
        """Bind the metrics view to one state snapshot and blueprint."""
        self._state = state
        self._blueprint = blueprint
        self._species = species
        self._registry = registry
        self._config = config

    @property
    def total(self) -> float:
        """Total number of individuals."""
        return float(self._state.individual_count.sum())

    @property
    def by_sex(self) -> NDArray[np.float64]:
        """Per-sex totals with shape ``(n_sexes,)``."""
        return self._state.individual_count.sum(axis=(1, 2))

    @property
    def by_age(self) -> NDArray[np.float64]:
        """Per-age totals with shape ``(n_ages,)``."""
        return self._state.individual_count.sum(axis=(0, 2))

    @property
    def genotype_counts(self) -> dict[str, float]:
        """Total count per zygote type, keyed by catalog name."""
        ic = self._state.individual_count
        counts = ic.sum(axis=(0, 1))
        names = self._blueprint.ztype_names
        return {
            name: float(counts[idx])
            for idx, name in enumerate(names)
            if idx < len(counts)
        }

    @property
    def genotype_frequencies(self) -> dict[str, float]:
        """Zygote-type fractions of the total population.

        Zero-total populations map every name to ``0.0``.
        """
        total = self.total
        if total <= 0.0:
            return dict.fromkeys(self._blueprint.ztype_names, 0.0)
        return {name: count / total for name, count in self.genotype_counts.items()}

    @property
    def allele_frequencies(self) -> dict[str, dict[str, float]]:
        """Per-locus allele frequencies from the live genotype counts.

        Alleles are decomposed through the species: each zygote type's
        ``Genotype`` contributes its maternal/paternal allele at every
        locus, weighted by the type's total count.  Gene-duplicated allele
        names merge across chromosomes under the same key.
        """
        counts = self.genotype_counts
        loci: List[Any] = [
            locus
            for chromosome in self._species.chromosomes
            for locus in chromosome.loci
        ]
        freqs: dict[str, dict[str, float]] = {}
        for locus in loci:
            allele_counts: dict[str, float] = {}
            ztypes = self._registry.index_to_ztype
            for name, count in counts.items():
                if count <= 0.0:
                    continue
                # The catalog name is "maternal|paternal[+slab]"; resolving
                # the genotype through the registry avoids re-parsing.
                genotype = self._genotype_for_name(name, ztypes)
                if genotype is None:
                    continue
                maternal, paternal = genotype.get_alleles_at_locus(locus)
                if maternal is not None:
                    allele_counts[maternal.name] = (
                        allele_counts.get(maternal.name, 0.0) + count
                    )
                if paternal is not None:
                    allele_counts[paternal.name] = (
                        allele_counts.get(paternal.name, 0.0) + count
                    )
            locus_total = sum(allele_counts.values())
            if locus_total > 0.0:
                freqs[locus.name] = {
                    allele: count / locus_total
                    for allele, count in allele_counts.items()
                }
            else:
                freqs[locus.name] = {}
        return freqs

    def _genotype_for_name(self, name: str, ztypes: List[Tuple[Any, str]]) -> Any:
        """Resolve one catalog name to its ``Genotype`` (or ``None``).

        Catalog names carry an optional slab qualifier (``"WT|WT@slab"`` or
        ``"WT|WT:default"``); the genotype part is matched exactly.
        """
        base = re.split(r"[@:]", name, maxsplit=1)[0]
        for genotype, _slab in ztypes:
            if genotype.name == base:
                return genotype
        return None

    @property
    def c_star(self) -> float:
        """Expected competition strength implied by the *current* state.

        Recomputed on demand from the current sex-age distribution via the
        equilibrium metric kernel (the density the population feels now).
        """
        expected_comp, _surv = self._equilibrium_metrics()
        return expected_comp

    @property
    def s_star(self) -> float:
        """Expected survival rate implied by the *current* state."""
        _comp, expected_surv = self._equilibrium_metrics()
        return expected_surv

    def _equilibrium_metrics(self) -> tuple[float, float]:
        """Compute the (C*, s*) pair from the current sex-age totals."""
        from natal.frontend.data._engine import equilibrium_metrics_dispatch

        config = self._config()
        ic = self._state.individual_count
        sex_age = ic.sum(axis=2)  # (n_sexes, n_ages)
        n_ages = int(config.n_ages)
        distribution = np.zeros((2, n_ages), dtype=np.float64)
        n_rows = min(2, sex_age.shape[0])
        distribution[:n_rows, :] = sex_age[:n_rows, :n_ages]
        reproduction = (
            config.age_based_reproduction_rates
            if config.age_based_reproduction_rates is not None
            else config.age_based_mating_rates[0]
        )
        return equilibrium_metrics_dispatch(
            carrying_capacity=float(config.carrying_capacity),
            eggs_per_female=float(config.eggs_per_female),
            sex_ratio=float(config.sex_ratio),
            survival_rates=config.age_based_survival_rates,
            reproduction_rates=reproduction,
            fertility=config.female_age_based_fertility,
            competition_weights=config.age_based_relative_competition_strength,
            new_adult_age=int(config.new_adult_age),
            n_ages=n_ages,
            declared_distribution=distribution,
            external_expected_eggs=None,
        )


class TickContext:
    """The live population object handed to single-parameter hooks.

    Constructed fresh for every callback invocation by the hook runner;
    hooks must not retain a context beyond their own execution.
    """

    def __init__(
        self,
        pop: BasePopulation[Any],
        *,
        tick: int,
        deme_id: int,
        state: Any,
        hook_index: int = 0,
        transaction: EventTransaction | None = None,
    ) -> None:
        """Bind the context to one event invocation.

        Args:
            pop: The owning population.
            tick: Current simulation tick.
            deme_id: Deme index (``0`` for panmictic populations, the live
                deme index under a SpatialPopulation).
            state: The writable state view for this callback.
            hook_index: Position of this hook within its event; folds into
                the RNG stream so same-tick hooks get independent draws.
        """
        self._active = True
        self._transaction = transaction
        self._rng: HookRng | None = None
        self._pop = pop
        self._tick = tick
        self._deme_id = deme_id
        self._state = state
        self._hook_index = hook_index
        self._stop_requested = False
        self._metrics: Optional[TickMetrics] = None
        self._blueprint: Optional[BlueprintView] = None

    # -- read-only coordinates -------------------------------------------------

    @property
    def tick(self) -> int:
        """Current simulation tick."""
        return self._tick

    @property
    def deme_id(self) -> int:
        """Deme index for this invocation (``0`` panmictic)."""
        return self._deme_id

    # -- blueprint / params / state / metrics -----------------------------------

    @property
    def blueprint(self) -> BlueprintView:
        """Read-only dimensions, name catalogs, and engine switches."""
        if self._blueprint is None:
            self._blueprint = _build_blueprint(self._pop)
        return self._blueprint

    @property
    def params(self) -> ParamsView:
        """Writable parameter surface (same writer stack as ``pop.params``).

        Attribute writes are bounds-validated, reach the draft, the live
        Rust session (when present), and the parameter snapshot log.
        """
        from natal.frontend.population._params_view import ParamsView

        self.ensure_active()
        return ParamsView(self._pop, self._prepare_parameters)

    @property
    def state(self) -> Any:
        """Writable state view (short-term loan; writes are effective)."""
        if callable(self._state):
            self._state = self._state()
        return self._state

    @property
    def metrics(self) -> TickMetrics:
        """On-demand metrics, recomputed on every property access."""
        if self._metrics is None:
            pop = self._pop
            self._metrics = TickMetrics(
                self.state,
                self.blueprint,
                pop.species,
                pop.index_registry,
                lambda: pop.config,
            )
        return self._metrics

    # -- actions ----------------------------------------------------------------

    def update(self) -> GuardedConfigurator:
        """Return a runtime ``Configurator`` (same syntax as the build chain).

        Returns:
            A configurator bound to the owning population.
        """
        self._prepare_parameters()
        cfg = self._pop.update()
        cfg._hook_context = self  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]  # event-bound configurator marker
        return GuardedConfigurator(cfg, self.ensure_active)

    def stop(self) -> None:
        """Request termination of the current run at the event boundary."""
        self.ensure_active()
        self._stop_requested = True

    @property
    def stop_requested(self) -> bool:
        """Whether :meth:`stop` was called on this context."""
        return self._stop_requested

    def ensure_active(self) -> None:
        """Reject every retained writer or sampler after callback completion."""
        if not self._active:
            raise RuntimeError("Hook context has expired")

    def _prepare_parameters(self) -> None:
        """Materialize the isolated parameter candidate only when requested."""
        self.ensure_active()
        prepare = getattr(self._pop, "_event_prepare_config", None)
        if prepare is not None:
            prepare()

    def invalidate(self) -> None:
        """Detach every writer and sampler when this callback ends."""
        self._active = False
        # The sampler's lifetime guard is a bound method. Break the cycle
        # without disabling guards on samplers explicitly retained by users.
        self._rng = None

    @property
    def rng(self) -> HookRng:
        """Return the same controlled Rust sampler throughout this callback."""
        self.ensure_active()
        if self._rng is None:
            if self._transaction is None:
                raise RuntimeError("This event has no native RNG transaction")
            self._rng = HookRng(self._transaction, self.ensure_active)
        return self._rng


def _build_blueprint(pop: BasePopulation[Any]) -> BlueprintView:
    """Project a population onto the read-only blueprint view."""
    # These dimensions and execution flags are fixed during a callback;
    # querying them must not materialize unrelated native parameter tensors.
    config = pop._config  # pyright: ignore[reportPrivateUsage]
    assert config is not None
    ic = pop._state.individual_count  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]  # shape probe only: shape is invariant and probing must never trigger the session pull (callbacks run inside the session borrow)
    discrete = bool(getattr(config, "discrete_generation", False))
    return BlueprintView(
        n_sexes=int(ic.shape[0]),
        n_ages=int(ic.shape[1]) if ic.ndim == 3 else 1,
        n_ztypes=int(ic.shape[-1]),
        discrete=discrete,
        stochastic=bool(config.stochastic),
        continuous_sampling=bool(config.continuous_sampling),
        extreme_speed_mode=int(getattr(config, "extreme_speed_mode", 0)),
        ztype_names=tuple(_catalog(config, "ztype_names", pop, "ztype")),
        gtype_names=tuple(_catalog(config, "gtype_names", pop, "gtype")),
    )


def _catalog(
    config: ModelDraft, field: str, pop: BasePopulation[Any], kind: str
) -> List[str]:
    """Resolve a name catalog from the draft or the registry fallback."""
    from natal.contracts.materialize import (
        gtype_names_from_registry,
        ztype_names_from_registry,
    )

    names: object = getattr(config, field, None)
    if isinstance(names, (list, tuple)):
        seq = [str(name) for name in cast("Sequence[object]", names)]
        if len(seq) > 0:
            return seq
    registry = pop.index_registry
    if kind == "ztype":
        return list(ztype_names_from_registry(registry.index_to_ztype))
    return list(gtype_names_from_registry(registry.index_to_gtype))


def state_view_for(
    pop: BasePopulation[Any],
    *,
    tick: int,
    ind_flat: NDArray[np.float64],
    sperm_flat: Optional[NDArray[np.float64]],
) -> Any:
    """Wrap flat engine arrays into the population's state NamedTuple shape.

    No data is copied: the returned state borrows the caller's memory.
    Inside engine ticks the flat arrays are per-callback copies that Rust
    writes back after the call; out-of-band callers (``trigger_event``,
    finish events) pass their own live arrays.

    Args:
        pop: The owning population (decides discrete vs structured shape).
        tick: Current tick.
        ind_flat: Flat individual counts ``(n_sexes*n_ages*n_ztypes,)``.
        sperm_flat: Flat sperm storage or ``None`` for discrete models.

    Returns:
        A ``PopulationState`` or ``DiscretePopulationState`` view.
    """
    from natal.frontend.data import (
        DiscretePopulationState,
        PopulationState,
    )

    live = pop._state.individual_count  # pyright: ignore[reportPrivateUsage, reportOptionalMemberAccess]  # shape probe only: same invariance argument as _build_blueprint
    n_sexes = int(live.shape[0])
    n_ages = int(live.shape[1])
    n_ztypes = int(live.shape[2])
    ind = ind_flat.reshape(n_sexes, n_ages, n_ztypes)
    if sperm_flat is None or sperm_flat.size == 0:
        return DiscretePopulationState(n_tick=tick, individual_count=ind)
    sperm = sperm_flat.reshape(n_ages, n_ztypes, n_ztypes)
    return PopulationState(n_tick=tick, individual_count=ind, sperm_storage=sperm)


class HookRunner:
    """Dispatches single-parameter callbacks for one population.

    The runner holds the callback descriptors grouped per event (sorted by
    priority) and materializes a :class:`TickContext` per callback.  It is
    the single Python-side callback dispatch path:

    - out-of-band: called directly from ``trigger_event`` and the
      finish-event executor.
    - in-tick: adapted into the ``(ind, sperm, tick, deme_id)`` callback
      signature and fired by the engine at event boundaries.
    """

    def __init__(self, pop: BasePopulation[Any]) -> None:
        """Bind the runner to a population and index its callbacks.

        Args:
            pop: The population whose compiled descriptors provide the
                callbacks.
        """
        from natal.frontend.hooks.types import (
            EVENT_ID_MAP,
            EVENT_NAMES,
        )

        self._pop = pop
        self._callbacks: dict[
            int, List[Tuple[int, Callable[[TickContext], Optional[int]], Any]]
        ] = {event_id: [] for event_id in range(len(EVENT_NAMES))}
        for desc in pop.compiled_hook_descriptors:
            if desc.callback is None:
                continue
            event_id = EVENT_ID_MAP.get(desc.event)
            if event_id is None:
                continue
            self._callbacks[event_id].append(
                (int(desc.priority), desc.callback, desc.deme_selector)
            )
        for entries in self._callbacks.values():
            entries.sort(key=lambda item: item[0])

    def has_callbacks(self) -> bool:
        """Return whether any event carries a Python callback."""
        return any(len(entries) > 0 for entries in self._callbacks.values())

    def _matches(self, selector: Any, deme_id: int) -> bool:
        """Evaluate one deme selector (``*``, int, range, or collection)."""
        if selector == "*":
            return True
        if isinstance(selector, int):
            return selector == deme_id
        if isinstance(selector, range):
            return deme_id in selector
        return deme_id in selector

    def run_event(
        self,
        event_id: int,
        *,
        tick: int,
        deme_id: int,
        state: Any,
        transaction: EventTransaction | None = None,
        only_index: int | None = None,
    ) -> int:
        """Run one event's callbacks in priority order.

        Args:
            event_id: Numeric event id (0 first, 1 early, 2 late, 3 finish).
            tick: Current tick.
            deme_id: Deme index for selector filtering.
            state: The state view handed to every callback.

        Returns:
            ``0`` to continue, ``1`` when a callback returned nonzero or
            called :meth:`TickContext.stop`.
        """
        for hook_index, (_priority, callback, selector) in enumerate(
            self._callbacks.get(event_id, [])
        ):
            if (only_index is not None and hook_index != only_index) or not self._matches(selector, deme_id):
                continue
            context = TickContext(
                self._pop,
                tick=tick,
                deme_id=deme_id,
                state=state,
                hook_index=hook_index,
                transaction=transaction,
            )
            try:
                result = callback(context)
            finally:
                context.invalidate()
            if result is not None and int(result) != 0:
                return RESULT_STOP
            if context.stop_requested:
                return RESULT_STOP
        return 0

    def rust_callbacks(self, event_id: int) -> list[Callable[..., int]]:
        """Give each Python callback its own native atomic commit boundary."""
        callbacks: list[Callable[..., int]] = []
        for index in range(len(self._callbacks.get(event_id, []))):
            callback = self.rust_callback(event_id, only_index=index)
            if callback is not None:
                callbacks.append(callback)
        return callbacks

    def rust_callback(self, event_id: int, only_index: int | None = None) -> Optional[Callable[..., int]]:
        """Build the Rust-bridge callback for one event, or ``None``.

        The native ABI passes two empty state placeholders, event coordinates,
        and an owned transaction. State arrays materialize only when requested
        through the context; they commit together with other candidates.
        """
        entries = self._callbacks.get(event_id, [])
        if not entries:
            return None

        def bridge(
            ind: NDArray[np.float64] | None,
            sperm: NDArray[np.float64] | None,
            tick: int,
            deme_id: int,
            transaction: EventTransaction,
        ) -> int:
            """Adapt the Rust callback ABI to :meth:`run_event`."""
            pop = self._pop
            original = pop._config  # pyright: ignore[reportPrivateUsage]  # candidate scope owns this temporary binding
            metadata_names = (
                "_current_definition", "_presets", "_manual_gamete", "_manual_zygote",
                "_gamete_modifiers", "_zygote_modifiers", "_reconfiguration_log",
            )
            original_metadata: dict[str, object] = {}
            original_log = pop._params_log  # pyright: ignore[reportPrivateUsage]  # failed event logs are rolled back
            original_active = getattr(pop, "_rust_run_active", False)
            pop._rust_run_active = True  # pyright: ignore[reportAttributeAccessIssue]  # native callback holds the session borrow
            original_tick = pop._tick  # pyright: ignore[reportPrivateUsage]
            pop._tick = tick  # pyright: ignore[reportPrivateUsage]  # event logs use the actual native tick
            pop._event_transaction = transaction  # pyright: ignore[reportAttributeAccessIssue]  # event-local native candidate
            prepared = False
            rollback_actions: list[Callable[[], None]] | None = None

            def prepare_config() -> None:
                """Defer Python configuration ownership until a callback needs it."""
                nonlocal prepared, rollback_actions
                if prepared:
                    return
                from natal.backends.rust.rust_backend import (
                    config_snapshot_from_session,
                )

                assert original is not None
                candidate = config_snapshot_from_session(transaction, original)
                for name in metadata_names:
                    if hasattr(pop, name):
                        original_metadata[name] = getattr(pop, name)
                        setattr(pop, name, copy.copy(original_metadata[name]) if name != "_current_definition" else original_metadata[name])
                pop._params_log = type(original_log)()  # pyright: ignore[reportPrivateUsage]  # allocate a log only for accessed parameter candidates
                pop._config = candidate  # pyright: ignore[reportPrivateUsage]
                rollback_actions = []
                pop._event_rollback_actions = rollback_actions  # pyright: ignore[reportAttributeAccessIssue]  # NATAL-managed recipe updates join this event's rollback.
                prepared = True

            pop._event_prepare_config = prepare_config  # pyright: ignore[reportAttributeAccessIssue]  # callback-local lazy projection used by public config reads
            try:
                def state_factory() -> Any:
                    # Any: the two model-specific state NamedTuples share this lazy boundary.
                    ind_values, sperm_values = transaction.state_arrays()
                    return state_view_for(pop, tick=int(tick), ind_flat=ind_values, sperm_flat=sperm_values if sperm_values.size else None)

                result = self.run_event(event_id, tick=int(tick), deme_id=int(deme_id), state=state_factory, transaction=transaction, only_index=only_index)
                transaction.validate_state()
                # Every validated writer has already submitted its changed fields
                # to the native candidate. Read-only callbacks need no return trip.
                if prepared:
                    from natal.frontend.hooks.types import EVENT_NAMES
                    for change_tick, _event, _deme, name, old, new in pop._params_log.details():  # pyright: ignore[reportPrivateUsage]
                        original_log.append_value(change_tick, name, old, new, EVENT_NAMES[event_id], int(deme_id))
                return result
            except BaseException:
                for rollback in reversed(rollback_actions or ()):
                    rollback()
                pop._config = original  # pyright: ignore[reportPrivateUsage]
                for name in metadata_names if prepared else ():
                    if name in original_metadata:
                        setattr(pop, name, original_metadata[name])
                    elif hasattr(pop, name):
                        delattr(pop, name)
                raise
            finally:
                pop._params_log = original_log  # pyright: ignore[reportPrivateUsage]
                pop._rust_run_active = original_active  # pyright: ignore[reportAttributeAccessIssue]
                pop._tick = original_tick  # pyright: ignore[reportPrivateUsage]
                pop._event_rollback_actions = None  # pyright: ignore[reportAttributeAccessIssue]
                pop._event_transaction = None  # pyright: ignore[reportAttributeAccessIssue]
                pop._event_prepare_config = None  # pyright: ignore[reportAttributeAccessIssue]

        bridge.__natal_transaction__ = True  # pyright: ignore[reportFunctionMemberAccess]  # native bridge ABI discriminator

        return bridge
