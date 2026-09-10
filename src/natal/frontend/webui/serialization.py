"""Typed JSON serialization for the NATAL web UI API layer.

Every function here is a pure read over the population object: no engine
mutation, no UI framework imports.  The Vue dashboard consumes exactly the
TypedDict shapes defined in this module, so the shapes double as the API
contract documentation.

Display rounding (integers for discrete models, significant digits for
continuous ones) is deliberately *not* applied here — the server returns raw
floats and the frontend formats for display.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    List,
    Protocol,
    Union,
    cast,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypedDict

from natal.frontend.data.state import DiscretePopulationState, PopulationState
from natal.frontend.genetics.structures.species import Species
from natal.frontend.model.draft import ModelDraft
from natal.frontend.modifiers.module import GameteModifier, ZygoteModifier
from natal.frontend.output.history import History
from natal.frontend.population.age_structured import AgeStructuredPopulation
from natal.frontend.population.base import BasePopulation
from natal.frontend.registry.index import IndexRegistry

if TYPE_CHECKING:
    from natal.frontend.genetics import Genotype
    from natal.frontend.hooks.types import CompiledHookDescriptor

#: The panmictic population types the API layer fully supports in Phase 1-2.
PanmicticPopulation = Union[
    BasePopulation[PopulationState],
    BasePopulation[DiscretePopulationState],
]

PanmicticState = Union[PopulationState, DiscretePopulationState]


@runtime_checkable
class GeneticStructureLike(Protocol):
    """Structural view of anything exposing the genetic dashboard surface.

    Satisfied by panmictic populations and by ``DemeSlice`` (which forwards
    genetic reads to its underlying deme), letting the config / hooks /
    genetics / registry endpoints serve both dashboard kinds through one
    serializer.
    """

    @property
    def registry(self) -> IndexRegistry: ...

    @property
    def config(self) -> ModelDraft: ...

    @property
    def species(self) -> Species: ...

    @property
    def name(self) -> str: ...

    @property
    def gamete_modifiers(self) -> List[tuple[int, str | None, GameteModifier]]: ...

    @property
    def zygote_modifiers(self) -> List[tuple[int, str | None, ZygoteModifier]]: ...

    def get_compiled_hooks(
        self, event: str | None = None
    ) -> list[CompiledHookDescriptor]: ...


# ---------------------------------------------------------------------------
# State snapshots
# ---------------------------------------------------------------------------


class SpermEntry(TypedDict):
    """One nonzero sperm-storage entry (age-structured populations only)."""

    age: int
    female_index: int
    male_index: int
    female_label: str
    male_label: str
    value: float


class GenotypeStateRow(TypedDict):
    """Per-genotype aggregation of one state snapshot."""

    index: int
    label: str
    ztype_indices: list[int]
    female: float
    male: float
    total: float
    female_per_age: list[float]
    male_per_age: list[float]
    viability: list[float]
    fecundity: list[float]


class StateSnapshot(TypedDict):
    """Full inspection snapshot: live state or one raw-history record."""

    tick: int
    mode: str  # "live" | "history"
    found: bool
    is_age_structured: bool
    total: float
    female: float
    male: float
    female_per_age: list[float]
    male_per_age: list[float]
    genotypes: list[GenotypeStateRow]
    sperm_storage: list[SpermEntry] | None
    history_len: int


def genotype_rows_builder(
    registry: IndexRegistry,
    config: ModelDraft,
    counts: NDArray[np.float64],
) -> list[GenotypeStateRow]:
    """Aggregate a ``(sex, age, ztype)`` tensor into per-genotype rows.

    Shared by the panmictic state snapshot and the spatial deme detail.
    """
    target_age = max(0, int(config.new_adult_age) - 1)
    female_axis = counts[0]
    male_axis = counts[1]

    genotype_rows: list[GenotypeStateRow] = []
    for g_idx, gt in enumerate(registry.index_to_genotype):
        z_indices = registry.ztype_indices_for(gt)
        f_by_age = female_axis[:, z_indices].sum(axis=1)
        m_by_age = male_axis[:, z_indices].sum(axis=1)
        v_f = float(config.viability_fitness[0, target_age, z_indices[0]])
        v_m = float(config.viability_fitness[1, target_age, z_indices[0]])
        fec_f = float(config.fecundity_fitness[0, z_indices[0]])
        fec_m = float(config.fecundity_fitness[1, z_indices[0]])
        genotype_rows.append(
            GenotypeStateRow(
                index=g_idx,
                label=str(gt),
                ztype_indices=z_indices,
                female=float(f_by_age.sum()),
                male=float(m_by_age.sum()),
                total=float(f_by_age.sum() + m_by_age.sum()),
                female_per_age=f_by_age.tolist(),
                male_per_age=m_by_age.tolist(),
                viability=[v_f, v_m],
                fecundity=[fec_f, fec_m],
            )
        )
    return genotype_rows


def serialize_state(
    population: PanmicticPopulation,
    state: PanmicticState,
    *,
    mode: str,
    found: bool,
) -> StateSnapshot:
    """Build a full inspection snapshot from *state*."""
    counts = state.individual_count

    genotype_rows = genotype_rows_builder(population.registry, population.config, counts)

    sperm: list[SpermEntry] | None = None
    if isinstance(state, PopulationState):
        sperm = _serialize_sperm(state.sperm_storage, population.registry)

    return StateSnapshot(
        tick=int(state.n_tick),
        mode=mode,
        found=found,
        is_age_structured=isinstance(population, AgeStructuredPopulation),
        total=float(counts.sum()),
        female=float(counts[0].sum()),
        male=float(counts[1].sum()),
        female_per_age=counts[0].sum(axis=1).tolist(),
        male_per_age=counts[1].sum(axis=1).tolist(),
        genotypes=genotype_rows,
        sperm_storage=sperm,
        history_len=len(population.history),
    )


def _serialize_sperm(
    sperm: NDArray[np.float64], registry: IndexRegistry
) -> list[SpermEntry]:
    """Serialize nonzero sperm-storage entries in sparse form.

    Args:
        sperm: ``(age, female_genotype, male_genotype)`` value tensor.
        registry: Index registry providing genotype labels.

    Returns:
        Sparse entries, one per nonzero (age, female, male) cell.
    """
    genotypes = [str(g) for g in registry.index_to_genotype]
    n_genotypes = len(genotypes)
    entries: list[SpermEntry] = []
    for age in range(sperm.shape[0]):
        for f_idx, m_idx in np.argwhere(sperm[age] > 0):
            fi = int(f_idx)
            mi = int(m_idx)
            entries.append(
                SpermEntry(
                    age=age,
                    female_index=fi,
                    male_index=mi,
                    female_label=genotypes[fi] if fi < n_genotypes else str(fi),
                    male_label=genotypes[mi] if mi < n_genotypes else str(mi),
                    value=float(sperm[age, fi, mi]),
                )
            )
    return entries


def state_at_tick(
    population: PanmicticPopulation, tick: int | None
) -> StateSnapshot:
    """Serialize the live state, or the raw-history record at *tick*.

    Args:
        population: Panmictic population under inspection.
        tick: Target tick, or ``None`` for the live state.

    Returns:
        The snapshot; when *tick* is not present in raw history the returned
        snapshot has ``found=False`` and carries the live state values.
    """
    if tick is None or tick == population.tick:
        return serialize_state(population, population.state, mode="live", found=True)
    history = population.history
    if history.schema.mode == "raw" and tick in history.ticks:
        state = raw_history_state(
            population.history, history.ticks.index(tick)
        )
        return serialize_state(population, state, mode="history", found=True)
    return serialize_state(population, population.state, mode="live", found=False)


def raw_history_state(
    history: History,
    index: int,
) -> PanmicticState:
    """Rebuild one typed state from a raw History record.

    Args:
        history: Raw-mode history to read from.
        index: Position along the record axis.

    Returns:
        The stored state (sperm storage included when recorded).
    """
    tick = history.ticks[index]
    counts = history.individual_count[index]
    sperm = history.sperm_storage
    if sperm is None:
        return DiscretePopulationState(tick, counts)
    return PopulationState(tick, counts, sperm[index])


# ---------------------------------------------------------------------------
# Allele frequencies
# ---------------------------------------------------------------------------


def known_allele_names(species: Species) -> list[str]:
    """Return every allele name declared by *species*, in stable order."""
    names: list[str] = []
    seen: set[str] = set()
    for chrom in species.chromosomes:
        for locus in chrom.loci:
            for gene in locus.alleles:
                if gene.name not in seen:
                    seen.add(gene.name)
                    names.append(gene.name)
    return names


def compute_allele_frequencies(
    registry: IndexRegistry,
    species: Species,
    counts: NDArray[np.float64],
) -> dict[str, float]:
    """Compute per-allele frequencies from a (sex, age, ztype) count tensor.

    Each individual contributes one gene copy per homolog, so both alleles
    of every genotype count toward their locus total (diploid semantics,
    matching ``BasePopulation.compute_allele_frequencies``).

    Args:
        registry: Index registry whose ztype order matches ``counts``.
        species: Species providing chromosome/locus/allele structure.
        counts: ``(sex, age, ztype)`` count tensor.

    Returns:
        Frequency per allele name; alleles absent from the population are
        omitted (callers fill zeros for known alleles).
    """
    genotype_counts = counts.sum(axis=(0, 1))
    allele_counts: dict[str, float] = {}
    locus_totals: dict[str, float] = {}

    for z_idx, (gt, _slab) in enumerate(registry.index_to_ztype):
        count = float(genotype_counts[z_idx])
        if count <= 0:
            continue
        for chrom in species.chromosomes:
            for locus in chrom.loci:
                mat, pat = gt.get_alleles_at_locus(locus)
                if mat is not None:
                    allele_counts[mat.name] = allele_counts.get(mat.name, 0.0) + count
                    locus_totals[locus.name] = locus_totals.get(locus.name, 0.0) + count
                if pat is not None:
                    allele_counts[pat.name] = allele_counts.get(pat.name, 0.0) + count
                    locus_totals[locus.name] = locus_totals.get(locus.name, 0.0) + count

    frequencies: dict[str, float] = {}
    gene_index = species.gene_index
    for allele_name, allele_total in allele_counts.items():
        gene = gene_index.get(allele_name)
        if gene is None:
            continue
        total = locus_totals.get(gene.locus.name, 0.0)
        if total > 0:
            frequencies[allele_name] = allele_total / total
    return frequencies


# ---------------------------------------------------------------------------
# History series (chart data)
# ---------------------------------------------------------------------------


class HistorySeries(TypedDict):
    """Downsampled time series for the dashboard charts."""

    ticks: list[int]
    total: list[float]
    female: list[float]
    male: list[float]
    known_alleles: list[str]
    allele_frequencies: dict[str, list[float]]
    truncated: bool


def history_series(
    population: PanmicticPopulation,
    *,
    max_points: int = 500,
    tick_from: int | None = None,
    tick_to: int | None = None,
) -> HistorySeries:
    """Build downsampled chart series from raw history.

    Args:
        population: Panmictic population to read.
        max_points: Target maximum number of samples (stride downsample).
        tick_from: Optional lower tick bound (inclusive).
        tick_to: Optional upper tick bound (inclusive).

    Returns:
        Aligned series arrays; ``truncated`` flags stride sampling.  Allele
        frequencies are zero-filled for every known allele so series stay
        aligned across the whole timeline.
    """
    history = population.history
    ticks: list[int] = []
    rows: list[NDArray[np.float64]] = []  # count tensor per collected point

    if history.schema.mode == "raw":
        counts_all = history.individual_count
        for index, tick in enumerate(history.ticks):
            if tick_from is not None and tick < tick_from:
                continue
            if tick_to is not None and tick > tick_to:
                continue
            ticks.append(int(tick))
            rows.append(counts_all[index])

    # Include the live state when not already recorded (record_every > 1 or
    # an uncommitted tick).
    if population.tick not in ticks:
        if (tick_to is None or population.tick <= tick_to) and (
            tick_from is None or population.tick >= tick_from
        ):
            ticks.append(int(population.tick))
            rows.append(population.state.individual_count)

    n = len(ticks)
    stride = max(1, n // max(1, max_points))
    kept_ticks = ticks[::stride]
    alleles = known_allele_names(population.species)

    registry = population.registry
    freq_by_allele: dict[str, list[float]] = {name: [] for name in alleles}
    for kept_index in range(len(kept_ticks)):
        row = rows[kept_index * stride]
        freqs = compute_allele_frequencies(registry, population.species, row)
        for name in alleles:
            freq_by_allele[name].append(freqs.get(name, 0.0))

    totals = [float(row.sum()) for row in rows]
    females = [float(row[0].sum()) for row in rows]
    males = [float(row[1].sum()) for row in rows]

    return HistorySeries(
        ticks=kept_ticks,
        total=totals[::stride],
        female=females[::stride],
        male=males[::stride],
        known_alleles=alleles,
        allele_frequencies=freq_by_allele,
        truncated=stride > 1,
    )


# ---------------------------------------------------------------------------
# Configuration, fitness, presets
# ---------------------------------------------------------------------------


class GrowthModeInfo(TypedDict):
    code: int
    name: str


class ConfigScalars(TypedDict):
    """Scalar model parameters shown in the config panel."""

    population_name: str
    stochastic: bool
    continuous_sampling: bool
    discrete_generation: bool
    extreme_speed_mode: int
    n_sexes: int
    n_ages: int
    n_genotypes: int
    n_gtypes: int
    n_glabs: int
    n_slabs: int
    new_adult_age: int
    carrying_capacity: float
    eggs_per_female: float
    sex_ratio: float
    sperm_displacement_rate: float
    low_density_growth_rate: float
    expected_competition_strength: float
    expected_survival_rate: float
    generation_time: float
    fixed_egg_count: bool
    juvenile_growth_mode: GrowthModeInfo


class FitnessRow(TypedDict):
    genotype: str
    age: float | None
    female: float
    male: float


class SexualSelectionRow(TypedDict):
    female_genotype: str
    male_genotype: str
    preference: float


class PresetModifierItem(TypedDict):
    id: int
    name: str
    kind: str


class PresetInfo(TypedDict):
    preset_name: str
    gamete_modifiers: list[PresetModifierItem]
    zygote_modifiers: list[PresetModifierItem]


class PresetsSummary(TypedDict):
    preset_count: int
    presets: list[PresetInfo]


class ConfigPayload(TypedDict):
    scalars: ConfigScalars
    full: dict[str, object]  # object: heterogeneous full ModelDraft dump
    presets: PresetsSummary
    viability: list[FitnessRow]
    fecundity: list[FitnessRow]
    sexual_selection: list[SexualSelectionRow]


def _growth_mode_name(mode: int) -> str:
    """Map a numeric growth mode constant to its name."""
    from natal.frontend.model import BEVERTON_HOLT, FIXED, LINEAR, NO_COMPETITION

    mapping = {
        NO_COMPETITION: "NO_COMPETITION",
        FIXED: "FIXED",
        LINEAR: "LOGISTIC",
        BEVERTON_HOLT: "BEVERTON_HOLT",
    }
    return mapping.get(int(mode), f"UNKNOWN_{mode}")


def to_jsonable(value: object) -> object:  # object: accepts arbitrary config-draft values, returns JSON-native equivalents
    """Recursively convert numpy containers to plain Python for JSON."""
    if isinstance(value, np.generic):
        # Every numpy scalar (bool_/integer/floating) via its typed base.
        # item()'s return is statically unknown, hence the justified cast.
        return cast("object", value.item())  # cast: item() is statically unknown but always a Python scalar
    if isinstance(value, np.ndarray):
        # Elements are numpy scalars; normalize them recursively.  The cast
        # is required because ndarray element types cannot be proven static.
        items = cast("list[object]", value.tolist())  # cast: ndarray element types cannot be proven statically
        return [to_jsonable(item) for item in items]
    if isinstance(value, (list, tuple)):
        return [  # cast: list/tuple element types cannot be proven statically
            to_jsonable(item) for item in cast("Iterable[object]", value)
        ]
    if isinstance(value, dict):
        return {
            key: to_jsonable(item)
            for key, item in cast(  # cast: dict value types cannot be proven statically
                "dict[object, object]", value
            ).items()
        }
    return value


def config_payload(population: GeneticStructureLike) -> ConfigPayload:
    """Serialize scalar parameters, fitness tables, and preset summary."""
    from natal.frontend.model.ecology import derive_equilibrium_metrics_from_draft

    config = population.config
    registry = population.registry
    genotypes = registry.index_to_genotype
    growth_mode = int(config.juvenile_growth_mode)
    expected_competition_strength, expected_survival_rate = (
        derive_equilibrium_metrics_from_draft(config)
    )

    scalars = ConfigScalars(
        population_name=population.name,
        stochastic=bool(config.stochastic),
        continuous_sampling=bool(config.continuous_sampling),
        discrete_generation=bool(config.discrete_generation),
        extreme_speed_mode=int(config.extreme_speed_mode),
        n_sexes=int(config.n_sexes),
        n_ages=int(config.n_ages),
        n_genotypes=int(config.n_ztypes),
        n_gtypes=int(config.n_gtypes),
        n_glabs=int(config.n_glabs),
        n_slabs=int(config.n_slabs),
        new_adult_age=int(config.new_adult_age),
        carrying_capacity=float(config.carrying_capacity),
        eggs_per_female=float(config.eggs_per_female),
        sex_ratio=float(config.sex_ratio),
        sperm_displacement_rate=float(config.sperm_displacement_rate),
        low_density_growth_rate=float(config.low_density_growth_rate),
        expected_competition_strength=float(expected_competition_strength),
        expected_survival_rate=float(expected_survival_rate),
        generation_time=float(config.generation_time),
        fixed_egg_count=bool(config.fixed_egg_count),
        juvenile_growth_mode=GrowthModeInfo(
            code=growth_mode, name=_growth_mode_name(growth_mode)
        ),
    )

    target_age = max(0, int(config.new_adult_age) - 1)
    viability: list[FitnessRow] = []
    fecundity: list[FitnessRow] = []
    for z_idx, (gt, slab) in enumerate(registry.index_to_ztype):
        # Show only the first slab per genotype to avoid duplicate rows.
        if slab != registry.slab_labels[0]:
            continue
        v_f = float(config.viability_fitness[0, target_age, z_idx])
        v_m = float(config.viability_fitness[1, target_age, z_idx])
        f_f = float(config.fecundity_fitness[0, z_idx])
        f_m = float(config.fecundity_fitness[1, z_idx])
        label = str(gt)
        drive_relevant = "Dr" in label or "Drive" in label
        if v_f != 1.0 or v_m != 1.0 or drive_relevant:
            viability.append(
                FitnessRow(genotype=label, age=float(target_age), female=v_f, male=v_m)
            )
        if f_f != 1.0 or f_m != 1.0 or drive_relevant:
            fecundity.append(
                FitnessRow(genotype=label, age=None, female=f_f, male=f_m)
            )

    sexual_selection: list[SexualSelectionRow] = []
    for f_idx, f_gt in enumerate(genotypes):
        for m_idx, m_gt in enumerate(genotypes):
            pref = float(config.sexual_selection_fitness[f_idx, m_idx])
            if pref != 1.0:
                sexual_selection.append(
                    SexualSelectionRow(
                        female_genotype=str(f_gt),
                        male_genotype=str(m_gt),
                        preference=pref,
                    )
                )

    full: dict[str, object] = {}  # object: heterogeneous full ModelDraft dump
    for key, value in config._asdict().items():
        full[key] = to_jsonable(value)
    full["juvenile_growth_mode_name"] = _growth_mode_name(growth_mode)

    return ConfigPayload(
        scalars=scalars,
        full=full,
        presets=presets_summary(population),
        viability=viability,
        fecundity=fecundity,
        sexual_selection=sexual_selection,
    )


def presets_summary(population: GeneticStructureLike) -> PresetsSummary:
    """Best-effort preset-centric summary of registered genetic modifiers."""
    preset_map: dict[str, PresetInfo] = {}

    def record_modifier(
        mod_type: str,
        mod_tuple: tuple[int, str | None, object],  # object: modifier tuples carry an opaque third element (the modifier entity)
    ) -> None:
        mod_id, mod_name, _ = mod_tuple
        name = mod_name if mod_name else f"{mod_type}_{mod_id}"
        if "/" in name:
            preset_name, suffix = name.split("/", 1)
        else:
            preset_name, suffix = name, mod_type
        info = preset_map.setdefault(
            preset_name,
            PresetInfo(
                preset_name=preset_name,
                gamete_modifiers=[],
                zygote_modifiers=[],
            ),
        )
        item = PresetModifierItem(id=int(mod_id), name=name, kind=suffix)
        if mod_type == "gamete":
            info["gamete_modifiers"].append(item)
        else:
            info["zygote_modifiers"].append(item)

    for mod in population.gamete_modifiers:
        record_modifier("gamete", mod)
    for mod in population.zygote_modifiers:
        record_modifier("zygote", mod)

    return PresetsSummary(
        preset_count=len(preset_map), presets=list(preset_map.values())
    )


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


class HookOpInfo(TypedDict):
    type: str
    genotypes: object  # object: mirrors HookOp.genotypes (str, list[str], or "*")
    ages: object  # object: mirrors HookOp.ages (int, list[int], range, or "*")
    sex: str
    param: object  # object: mirrors HookOp.param (float; symbolic forms possible)
    condition: str | None


class HookInfo(TypedDict):
    event: str
    name: str
    priority: int
    kind: str  # "declarative" | "callback"
    operations: list[HookOpInfo] | None
    signature: str | None
    source: str | None


def hooks_payload(population: GeneticStructureLike) -> list[HookInfo]:
    """Serialize compiled hook descriptors for the hooks panel and export."""
    import inspect

    from natal.frontend.hooks.entry.declarative import OpType

    op_type_names = {
        OpType.SCALE: "scale",
        OpType.SET: "set_count",
        OpType.ADD: "add",
        OpType.SUBTRACT: "subtract",
        OpType.KILL: "kill",
        OpType.SAMPLE: "sample",
        OpType.STOP_IF_ZERO: "stop_if_zero",
        OpType.STOP_IF_BELOW: "stop_if_below",
        OpType.STOP_IF_ABOVE: "stop_if_above",
        OpType.STOP_IF_EXTINCTION: "stop_if_extinction",
    }

    def op_type_name(op_type: OpType) -> str:
        return op_type_names.get(op_type, op_type.name.lower())

    def ages_value(ages: int | list[int] | range | str) -> object:  # object: JSON-ready form may be number, list, or "*"
        if isinstance(ages, range):
            return [float(a) for a in ages]
        if isinstance(ages, list):
            return [float(a) for a in ages]
        if isinstance(ages, int):
            return float(ages)
        return ages

    payload: list[HookInfo] = []
    descriptors: list[CompiledHookDescriptor] = list(
        population.get_compiled_hooks()
    )
    for desc in descriptors:
        operations: list[HookOpInfo] | None = None
        signature: str | None = None
        source: str | None = None
        if desc.ops:
            kind = "declarative"
            operations = []
            for op in desc.ops:
                operations.append(
                    HookOpInfo(
                        type=op_type_name(op.op_type),
                        genotypes=op.genotypes,
                        ages=ages_value(op.ages),
                        sex=op.sex,
                        param=op.param,
                        condition=op.condition,
                    )
                )
        elif desc.callback is not None:
            kind = "callback"
            try:
                signature = str(inspect.signature(desc.callback))
            except (ValueError, TypeError):
                signature = None
            try:
                source = inspect.getsource(desc.callback)
            except (OSError, TypeError):
                source = None
        else:
            kind = "compiled"
        payload.append(
            HookInfo(
                event=str(desc.event),
                name=desc.name,
                priority=int(desc.priority),
                kind=kind,
                operations=operations,
                signature=signature,
                source=source,
            )
        )
    return payload


# ---------------------------------------------------------------------------
# Genetics matrices
# ---------------------------------------------------------------------------


class Matrix2D(TypedDict):
    row_labels: list[str]
    col_labels: list[str]
    data: list[list[float]]


class FertilizationMatrix(TypedDict):
    row_labels: list[str]
    col_labels: list[str]
    zygote_labels: list[str]
    primary_index: list[list[float]]  # NaN when the pair produces nothing
    primary_probability: list[list[float]]  # probability of the primary zygote; NaN when empty
    cell_text: list[list[str]]
    too_large: bool


class GeneticsPayload(TypedDict):
    meiosis: list[Matrix2D]  # one per sex: female first
    fertilization: FertilizationMatrix


def genetics_matrices(population: GeneticStructureLike) -> GeneticsPayload:
    """Serialize meiosis and fertilization probability structures.

    The fertilization matrix colors cells by their most probable offspring
    zygote and annotates each cell with every outcome above 1% — the same
    presentation the legacy Plotly view used.
    """
    config = population.config
    registry = population.registry
    genotypes = registry.index_to_genotype

    row_labels = [str(g) for g in genotypes]
    col_labels: list[str] = []
    for hg_obj, glab_str in registry.index_to_gtype:
        label = str(hg_obj)
        if int(config.n_glabs) > 1:
            label = f"{label} [{glab_str}]"
        col_labels.append(label)

    meiosis: list[Matrix2D] = []
    z2g = config.zygotes_to_gametes_map
    for sex_idx in range(int(config.n_sexes)):
        meiosis.append(
            Matrix2D(
                row_labels=row_labels,
                col_labels=col_labels,
                data=z2g[sex_idx].tolist(),
            )
        )

    n_gametes = int(config.n_gtypes)
    g2z = config.gametes_to_zygotes_map
    too_large = n_gametes > 40
    primary: list[list[float]] = []
    probability: list[list[float]] = []
    cell_text: list[list[str]] = []
    if not too_large:
        for r in range(n_gametes):
            primary_row: list[float] = []
            probability_row: list[float] = []
            text_row: list[str] = []
            for c in range(n_gametes):
                probs = g2z[r, c, :]
                if float(probs.sum()) < 1e-9:
                    primary_row.append(float("nan"))
                    probability_row.append(float("nan"))
                    text_row.append("")
                    continue
                order = np.argsort(-probs)
                primary_row.append(float(order[0]))
                probability_row.append(float(probs[order[0]]))
                outcomes: list[str] = []
                for idx in order:
                    p = float(probs[idx])
                    if p < 0.01:
                        break
                    outcomes.append(f"{genotypes[int(idx)]} ({p:.0%})")
                text_row.append("\n".join(outcomes))
            primary.append(primary_row)
            probability.append(probability_row)
            cell_text.append(text_row)

    return GeneticsPayload(
        meiosis=meiosis,
        fertilization=FertilizationMatrix(
            row_labels=col_labels,
            col_labels=col_labels,
            zygote_labels=[str(g) for g in genotypes],
            primary_index=primary,
            primary_probability=probability,
            cell_text=cell_text,
            too_large=too_large,
        ),
    )


# ---------------------------------------------------------------------------
# Registry (static structure)
# ---------------------------------------------------------------------------


class GenotypeEntry(TypedDict):
    index: int
    label: str
    ztype_indices: list[int]
    svg: str


class ZTypeEntry(TypedDict):
    index: int
    genotype_index: int
    genotype_label: str
    slab: str


class GTypeEntry(TypedDict):
    index: int
    label: str
    gamete_label: str


class AlleleEntry(TypedDict):
    name: str
    locus: str
    color: str


class RegistryPayload(TypedDict):
    genotypes: list[GenotypeEntry]
    ztypes: list[ZTypeEntry]
    gtypes: list[GTypeEntry]
    alleles: list[AlleleEntry]
    unordered_genotype_labels: list[str]


def _genotype_svg(gt: Genotype, species: Species) -> str:
    """Render the genotype cell SVG (legacy visualization helper)."""
    from natal.frontend.ui.visualization import render_cell_svg

    return render_cell_svg(gt, species, size=80)


def registry_payload(population: GeneticStructureLike) -> RegistryPayload:
    """Serialize the static genetic structure the UI needs once at load."""
    from natal.frontend.ui.visualization import get_allele_color

    registry = population.registry
    species = population.species

    genotype_entries: list[GenotypeEntry] = []
    for g_idx, gt in enumerate(registry.index_to_genotype):
        genotype_entries.append(
            GenotypeEntry(
                index=g_idx,
                label=str(gt),
                ztype_indices=registry.ztype_indices_for(gt),
                svg=_genotype_svg(gt, species),
            )
        )

    index_by_identity: dict[int, int] = {
        id(gt): g_idx for g_idx, gt in enumerate(registry.index_to_genotype)
    }

    ztypes: list[ZTypeEntry] = []
    for z_idx, (gt, slab) in enumerate(registry.index_to_ztype):
        ztypes.append(
            ZTypeEntry(
                index=z_idx,
                genotype_index=index_by_identity.get(id(gt), -1),
                genotype_label=str(gt),
                slab=slab,
            )
        )

    gtypes: list[GTypeEntry] = []
    for gtype_idx, (hg_obj, glab_str) in enumerate(registry.index_to_gtype):
        gtypes.append(
            GTypeEntry(
                index=gtype_idx,
                label=str(hg_obj),
                gamete_label=glab_str,
            )
        )

    alleles: list[AlleleEntry] = []
    for chrom in species.chromosomes:
        for locus in chrom.loci:
            for gene in locus.alleles:
                alleles.append(
                    AlleleEntry(
                        name=gene.name,
                        locus=locus.name,
                        color=get_allele_color(gene.name),
                    )
                )

    return RegistryPayload(
        genotypes=genotype_entries,
        ztypes=ztypes,
        gtypes=gtypes,
        alleles=alleles,
        unordered_genotype_labels=_unordered_genotype_labels(registry),
    )


def _unordered_genotype_labels(registry: IndexRegistry) -> list[str]:
    """Unique unordered (``::``) genotype labels, sorted alphabetically.

    Reuses the legacy dashboard helper: for each genotype it builds
    ``hapA::hapB`` per chromosome with alphabetically sorted haplotype
    strings, joined by ``; `` across chromosomes.
    """
    from natal.frontend.ui.dashboard_helpers import get_unordered_genotype_labels

    return get_unordered_genotype_labels(registry.index_to_genotype)


# ---------------------------------------------------------------------------
# Export (legacy-compatible schema)
# ---------------------------------------------------------------------------


def export_payload(
    population: PanmicticPopulation,
    *,
    include_config: bool = True,
    include_history: bool = True,
    include_hooks: bool = True,
) -> dict[str, object]:  # object: heterogeneous legacy export schema
    """Build the full export dict (schema-compatible with the NiceGUI UI)."""
    payload: dict[str, object] = {  # object: heterogeneous legacy export schema
        "population_name": population.name
    }

    if include_history:
        history = population.history
        history_list: list[dict[str, object]] = []  # object: heterogeneous legacy export records
        if history.schema.mode == "raw":
            for index in range(len(history)):
                history_list.append(
                    _semanticize_state(
                        population, raw_history_state(history, index)
                    )
                )
        if not history_list or history_list[-1].get("tick") != population.tick:
            history_list.append(_semanticize_state(population, population.state))
        payload["history"] = history_list

    if include_config:
        cfg = config_payload(population)
        payload["configuration"] = {
            "parameters": cfg["scalars"],
            "all_config": cfg["full"],
            "presets_visualization": cfg["presets"],
            "fitness": {
                "viability": cfg["viability"],
                "fecundity": cfg["fecundity"],
                "sexual_selection": cfg["sexual_selection"],
            },
        }

    if include_hooks:
        payload["hooks"] = hooks_payload(population)
    return payload


def _semanticize_state(
    population: PanmicticPopulation,
    state: PanmicticState,
) -> dict[str, object]:  # object: heterogeneous legacy export record
    """One history record in the legacy export layout (nested dicts)."""
    registry = population.registry
    genotypes = [str(g) for g in registry.index_to_genotype]
    counts = state.individual_count

    state_dict: dict[str, object] = {  # object: heterogeneous legacy export record
        "tick": int(state.n_tick),
        "individual_count": {"female": [], "male": []},
    }
    individual = state_dict["individual_count"]
    if isinstance(individual, dict):
        female_list: list[dict[str, object]] = []  # object: sparse per-age export rows
        male_list: list[dict[str, object]] = []  # object: sparse per-age export rows
        for age in range(counts.shape[1]):
            female_counts: dict[str, float] = {}
            male_counts: dict[str, float] = {}
            for g_idx, g_str in enumerate(genotypes):
                f_count = float(counts[0, age, g_idx])
                m_count = float(counts[1, age, g_idx])
                if f_count > 0:
                    female_counts[g_str] = f_count
                if m_count > 0:
                    male_counts[g_str] = m_count
            if female_counts:
                female_list.append({"age": float(age), "counts": female_counts})
            if male_counts:
                male_list.append({"age": float(age), "counts": male_counts})
        individual["female"] = female_list
        individual["male"] = male_list

    if isinstance(state, PopulationState):
        sperm = state.sperm_storage
        sperm_data: list[dict[str, object]] = []  # object: heterogeneous sperm export blocks
        for age in range(sperm.shape[0]):
            entries: list[dict[str, object]] = []  # object: heterogeneous sperm export entries
            for f_idx, m_idx in np.argwhere(sperm[age] > 0):
                fi = int(f_idx)
                mi = int(m_idx)
                entries.append(
                    {
                        "female_genotype": genotypes[fi],
                        "male_genotype": genotypes[mi],
                        "value": float(sperm[age, fi, mi]),
                    }
                )
            if entries:
                sperm_data.append({"age": float(age), "entries": entries})
        if sperm_data:
            state_dict["sperm_storage"] = sperm_data
    return state_dict


# ---------------------------------------------------------------------------
# Observation queries
# ---------------------------------------------------------------------------


class ObservationGroupSpec(TypedDict, total=False):
    genotype: list[str] | str
    sex: str
    age: list[int]


class ObservationRequestDict(TypedDict):
    groups: dict[str, ObservationGroupSpec]
    collapse_age: bool


class ObservationRow(TypedDict):
    group: str
    age: int | None
    female: float
    male: float
    total: float


class ObservationResultPayload(TypedDict):
    labels: list[str]
    collapse_age: bool
    rows: list[ObservationRow]


def apply_observation(
    population: PanmicticPopulation,
    groups: dict[str, dict[str, object]],  # object: legacy observation spec mapping (mixed value types)
    collapse_age: bool,
) -> ObservationResultPayload:
    """Build an observation from group specs and apply it to the live state.

    Group spec keys mirror the legacy observation panel: ``genotype``
    (list of labels or ``"*"`` pattern), ``sex`` (``"female"``/``"male"``,
    omitted for both), and ``age`` (``[start, end]`` inclusive).  The
    legacy spellings are normalized to :class:`IndividualSelector` values
    at the observation-filter boundary.
    """
    from natal.frontend.output.observation import ObservationFilter

    obs_filter = ObservationFilter(population.registry)
    observation = obs_filter.build_filter(
        diploid_genotypes=population.registry.index_to_genotype,
        groups=groups,
        collapse_age=collapse_age,
    )
    observed = observation.apply(population.state.individual_count)
    labels = list(observation.labels)

    rows: list[ObservationRow] = []
    for g_idx, label in enumerate(labels):
        if collapse_age or observed.ndim == 2:
            f_val = float(observed[g_idx, 0])
            m_val = float(observed[g_idx, 1])
            rows.append(
                ObservationRow(
                    group=label, age=None, female=f_val, male=m_val,
                    total=f_val + m_val,
                )
            )
        else:
            for a_idx in range(observed.shape[2]):
                f_val = float(observed[g_idx, 0, a_idx])
                m_val = float(observed[g_idx, 1, a_idx])
                rows.append(
                    ObservationRow(
                        group=label,
                        age=a_idx,
                        female=f_val,
                        male=m_val,
                        total=f_val + m_val,
                    )
                )
    return ObservationResultPayload(
        labels=labels, collapse_age=collapse_age, rows=rows
    )


# ---------------------------------------------------------------------------
# Debug: params log, raw dump, state diff
# ---------------------------------------------------------------------------


class ParamChangeRow(TypedDict):
    tick: int
    name: str
    old: float
    new: float


def params_log_rows(population: PanmicticPopulation) -> list[ParamChangeRow]:
    """Serialize the parameter-change audit log (Python + Rust merged)."""
    return [
        ParamChangeRow(tick=int(tick), name=name, old=float(old), new=float(new))
        for tick, name, old, new in population.params_log
    ]


class DiffGenotypeRow(TypedDict):
    label: str
    female_a: float
    female_b: float
    male_a: float
    male_b: float
    delta_female: float
    delta_male: float
    delta_total: float


class DiffPayload(TypedDict):
    tick_a: int
    tick_b: int
    found_a: bool
    found_b: bool
    total_a: float
    total_b: float
    delta_total: float
    genotypes: list[DiffGenotypeRow]


def state_diff(
    population: PanmicticPopulation, tick_a: int, tick_b: int
) -> DiffPayload:
    """Diff the full state between two ticks from raw history.

    Pure history reads — the engine is not touched.  Ticks missing from
    history are reported through ``found_a``/``found_b`` with the live state
    substituted when the requested tick equals the current tick.
    """

    def counts_at(tick: int) -> tuple[bool, NDArray[np.float64]]:
        if tick == population.tick:
            return True, population.state.individual_count
        history = population.history
        if history.schema.mode == "raw" and tick in history.ticks:
            index = history.ticks.index(tick)
            return True, history.individual_count[index]
        return False, population.state.individual_count

    found_a, counts_a = counts_at(tick_a)
    found_b, counts_b = counts_at(tick_b)

    registry = population.registry
    rows: list[DiffGenotypeRow] = []
    for gt in registry.index_to_genotype:
        z_indices = registry.ztype_indices_for(gt)
        f_a = float(counts_a[0][:, z_indices].sum())
        m_a = float(counts_a[1][:, z_indices].sum())
        f_b = float(counts_b[0][:, z_indices].sum())
        m_b = float(counts_b[1][:, z_indices].sum())
        rows.append(
            DiffGenotypeRow(
                label=str(gt),
                female_a=f_a,
                female_b=f_b,
                male_a=m_a,
                male_b=m_b,
                delta_female=f_b - f_a,
                delta_male=m_b - m_a,
                delta_total=(f_b + m_b) - (f_a + m_a),
            )
        )

    total_a = float(counts_a.sum())
    total_b = float(counts_b.sum())
    return DiffPayload(
        tick_a=tick_a,
        tick_b=tick_b,
        found_a=found_a,
        found_b=found_b,
        total_a=total_a,
        total_b=total_b,
        delta_total=total_b - total_a,
        genotypes=rows,
    )


class RawStateDump(TypedDict):
    """Unrounded array dump for the raw state inspector."""

    tick: int
    mode: str
    found: bool
    individual_count: list[list[list[float]]]
    sperm_storage: list[list[list[float]]] | None


def raw_state_dump(
    population: PanmicticPopulation, tick: int | None
) -> RawStateDump:
    """Dump the (sex, age, ztype) tensor for the live or historical state.

    Args:
        population: Panmictic population under inspection.
        tick: Target tick, or ``None`` for the live state.

    Returns:
        Nested plain lists; ``found=False`` when *tick* is not in history.
    """
    history = population.history
    if tick is None or tick == population.tick:
        state = population.state
        return RawStateDump(
            tick=int(state.n_tick),
            mode="live",
            found=True,
            individual_count=state.individual_count.tolist(),
            sperm_storage=state.sperm_storage.tolist()
            if isinstance(state, PopulationState)
            else None,
        )
    if history.schema.mode == "raw" and tick in history.ticks:
        resolved = raw_history_state(history, history.ticks.index(tick))
        return RawStateDump(
            tick=int(resolved.n_tick),
            mode="history",
            found=True,
            individual_count=resolved.individual_count.tolist(),
            sperm_storage=resolved.sperm_storage.tolist()
            if isinstance(resolved, PopulationState)
            else None,
        )
    live = population.state
    return RawStateDump(
        tick=int(live.n_tick),
        mode="live",
        found=False,
        individual_count=live.individual_count.tolist(),
        sperm_storage=live.sperm_storage.tolist()
        if isinstance(live, PopulationState)
        else None,
    )
