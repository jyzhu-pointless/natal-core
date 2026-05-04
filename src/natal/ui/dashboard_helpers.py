# type: ignore
"""
Shared UI helper functions for NATAL dashboards.

Extracted from dashboard_population.py to avoid duplication
between the panmictic and spatial dashboards.
"""

import inspect
from typing import Any

import numpy as np

try:
    from nicegui import ui
    HAS_NICEGUI = True
except ImportError:
    HAS_NICEGUI = False

from natal.population_config import CONCAVE, FIXED, LINEAR, NO_COMPETITION


def get_unordered_genotype_labels(genotypes: list[Any]) -> list[str]:
    """Generate unique unordered (``::``) genotype labels from a genotype list.

    For each genotype, builds a label in the form ``hapstrA::hapstrB``
    (sorted alphabetically so ``WT|Dr`` and ``Dr|WT`` both become ``WT::Dr``).
    Multi-chromosome: ``hapA::hapB; hapC::hapC``.

    Returns sorted unique labels suitable for dropdown options.
    """
    seen: set[str] = set()
    labels: list[str] = []
    for gt in genotypes:
        chrom_pairs: list[str] = []
        for chrom in gt.species.chromosomes:
            mat_hap = gt.maternal.get_haplotype_for_chromosome(chrom)
            pat_hap = gt.paternal.get_haplotype_for_chromosome(chrom)

            def _hap_str(hap: Any, loci: list[Any]) -> str:
                if hap is None:
                    return ""
                names: list[str] = []
                for locus in loci:
                    gene = hap.get_gene_at_locus(locus)
                    names.append(gene.name if gene else "")
                return "/".join(names)

            mat_str = _hap_str(mat_hap, chrom.loci)
            pat_str = _hap_str(pat_hap, chrom.loci)
            a_str, b_str = sorted([mat_str, pat_str])
            chrom_pairs.append(f"{a_str}::{b_str}")

        label = "; ".join(chrom_pairs)
        if label not in seen:
            seen.add(label)
            labels.append(label)

    labels.sort()
    return labels


def numpy_converter(obj: Any) -> Any:
    """JSON serialization helper for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def jsonable_config_value(value: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, tuple):
        return [jsonable_config_value(v) for v in value]
    if isinstance(value, list):
        return [jsonable_config_value(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable_config_value(v) for k, v in value.items()}
    return value


def growth_mode_name(mode: int) -> str:
    """Map a numeric growth mode constant to its name."""
    mapping = {
        NO_COMPETITION: "NO_COMPETITION",
        FIXED: "FIXED",
        LINEAR: "LINEAR",
        CONCAVE: "CONCAVE",
    }
    return mapping.get(int(mode), f"UNKNOWN_{mode}")


def format_op(op: Any) -> str:
    """Format a declarative HookOp into a human-readable HTML string."""
    from natal.hooks.declarative import OpType

    try:
        normalized_type = OpType(int(op.op_type))
        type_name = normalized_type.name.lower()
    except (ValueError, TypeError):
        type_name = str(op.op_type).split(".")[-1].lower()

    parts = [f"<b>{type_name}</b>"]

    # Genotypes
    if op.genotypes == "*":
        parts.append("ALL genotypes")
    elif isinstance(op.genotypes, list):
        parts.append(f"genotypes {op.genotypes}")
    else:
        parts.append(f"genotype {op.genotypes}")

    # Sex
    if op.sex != "both":
        parts.append(f"({op.sex} only)")

    # Param
    if op.op_type in (OpType.SCALE, OpType.KILL):
        parts.append(f"by factor {op.param}")
    elif op.op_type in (OpType.ADD, OpType.SUBTRACT):
        parts.append(f"by {op.param}")
    elif op.op_type == OpType.SET:
        parts.append(f"to {op.param}")

    # Ages
    if op.ages != "*":
        parts.append(f"at ages {op.ages}")

    # Condition
    if op.condition:
        parts.append(f"WHEN <span class='text-blue-600 font-mono'>{op.condition}</span>")

    return " ".join(parts)


def get_hooks_data(population: Any) -> list[dict]:
    """Serialize hook information for export."""
    from natal.hooks.types import OpType

    op_type_name_map = {
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

    def normalize_op_type(op_type: Any) -> str:
        try:
            enum_value = OpType(int(op_type))
            return op_type_name_map.get(enum_value, enum_value.name.lower())
        except (ValueError, TypeError):
            if hasattr(op_type, "name"):
                return str(op_type.name).lower()
            return str(op_type).lower()

    def normalize_ages(ages: Any) -> Any:
        if ages == "*":
            return "*"
        if isinstance(ages, range):
            return [float(a) for a in ages]
        if isinstance(ages, (list, tuple, np.ndarray)):
            return [float(a) for a in ages]
        if isinstance(ages, (int, float, np.integer, np.floating)):
            return float(ages)
        return str(ages)

    hooks_data = []
    for desc in population.get_compiled_hooks():
        hook_info = {
            "event": desc.event,
            "name": desc.name,
            "priority": desc.priority,
        }
        if hasattr(desc, "ops") and desc.ops:
            hook_info["type"] = "declarative"
            op_list = []
            for op in desc.ops:
                op_dict = {
                    "type": normalize_op_type(op.op_type),
                    "genotypes": op.genotypes,
                    "ages": normalize_ages(op.ages),
                    "sex": op.sex,
                    "param": op.param,
                    "condition": op.condition,
                }
                op_list.append(op_dict)
            hook_info["operations"] = op_list
        else:
            hook_info["type"] = "custom"
            target_fn = desc.njit_fn or desc.py_wrapper
            if target_fn:
                try:
                    sig = inspect.signature(target_fn)
                    hook_info["signature"] = str(sig)
                except (ValueError, TypeError):
                    hook_info["signature"] = "N/A"
        hooks_data.append(hook_info)
    return hooks_data


def render_single_hook(desc: Any, is_global: bool = False) -> None:
    """Render one compiled hook descriptor as an expansion panel."""
    if not HAS_NICEGUI:
        return

    label = f"{desc.name} ({desc.event})"
    if is_global:
        label = f"[Global] {desc.name} ({desc.event})"

    with ui.expansion(label, icon="code").classes("w-full border rounded mb-2"):
        with ui.column().classes("p-2"):
            ui.label(f"Priority: {desc.priority}").classes("text-xs text-gray-500")
            if is_global:
                ui.label("Applied to: ALL demes").classes("text-xs text-green-600 font-semibold")

            if desc.plan:
                if hasattr(desc, "ops") and desc.ops:
                    ui.label("Declarative Operations:").classes("font-bold text-base")
                    with ui.column().classes("gap-1"):
                        for op in desc.ops:
                            ui.html(format_op(op)).classes("text-sm font-mono p-1 border-b bg-gray-50 rounded")
                else:
                    ui.label("Compiled Plan (Low-level arrays)").classes("text-sm text-gray-400")
            else:
                if desc.py_wrapper:
                    try:
                        code = inspect.getsource(desc.py_wrapper)
                        ui.code(code, language="python").classes("w-full text-sm")
                    except OSError:
                        ui.label("(Source code unavailable)").classes("italic")
                elif desc.njit_fn:
                    ui.label("Custom Numba Hook").classes("font-bold")


def render_hooks_panel(population: Any) -> None:
    """Render the hooks expansion-panel list into the current NiceGUI container."""
    if not HAS_NICEGUI:
        return

    hooks = population.get_compiled_hooks()
    if not hooks:
        ui.label("No hooks registered.").classes("text-gray-500 italic")
        return

    for desc in hooks:
        render_single_hook(desc)


def build_observation_from_specs(
    population: Any,
    specs: list[dict],
    collapse_age: bool = False,
) -> Any:
    """Build an Observation from a list of group spec dicts.

    Each spec dict may contain keys: ``genotype``, ``age``, ``sex``.
    """
    from natal.observation import ObservationFilter

    obs_filter = ObservationFilter(population.registry)
    # Convert list of dicts to dict of dicts with group_N keys
    groups = {f"group_{i}": spec for i, spec in enumerate(specs)}
    return obs_filter.build_filter(
        diploid_genotypes=population.species,
        groups=groups,
        collapse_age=collapse_age,
    )


def render_observation_results(observation: Any, state_or_observed: Any) -> None:
    """Render observation results as a table.

    ``state_or_observed`` can be either a state object with ``individual_count``,
    or a pre-computed numpy array from ``observation.apply()``.
    """
    if not HAS_NICEGUI:
        return

    # Accept both state objects and pre-applied arrays
    if hasattr(state_or_observed, "individual_count"):
        observed = observation.apply(state_or_observed.individual_count)
    else:
        observed = state_or_observed
    labels = observation.labels
    collapse_age = observation.collapse_age

    # observed shape: (n_groups, n_sexes) or (n_groups, n_sexes, n_ages)
    rows = []
    for g_idx, label in enumerate(labels):
        if collapse_age or observed.ndim == 2:
            f_val = float(observed[g_idx, 0]) if observed.shape[1] > 0 else 0.0
            m_val = float(observed[g_idx, 1]) if observed.shape[1] > 1 else 0.0
            rows.append({"Group": label, "Female": f_val, "Male": m_val, "Total": f_val + m_val})
        else:
            for a_idx in range(observed.shape[2]):
                f_val = float(observed[g_idx, 0, a_idx]) if observed.shape[1] > 0 else 0.0
                m_val = float(observed[g_idx, 1, a_idx]) if observed.shape[1] > 1 else 0.0
                rows.append({
                    "Group": label,
                    "Age": a_idx,
                    "Female": f_val,
                    "Male": m_val,
                    "Total": f_val + m_val,
                })

    if not rows:
        ui.label("No observation data.").classes("text-gray-500 italic")
        return

    columns = [{"name": k, "label": k, "field": k} for k in rows[0].keys()]
    ui.table(columns=columns, rows=rows).props("dense flat").classes("w-full")


class ObservationPanel:
    """Reusable Observation group editor + results panel.

    Usage in a dashboard ``build_layout()``:

        self.obs_panel = ObservationPanel(
            genotype_labels=genotype_labels,
            get_state=lambda: self.pop.state,
            get_registry=lambda: self.pop.registry,
        )
        # ... later, inside a tab panel:
        self.obs_panel.build(container)
    """

    def __init__(
        self,
        genotype_labels: list[str],
        get_state: Any,
        get_registry: Any,
    ) -> None:
        if not HAS_NICEGUI:
            return
        self._genotype_labels = genotype_labels
        self._get_state = get_state
        self._get_registry = get_registry
        self._group_specs: list[dict] = []
        self._observation: Any = None
        self._collapse_age: Any = None
        self._groups_container: Any = None
        self._results_container: Any = None

    def build(self, container: Any) -> None:
        """Render the full observation UI into *container*."""
        if not HAS_NICEGUI:
            return
        with container:
            with ui.row().classes("w-full gap-6"):
                with ui.card().classes("w-[26rem] shrink-0 p-4 border rounded shadow-sm"):
                    ui.label("Observation Groups").classes("text-lg font-bold text-gray-700 mb-3")
                    ui.label(
                        "Select genotypes or type patterns (e.g. WT::Dr, WT|*, {A,B}|*)."
                    ).classes("text-sm text-gray-500 mb-2")
                    self._groups_container = ui.column().classes("w-full gap-2")
                    with ui.row().classes("w-full gap-2 mt-2"):
                        ui.button(
                            "Add Group", on_click=self._add_group, icon="add",
                        ).props("flat").classes("flex-grow")
                    self._collapse_age = ui.checkbox("Collapse Age", value=False)
                    with ui.row().classes("w-full gap-2 mt-2"):
                        ui.button(
                            "Apply", on_click=self._apply, icon="refresh",
                        ).props("color=primary").classes("flex-grow")

                with ui.card().classes("flex-1 p-4 border rounded shadow-sm"):
                    ui.label("Observed Counts").classes("text-lg font-bold text-gray-700 mb-3")
                    self._results_container = ui.column().classes("w-full")

    # -- group management -------------------------------------------------

    def _add_group(self) -> None:
        self._group_specs.append({"genotype": None, "sex": "both"})
        self._rerender_groups()

    def _remove_group(self, idx: int) -> None:
        if idx < len(self._group_specs):
            del self._group_specs[idx]
            self._rerender_groups()

    def _update_spec(self, idx: int, key: str, value: Any) -> None:
        if idx < len(self._group_specs):
            self._group_specs[idx][key] = value

    def _rerender_groups(self) -> None:
        if self._groups_container is None:
            return
        self._groups_container.clear()
        with self._groups_container:
            for i, spec in enumerate(self._group_specs):
                self._render_group_row(i, spec)

    def _render_group_row(self, idx: int, spec: dict) -> None:
        with ui.card().classes("p-2 border rounded w-full") as card:
            card.props("flat")
            with ui.row().classes("items-center gap-2 w-full"):
                ui.label(f"G{idx}").classes("text-xs font-bold text-gray-500 w-8")

                sel = ui.select(
                    label="Genotype", options=self._genotype_labels,
                    value=spec.get("genotype"),
                    multiple=True,
                    new_value_mode="add",
                ).classes("flex-grow")
                sel.on_value_change(lambda e, i=idx: self._update_spec(i, "genotype", e.value))

                sex = ui.select(
                    label="Sex",
                    options={"both": "Both", "female": "Female", "male": "Male"},
                    value=spec.get("sex", "both"),
                ).classes("w-24")
                sex.on_value_change(lambda e, i=idx: self._update_spec(i, "sex", e.value))

                a_start = ui.number(
                    label="Age Start", value=spec.get("age_start"), min=0, precision=0,
                ).classes("w-20")
                a_start.on_value_change(lambda e, i=idx: self._update_spec(i, "age_start", e.value))

                a_end = ui.number(
                    label="Age End", value=spec.get("age_end"), min=0, precision=0,
                ).classes("w-20")
                a_end.on_value_change(lambda e, i=idx: self._update_spec(i, "age_end", e.value))

                ui.button(
                    icon="delete", on_click=lambda i=idx: self._remove_group(i),
                ).props("flat round dense").classes("text-red-500")

    # -- apply ------------------------------------------------------------

    def _apply(self) -> None:
        from natal.observation import ObservationFilter

        if self._results_container is None:
            return
        self._results_container.clear()

        if not self._group_specs:
            with self._results_container:
                ui.label("No observation groups defined.").classes("text-gray-500 italic")
            return

        groups: dict[str, dict] = {}
        for i, spec in enumerate(self._group_specs):
            gs: dict = {}
            gv = spec.get("genotype")
            if gv and gv != "*":
                gs["genotype"] = gv
            sv = spec.get("sex", "both")
            if sv and sv != "both":
                gs["sex"] = sv
            a_s = spec.get("age_start")
            a_e = spec.get("age_end")
            if a_s is not None and a_e is not None:
                gs["age"] = (int(a_s), int(a_e))
            groups[f"group_{i}"] = gs

        try:
            registry = self._get_registry()
            obs_filter = ObservationFilter(registry)
            obs = obs_filter.build_filter(
                diploid_genotypes=registry.index_to_genotype,
                groups=groups,
                collapse_age=bool(self._collapse_age.value) if self._collapse_age else False,
            )
            self._observation = obs
            state = self._get_state()
            with self._results_container:
                render_observation_results(obs, state)
        except Exception as e:
            with self._results_container:
                ui.label(f"Error: {e}").classes("text-red-600")
