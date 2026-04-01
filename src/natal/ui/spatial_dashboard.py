# type: ignore
"""NiceGUI dashboard for spatial populations."""

from __future__ import annotations

import time
from typing import Any

try:
    from nicegui import run, ui

    HAS_NICEGUI = True
except ImportError:
    HAS_NICEGUI = False

from natal.population_state import PopulationState
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import HexGrid
from natal.visualization import get_allele_color, render_cell_svg


class SpatialDashboard:
    """Real-time dashboard for a ``SpatialPopulation``."""

    def __init__(self, population: SpatialPopulation):
        """Initialize the spatial dashboard."""
        if not HAS_NICEGUI:
            raise ImportError("NiceGUI is required. Please install it with: pip install nicegui")

        self.pop = population
        self.is_running = False
        self.is_processing = False
        self._tick_timer = None

        self.selected_deme_idx = 0
        self._last_chart_tick = -1
        self._chart_history: list[list[float]] = []
        self._allele_freq_history: dict[str, list[list[float]]] = {}
        self._record_snapshot()

    async def _run_step(self) -> None:
        """Execute one spatial simulation step."""
        if self.is_processing:
            return

        self.is_processing = True
        self.status_spinner.visible = True
        self.status_label.text = "Running..."

        if not any(getattr(deme, "_finished", False) for deme in self.pop.demes):
            if self.slider_speed.value <= 0:

                def run_batch() -> None:
                    start = time.time()
                    ticks = 0
                    while (
                        time.time() - start < 0.1
                        and not any(getattr(deme, "_finished", False) for deme in self.pop.demes)
                        and ticks < 50
                    ):
                        self.pop.run_tick()
                        ticks += 1

                await run.io_bound(run_batch)
            else:
                await run.io_bound(self.pop.run_tick)

            self.refresh_ui()

        if any(getattr(deme, "_finished", False) for deme in self.pop.demes):
            self.is_running = False
            self.btn_play.props("icon=play_arrow")
            self.btn_play.text = "Play"
            self.status_label.text = "Finished"
        else:
            self.status_label.text = "Ready"

        self.status_spinner.visible = False
        self.is_processing = False

    def _toggle_play(self) -> None:
        """Toggle autoplay."""
        self.is_running = not self.is_running

    async def _on_timer(self) -> None:
        """Drive autoplay from the UI loop."""
        if self.is_running:
            await self._run_step()

    def _update_timer_interval(self) -> None:
        """Update timer interval based on the speed slider."""
        value = float(self.slider_speed.value)
        self._tick_timer.interval = 0.01 if value <= 0.0 else value

    def _record_snapshot(self) -> None:
        """Append the current aggregate state to local chart history."""
        if self.pop.tick == self._last_chart_tick:
            return

        total = float(self.pop.get_total_count())
        self._chart_history.append([float(self.pop.tick), total])

        for allele, freq in self.pop.compute_allele_frequencies().items():
            if allele not in self._allele_freq_history:
                self._allele_freq_history[allele] = []
            self._allele_freq_history[allele].append([float(self.pop.tick), float(freq)])

        self._last_chart_tick = self.pop.tick

    def refresh_ui(self) -> None:
        """Refresh all reactive UI elements."""
        self._record_snapshot()
        self._update_global_stats()
        self._update_charts()
        self._render_landscape()
        self._render_migration_panel()
        self._update_selected_deme()

    def _update_global_stats(self) -> None:
        """Refresh aggregate summary labels."""
        self.lbl_tick.text = str(self.pop.tick)
        self.lbl_total.text = str(self.pop.get_total_count())
        self.lbl_females.text = str(self.pop.get_female_count())
        self.lbl_males.text = str(self.pop.get_male_count())
        self.lbl_mode.text = self.pop.migration_mode
        self.lbl_history_count.text = f"(Current session: {len(self._chart_history)} snapshots)"

    def _update_charts(self) -> None:
        """Update global population and allele-frequency charts."""
        self.chart_pop.options["series"][0]["data"] = self._chart_history
        self.chart_pop.update()

        series_map = {series["name"]: series for series in self.chart_allele.options["series"]}
        for allele, data in self._allele_freq_history.items():
            if allele not in series_map:
                series = {
                    "name": allele,
                    "data": data,
                    "color": get_allele_color(allele),
                }
                self.chart_allele.options["series"].append(series)
                continue
            series_map[allele]["data"] = data

        self.chart_allele.update()

    def _deme_button_style(self, idx: int, count: float, max_count: float) -> str:
        """Return inline style for one deme button."""
        ratio = 0.0 if max_count <= 0.0 else min(1.0, count / max_count)
        fill = int(245 - 110 * ratio)
        outline = "4px solid #0f172a" if idx == self.selected_deme_idx else "1px solid #94a3b8"
        base = [
            f"background: rgb(14, 165, 233, {0.18 + 0.55 * ratio:.3f})",
            "color: #0f172a",
            f"border: {outline}",
            "min-width: 5rem",
            "min-height: 5rem",
            "padding: 0.75rem",
            "display: flex",
            "flex-direction: column",
            "justify-content: center",
            "align-items: center",
            "gap: 0.2rem",
            "box-shadow: inset 0 0 0 9999px rgba(255,255,255,0.15)",
            f"filter: saturate({0.8 + ratio * 0.5:.3f}) brightness({fill / 200:.3f})",
        ]

        topology = self.pop.topology
        if isinstance(topology, HexGrid):
            base.extend(
                [
                    "width: 5.5rem",
                    "height: 6rem",
                    "clip-path: polygon(25% 6%, 75% 6%, 100% 50%, 75% 94%, 25% 94%, 0 50%)",
                ]
            )
        else:
            base.extend(
                [
                    "width: 5.5rem",
                    "height: 5.5rem",
                    "border-radius: 0.8rem",
                ]
            )
        return "; ".join(base)

    def _render_landscape(self) -> None:
        """Render the clickable landscape geometry."""
        self.landscape_container.clear()
        topology = self.pop.topology
        counts = [float(deme.state.individual_count.sum()) for deme in self.pop.demes]
        max_count = max(counts) if counts else 0.0

        with self.landscape_container:
            if topology is None:
                with ui.row().classes("w-full gap-3 flex-wrap"):
                    for idx, deme in enumerate(self.pop.demes):
                        with ui.button(
                            f"{deme.name}\n{int(counts[idx])}",
                            on_click=lambda index=idx: self._select_deme(index),
                        ).props("flat"):
                            pass
                return

            for row in range(topology.rows):
                margin_left = "3rem" if isinstance(topology, HexGrid) and row % 2 == 1 else "0"
                with ui.row().classes("gap-3 no-wrap").style(f"margin-left: {margin_left}"):
                    for col in range(topology.cols):
                        idx = topology.to_index((row, col))
                        label = f"{self.pop.deme(idx).name}\n{int(counts[idx])}"
                        button = ui.button(
                            label,
                            on_click=lambda index=idx: self._select_deme(index),
                        ).props("flat")
                        button.style(self._deme_button_style(idx, counts[idx], max_count))

    def _render_migration_panel(self) -> None:
        """Render migration details for the selected deme."""
        self.migration_container.clear()
        weights = self.pop.migration_row(self.selected_deme_idx)
        selected = self.pop.deme(self.selected_deme_idx)

        ranked_targets = sorted(
            [
                (
                    idx,
                    float(weight),
                    self.pop.topology.from_index(idx) if self.pop.topology is not None else None,
                )
                for idx, weight in enumerate(weights)
                if weight > 0.0
            ],
            key=lambda item: item[1],
            reverse=True,
        )

        with self.migration_container:
            ui.label(
                f"Source deme: {selected.name} (index {self.selected_deme_idx})"
            ).classes("text-base font-semibold text-slate-700")
            ui.label(
                f"Migration mode: {self.pop.migration_mode}, rate={self.pop.migration_rate:.3f}"
            ).classes("text-sm text-slate-500")

            if self.pop.migration_kernel is not None:
                ui.label("Kernel").classes("text-sm font-semibold text-slate-600 mt-2")
                kernel_rows = [
                    {f"c{col}": f"{float(value):.3f}" for col, value in enumerate(row)}
                    for row in self.pop.migration_kernel
                ]
                columns = [
                    {"name": f"c{col}", "label": str(col), "field": f"c{col}"}
                    for col in range(self.pop.migration_kernel.shape[1])
                ]
                ui.table(columns=columns, rows=kernel_rows).props("dense flat").classes("w-full")

            ui.label("Outbound weights").classes("text-sm font-semibold text-slate-600 mt-2")
            rows = [
                {
                    "deme": self.pop.deme(idx).name,
                    "index": idx,
                    "coord": "-" if coord is None else str(coord),
                    "weight": f"{weight:.3f}",
                }
                for idx, weight, coord in ranked_targets[:12]
            ]
            ui.table(
                columns=[
                    {"name": "deme", "label": "Deme", "field": "deme"},
                    {"name": "index", "label": "Index", "field": "index"},
                    {"name": "coord", "label": "Coord", "field": "coord"},
                    {"name": "weight", "label": "Weight", "field": "weight"},
                ],
                rows=rows,
            ).props("dense flat").classes("w-full")

    def _select_deme(self, idx: int) -> None:
        """Select one deme and refresh the detail view."""
        self.selected_deme_idx = idx
        self.tabs_main.set_value("deme")
        self._render_landscape()
        self._render_migration_panel()
        self._update_selected_deme()

    def _update_selected_deme(self) -> None:
        """Refresh the selected-deme detail panel."""
        deme = self.pop.deme(self.selected_deme_idx)
        state = deme.state
        self.lbl_selected_name.text = deme.name
        self.lbl_selected_total.text = str(deme.get_total_count())
        self.lbl_selected_females.text = str(deme.get_female_count())
        self.lbl_selected_males.text = str(deme.get_male_count())
        self._render_deme_genotypes(state)

    def _render_deme_genotypes(self, state: PopulationState) -> None:
        """Render genotype cards for the selected deme."""
        self.genotype_container.clear()
        registry = self.pop.demes[0].registry

        with self.genotype_container:
            for genotype_idx, genotype in enumerate(registry.index_to_genotype):
                female_total = float(state.individual_count[0, :, genotype_idx].sum())
                male_total = float(state.individual_count[1, :, genotype_idx].sum())
                if female_total <= 0.0 and male_total <= 0.0:
                    continue

                with ui.card().classes("items-center p-3 border rounded shadow-sm w-44"):
                    ui.html(render_cell_svg(genotype, self.pop.species, size=72))
                    ui.label(str(genotype)).classes("text-sm font-bold text-center text-gray-800")
                    with ui.row().classes("w-full justify-between text-sm"):
                        ui.label(f"F: {int(female_total)}").classes("text-pink-600 font-semibold")
                        ui.label(f"M: {int(male_total)}").classes("text-blue-600 font-semibold")

                    age_totals = state.individual_count[:, :, genotype_idx].sum(axis=0)
                    with ui.column().classes("w-full gap-0.5 text-xs text-gray-500"):
                        for age_idx, age_total in enumerate(age_totals):
                            if float(age_total) <= 0.0:
                                continue
                            ui.label(f"Age {age_idx}: {int(age_total)}")

    def reset_simulation(self) -> None:
        """Reset the spatial simulation and the dashboard session history."""
        for deme in self.pop.demes:
            deme.reset()
        self.is_running = False
        self._chart_history = []
        self._allele_freq_history = {}
        self._last_chart_tick = -1
        self._record_snapshot()

        self.btn_play.props("icon=play_arrow")
        self.btn_play.text = "Play"
        self.status_label.text = "Ready"
        self.refresh_ui()
        ui.notify("Spatial population reset.")

    def build_layout(self) -> None:
        """Construct the NiceGUI layout."""
        with ui.header().classes("items-center justify-between bg-slate-900 text-white"):
            ui.label("Spatial NATAL Dashboard").classes("text-2xl font-bold")
            ui.label(f"Population: {self.pop.name}").classes("text-base opacity-80")

        with ui.left_drawer(value=True).classes("bg-gray-50 p-4 shadow-lg border-r").props("width=320"):
            ui.label("Control Panel").classes("text-xl font-bold text-gray-700 mb-4")

            with ui.row().classes("items-center gap-2 mb-4 p-2 bg-white rounded border"):
                self.status_spinner = ui.spinner(size="sm").props("color=primary")
                self.status_label = ui.label("Ready").classes("text-base font-medium text-gray-600")
                self.status_spinner.visible = False

            ui.label("Global State").classes("text-sm font-bold text-gray-400 uppercase mb-2")
            with ui.grid(columns=2).classes("w-full gap-y-2 gap-x-4 mb-4"):
                ui.label("Tick").classes("font-bold text-gray-600")
                self.lbl_tick = ui.label(str(self.pop.tick)).classes("text-right font-mono")
                ui.label("Total").classes("font-bold text-gray-600")
                self.lbl_total = ui.label(str(self.pop.get_total_count())).classes("text-right font-mono")
                ui.label("Females").classes("font-bold text-pink-600")
                self.lbl_females = ui.label(str(self.pop.get_female_count())).classes("text-right font-mono text-pink-600")
                ui.label("Males").classes("font-bold text-blue-600")
                self.lbl_males = ui.label(str(self.pop.get_male_count())).classes("text-right font-mono text-blue-600")
                ui.label("Migration").classes("font-bold text-gray-600")
                self.lbl_mode = ui.label(self.pop.migration_mode).classes("text-right font-mono")

            ui.label("Interval (s) (0=Unlimited)").classes("text-sm font-bold text-gray-400 uppercase mt-4 mb-2")
            self.slider_speed = ui.slider(min=0.0, max=0.2, value=0.05, step=0.005).props("label-always")
            self.slider_speed.on_value_change(self._update_timer_interval)

            with ui.column().classes("w-full gap-2 mt-4"):
                with ui.row().classes("w-full gap-2"):
                    ui.button("Step", on_click=self._run_step).props("icon=skip_next outline").classes("flex-grow")

                    def update_play_state(event: Any) -> None:
                        self._toggle_play()
                        icon = "pause" if self.is_running else "play_arrow"
                        text = "Pause" if self.is_running else "Play"
                        event.sender.props(f"icon={icon}")
                        event.sender.text = text

                    self.btn_play = ui.button("Play", on_click=update_play_state).props("icon=play_arrow").classes("flex-grow")

                ui.button("Reset", on_click=self.reset_simulation).props("icon=restart_alt flat color=grey").classes("w-full")

            self.lbl_history_count = ui.label("").classes("text-sm text-gray-400 italic mt-4")
            ui.separator().classes("my-4")

            ui.label("Selected Deme").classes("text-sm font-bold text-gray-400 uppercase mb-2")
            with ui.grid(columns=2).classes("w-full gap-y-2 gap-x-4"):
                ui.label("Name").classes("font-bold text-gray-600")
                self.lbl_selected_name = ui.label("").classes("text-right font-mono")
                ui.label("Total").classes("font-bold text-gray-600")
                self.lbl_selected_total = ui.label("").classes("text-right font-mono")
                ui.label("Females").classes("font-bold text-pink-600")
                self.lbl_selected_females = ui.label("").classes("text-right font-mono text-pink-600")
                ui.label("Males").classes("font-bold text-blue-600")
                self.lbl_selected_males = ui.label("").classes("text-right font-mono text-blue-600")

        with ui.column().classes("w-full p-4 gap-6"):
            with ui.card().classes("w-full p-0 gap-0 border-none shadow-sm"):
                with ui.row().classes("w-full no-wrap"):
                    self.chart_pop = ui.highchart(
                        {
                            "title": {"text": "Total Population"},
                            "chart": {"type": "line", "animation": False, "height": 300},
                            "xAxis": {"title": {"text": "Tick"}},
                            "yAxis": {"title": {"text": "Count"}},
                            "series": [{"name": "TotalPop", "data": []}],
                            "plotOptions": {"series": {"marker": {"enabled": False}}},
                        }
                    ).classes("w-1/2 h-80")

                    self.chart_allele = ui.highchart(
                        {
                            "title": {"text": "Global Allele Frequencies"},
                            "chart": {"type": "line", "animation": False, "height": 300},
                            "xAxis": {"title": {"text": "Tick"}},
                            "yAxis": {"title": {"text": "Freq"}, "min": 0.0, "max": 1.0},
                            "series": [],
                            "plotOptions": {"series": {"marker": {"enabled": False}}},
                        }
                    ).classes("w-1/2 h-80")

            with ui.tabs().classes("w-full justify-start border-b") as tabs:
                self.tabs_main = tabs
                tab_overview = ui.tab(name="overview", label="Landscape", icon="grid_view")
                tab_deme = ui.tab(name="deme", label="Selected Deme", icon="search")

            with ui.tab_panels(tabs, value="overview").classes("w-full bg-transparent p-0"):
                with ui.tab_panel(tab_overview).classes("w-full"):
                    with ui.row().classes("w-full gap-6 items-start"):
                        with ui.card().classes("flex-1 p-4 border rounded shadow-sm"):
                            ui.label("Landscape").classes("text-lg font-bold text-gray-700 mb-3")
                            self.landscape_container = ui.column().classes("gap-3")
                        with ui.card().classes("w-[26rem] p-4 border rounded shadow-sm"):
                            ui.label("Migration Rule").classes("text-lg font-bold text-gray-700 mb-3")
                            self.migration_container = ui.column().classes("w-full gap-2")

                with ui.tab_panel(tab_deme).classes("w-full"):
                    ui.label("Selected Deme State").classes("text-xl font-bold text-gray-700 mb-4")
                    self.genotype_container = ui.row().classes("w-full flex-wrap gap-4")

        self._tick_timer = ui.timer(0.1, self._on_timer)
        self.refresh_ui()


def launch_spatial(
    population: SpatialPopulation,
    port: int = 8080,
    title: str = "NATAL Spatial Dashboard",
) -> None:
    """Launch the embedded spatial dashboard."""

    @ui.page("/")
    def main_page() -> None:
        dashboard = SpatialDashboard(population)
        dashboard.build_layout()

    print(f"Starting Spatial Dashboard at http://localhost:{port}")
    ui.run(title=title, port=port, show=False, reload=False, favicon="natal.svg")
