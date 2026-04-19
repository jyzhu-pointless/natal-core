# type: ignore
"""NiceGUI dashboard for spatial populations."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    from nicegui import run, ui

    HAS_NICEGUI = True
except ImportError:
    HAS_NICEGUI = False

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore

from natal.population_state import PopulationState
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import HexGrid
from natal.visualization import get_allele_color, render_cell_svg


class SpatialDashboard:
    """Real-time dashboard for a ``SpatialPopulation``."""

    LARGE_LANDSCAPE_THRESHOLD = 400

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
        self.show_numbers = False  # Toggle for displaying population counts as numbers
        self._max_count_history = 0.0  # Track historical maximum for colorbar range
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

    def _get_hex_vertices(self, center_x: float, center_y: float, size: float = 1.0) -> tuple[list[float], list[float]]:
        """Get the vertices of a regular hexagon (pointy-top orientation)."""
        # Pointy-top regular hexagon: top vertex at 90 degrees.
        angles = [90 + i * 60 for i in range(6)]
        xs = [center_x + size * np.cos(np.radians(angle)) for angle in angles]
        ys = [center_y + size * np.sin(np.radians(angle)) for angle in angles]
        # Close the polygon
        xs.append(xs[0])
        ys.append(ys[0])
        return xs, ys

    def _get_square_vertices(self, center_x: float, center_y: float, size: float = 1.0) -> tuple[list[float], list[float]]:
        """Get the vertices of a square."""
        half = size / np.sqrt(2)
        xs = [center_x - half, center_x + half, center_x + half, center_x - half, center_x - half]
        ys = [center_y - half, center_y - half, center_y + half, center_y + half, center_y - half]
        return xs, ys

    def _get_color_for_value(self, value: float, min_val: float, max_val: float) -> str:
        """Get RGB color from viridis colormap."""
        if max_val <= min_val:
            normalized = 0.5
        else:
            normalized = (value - min_val) / (max_val - min_val)
        normalized = np.clip(normalized, 0.0, 1.0)

        # Use matplotlib if available, otherwise use builtin viridis approximation
        if plt is not None:
            cmap = plt.get_cmap("viridis")
            rgba = cmap(normalized)
            return f"rgb({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)})"

        # Builtin viridis approximation (dark purple to yellow)
        # Simplified from matplotlib viridis colormap
        if normalized < 0.25:
            # Dark purple to purple
            t = normalized / 0.25
            r = int(68 + (57 - 68) * t)
            g = int(1 + (83 - 1) * t)
            b = int(84 + (130 - 84) * t)
        elif normalized < 0.5:
            # Purple to blue
            t = (normalized - 0.25) / 0.25
            r = int(57 + (33 - 57) * t)
            g = int(83 + (145 - 83) * t)
            b = int(130 + (166 - 130) * t)
        elif normalized < 0.75:
            # Blue to cyan/green
            t = (normalized - 0.5) / 0.25
            r = int(33 + (40 - 33) * t)
            g = int(145 + (182 - 145) * t)
            b = int(166 + (140 - 166) * t)
        else:
            # Cyan to yellow
            t = (normalized - 0.75) / 0.25
            r = int(40 + (253 - 40) * t)
            g = int(182 + (231 - 182) * t)
            b = int(140 + (37 - 140) * t)

        return f"rgb({r}, {g}, {b})"

    def _use_large_landscape_mode(self) -> bool:
        """Return whether the landscape should use the scalable heatmap mode."""
        topology = self.pop.topology
        return topology is not None and self.pop.n_demes > self.LARGE_LANDSCAPE_THRESHOLD

    def _build_large_landscape_figure(
        self,
        topology: Any,
        counts: list[float],
        max_count: float,
    ) -> Any:
        """Build one scalable row/col heatmap for large landscapes."""
        z = np.zeros((topology.rows, topology.cols), dtype=np.float64)
        customdata = np.zeros((topology.rows, topology.cols), dtype=np.int64)
        text = np.empty((topology.rows, topology.cols), dtype=object)

        for row in range(topology.rows):
            for col in range(topology.cols):
                idx = topology.to_index((row, col))
                count = counts[idx]
                z[row, col] = count
                customdata[row, col] = idx
                text[row, col] = f"Index: {idx}<br>Coord: ({row}, {col})<br>Count: {int(count):,}"

        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=z,
                    x=np.arange(topology.cols, dtype=np.int64),
                    y=np.arange(topology.rows, dtype=np.int64),
                    customdata=customdata,
                    text=text,
                    hovertemplate="%{text}<extra></extra>",
                    colorscale="Viridis",
                    zmin=0.0,
                    zmax=max_count,
                    showscale=True,
                    colorbar={
                        "title": "Population Count",
                        "thickness": 15,
                        "len": 0.8,
                        "orientation": "h",  # Horizontal orientation
                        "x": 0.5,
                        "xanchor": "center",
                        "y": -0.15,
                        "yanchor": "top",
                    },
                )
            ]
        )

        selected_coord = topology.from_index(self.selected_deme_idx)
        fig.add_shape(
            type="rect",
            x0=selected_coord[1] - 0.5,
            x1=selected_coord[1] + 0.5,
            y0=selected_coord[0] - 0.5,
            y1=selected_coord[0] + 0.5,
            line={"color": "#0f172a", "width": 3},
            fillcolor="rgba(0,0,0,0)",
        )
        fig.update_layout(
            title="Landscape",
            xaxis={"title": "Column", "constrain": "domain", "dtick": max(1, topology.cols // 10)},
            yaxis={"title": "Row", "autorange": "reversed", "dtick": max(1, topology.rows // 10)},
            height=500,
            transition={"duration": 0},
            margin={"l": 10, "r": 10, "t": 40, "b": 60},  # Increased bottom margin for horizontal colorbar
        )
        return fig

    def _render_landscape(self) -> None:
        """Render the clickable landscape geometry using plotly."""
        self.landscape_container.clear()
        topology = self.pop.topology
        counts = [float(deme.state.individual_count.sum()) for deme in self.pop.demes]
        current_max = max(counts) if counts else 1.0
        # Only update historical max if current max exceeds 110% of historical max
        if current_max > 1.1 * self._max_count_history:
            self._max_count_history = current_max
        # Colorbar range is always 110% of historical max for stability
        max_count = 1.1 * self._max_count_history

        if not HAS_PLOTLY:
            ui.label("Plotly is required for landscape visualization. Install with: pip install plotly").classes(
                "text-red-600 font-bold"
            )
            return

        with self.landscape_container:
            if topology is None:
                # Non-spatial population: arrange demes in a grid
                fig = go.Figure()
                n_demes = len(self.pop.demes)
                cols = int(np.ceil(np.sqrt(n_demes)))

                for idx, count in enumerate(counts):
                    row = idx // cols
                    col = idx % cols
                    x, y = col, -row
                    color = self._get_color_for_value(count, 0, max_count)
                    hover_text = f"Index: {idx}<br>Count: {int(count):,}"
                    if self.show_numbers:
                        hover_text += f"<br>{int(count)}"
                    fig.add_trace(
                        go.Scatter(
                            x=[x],
                            y=[y],
                            mode="markers",
                            marker={"size": 15, "color": color},
                            hovertext=hover_text,
                            hoverinfo="text",
                            customdata=[idx],
                            showlegend=False,
                        )
                    )
                    if self.show_numbers:
                        fig.add_trace(
                            go.Scatter(
                                x=[x],
                                y=[y],
                                mode="text",
                                text=[str(int(count))],
                                textposition="middle center",
                                textfont={"size": 10, "color": "#111827"},
                                hoverinfo="skip",
                                showlegend=False,
                            )
                        )
                    if idx == self.selected_deme_idx:
                        fig.add_shape(
                            type="circle",
                            x0=x - 0.6,
                            y0=y - 0.6,
                            x1=x + 0.6,
                            y1=y + 0.6,
                            line={"color": "#0f172a", "width": 3},
                        )

                fig.update_layout(
                    title="Landscape",
                    xaxis={"showgrid": False, "zeroline": False},
                    yaxis={"showgrid": False, "zeroline": False},
                    hovermode="closest",
                    height=400,
                )
                plot_element = ui.plotly(fig).classes("w-full border rounded").props('style="height: 400px; width: 100%;"')
                plot_element.on("plotly_click", self._on_landscape_click, ["points"])
                return

            if self._use_large_landscape_mode():
                ui.label(
                    "Large landscape mode uses a scalable row/column heatmap for responsiveness."
                ).classes("text-sm text-slate-500")
                fig = self._build_large_landscape_figure(
                    topology=topology,
                    counts=counts,
                    max_count=max_count,
                )
                plot_element = ui.plotly(fig).classes("w-full border rounded").props('style="height: 500px; width: 100%;"')
                plot_element.on("plotly_click", self._on_landscape_click, ["points"])
                return

            # Hex or Square grid topology
            is_hex = isinstance(topology, HexGrid)
            fig = go.Figure()

            # Calculate positions for each hex/square
            hex_size = 1.0
            if is_hex:
                # Parallelogram grid: regular hexagonal arrangement
                # Spacing remains the same as in odd-r layout for consistent geometry
                x_spacing = float(np.sqrt(3) * hex_size)  # 1.732
                y_spacing = 1.5 * hex_size                # 1.5
            else:
                # For squares: spacing is 2*size
                x_spacing = 2.0
                y_spacing = 2.0

            for row in range(topology.rows):
                for col in range(topology.cols):
                    idx = topology.to_index((row, col))
                    count = counts[idx]

                    if is_hex:
                        # Parallelogram grid: continuous diagonal offset
                        # Each row shifts by half a hexagon width relative to previous row
                        x = col * x_spacing + row * (x_spacing / 2.0)
                        y = row * y_spacing
                    else:
                        x = col * x_spacing
                        y = row * y_spacing

                    color = self._get_color_for_value(count, 0, max_count)

                    # Draw the polygon (hex or square)
                    if is_hex:
                        xs, ys = self._get_hex_vertices(x, y, hex_size)
                    else:
                        xs, ys = self._get_square_vertices(x, y, hex_size)

                    # Add filled polygon as shape
                    fig.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode="lines",
                            fill="toself",
                            fillcolor=color,
                            line={"color": "rgba(100, 100, 100, 0.3)", "width": 1},
                            hoverinfo="skip",
                            showlegend=False,
                            name="",
                        )
                    )

                    # Add an invisible hit target as the sole interaction layer.
                    # Hover labels and click selection are both handled on this trace.
                    fig.add_trace(
                        go.Scatter(
                            x=[x],
                            y=[y],
                            mode="markers",
                            marker={
                                "size": 64 if is_hex else 56,
                                "symbol": "hexagon" if is_hex else "square",
                                "opacity": 0.001,
                                "color": "rgba(0,0,0,0)",
                                "line": {"width": 0},
                            },
                            hovertemplate=f"Index: {idx}<br>Count: {int(count):,}<extra></extra>",
                            customdata=[idx],
                            showlegend=False,
                            name="",
                        )
                    )

                    # Keep the selection outline in the top interaction stack so it
                    # cannot be visually covered by the invisible hit target.
                    if idx == self.selected_deme_idx:
                        fig.add_trace(
                            go.Scatter(
                                x=xs,
                                y=ys,
                                fill=None,
                                line={"color": "#0f172a", "width": 4},
                                hoverinfo="skip",
                                showlegend=False,
                                name="",
                            )
                        )

                    if self.show_numbers:
                        fig.add_trace(
                            go.Scatter(
                                x=[x],
                                y=[y],
                                mode="text",
                                text=[str(int(count))],
                                textposition="middle center",
                                textfont={"size": 10, "color": "#111827"},
                                customdata=[idx],
                                hoverinfo="skip",
                                showlegend=False,
                                name="",
                            )
                        )

            # Add horizontal colorbar at the bottom
            cbar_trace = go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "colorscale": "Viridis",
                    "cmid": max_count / 2,
                    "colorbar": {
                        "title": "Population Count",
                        "thickness": 15,
                        "len": 0.8,
                        "orientation": "h",  # Horizontal orientation
                        "x": 0.5,
                        "xanchor": "center",
                        "y": -0.15,
                        "yanchor": "top",
                    },
                    "cmin": 0,
                    "cmax": max_count,
                    "size": 0,
                },
                showlegend=False,
            )
            fig.add_trace(cbar_trace)

            fig.update_layout(
                title="Landscape",
                xaxis={"showgrid": False, "zeroline": False, "showticklabels": False, "constrain": "domain"},
                yaxis={
                    "showgrid": False,
                    "zeroline": False,
                    "showticklabels": False,
                    "scaleanchor": "x",
                    "scaleratio": 1,
                },
                hovermode="closest",
                height=500,
                transition={"duration": 0},
                margin={"l": 10, "r": 10, "t": 40, "b": 60},  # Increased bottom margin for horizontal colorbar
            )

            # Use NiceGUI plotly widget to keep landscape inside the panel with built-in zoom.
            plot_element = ui.plotly(fig).classes("w-full border rounded").props('style="height: 500px; width: 100%;"')
            plot_element.on("plotly_click", self._on_landscape_click, ["points"])

    def _on_landscape_click(self, e: Any) -> None:
        """Handle plotly click and select the clicked deme."""
        args = getattr(e, "args", None)
        if not isinstance(args, dict):
            return

        points = args.get("points")
        if not isinstance(points, list) or not points:
            return

        customdata = points[0].get("customdata")

        # NiceGUI/Plotly may serialize customdata as scalar or list-like.
        if isinstance(customdata, (list, tuple)):
            if not customdata:
                return
            customdata = customdata[0]

        idx: int | None
        try:
            idx = int(customdata) if customdata is not None else None
        except (TypeError, ValueError):
            idx = None

        if idx is None and self.pop.topology is not None:
            point_x = points[0].get("x")
            point_y = points[0].get("y")
            try:
                col = int(point_x)
                row = int(point_y)
            except (TypeError, ValueError):
                return
            normalized = self.pop.topology.normalize_coord(row, col)
            if normalized is None:
                return
            idx = self.pop.topology.to_index(normalized)

        if idx is not None and 0 <= idx < len(self.pop.demes):
            self._select_deme(idx)

    def _selected_deme_age_rows(self, state: PopulationState) -> list[dict[str, int]]:
        """Return age summary rows for the selected deme."""
        if state.individual_count.ndim != 3:
            return []

        age_totals = state.individual_count.sum(axis=(0, 2))
        age_female_totals = state.individual_count[0].sum(axis=1)
        age_male_totals = state.individual_count[1].sum(axis=1)

        rows: list[dict[str, int]] = []
        for age_idx in range(age_totals.shape[0]):
            rows.append(
                {
                    "age": int(age_idx),
                    "female": int(age_female_totals[age_idx]),
                    "male": int(age_male_totals[age_idx]),
                    "total": int(age_totals[age_idx]),
                }
            )
        return rows

    def _selected_genotype_rows(self, state: PopulationState) -> list[dict[str, Any]]:
        """Return genotype cards for the selected deme."""
        deme_idx = self.selected_deme_idx if 0 <= self.selected_deme_idx < len(self.pop.demes) else 0
        deme = self.pop.demes[deme_idx]
        registry = deme.registry
        genotypes = registry.index_to_genotype
        ind_count = state.individual_count
        n_ages = ind_count.shape[1]
        deme_config = deme.export_config()
        target_age_fit = max(0, int(deme_config.new_adult_age) - 1)

        rows: list[dict[str, Any]] = []
        for genotype_idx, genotype in enumerate(genotypes):
            female_total = int(ind_count[0, :, genotype_idx].sum())
            male_total = int(ind_count[1, :, genotype_idx].sum())
            if female_total <= 0 and male_total <= 0:
                continue

            age_rows: list[dict[str, int]] = []
            if ind_count.ndim == 3:
                for age_idx in range(1, n_ages):
                    female_age = int(ind_count[0, age_idx, genotype_idx])
                    male_age = int(ind_count[1, age_idx, genotype_idx])
                    if female_age > 0 or male_age > 0:
                        age_rows.append(
                            {
                                "age": int(age_idx),
                                "female": female_age,
                                "male": male_age,
                                "total": female_age + male_age,
                            }
                        )

            rows.append(
                {
                    "genotype": genotype,
                    "female": female_total,
                    "male": male_total,
                    "fitness": self._get_genotype_fitness(deme_config, genotype_idx, target_age_fit),
                    "age_rows": age_rows,
                }
            )

        return rows

    def _get_genotype_fitness(self, config: Any, g_idx: int, target_age: int) -> dict[str, str]:
        """Return formatted viability and fecundity values for one genotype."""
        # TODO: When SpatialPopulation supports both shared and per-deme mutable
        # configuration layers, explicitly resolve invariants vs per-deme overrides here.

        v_f = config.viability_fitness[0, target_age, g_idx]
        v_m = config.viability_fitness[1, target_age, g_idx]
        f_f = config.fecundity_fitness[0, g_idx]
        f_m = config.fecundity_fitness[1, g_idx]

        result: dict[str, str] = {}
        if v_f != 1.0 or v_m != 1.0:
            result["viability"] = f"V: {v_f:.2g}(F) / {v_m:.2g}(M)"
        if f_f != 1.0 or f_m != 1.0:
            result["fecundity"] = f"F: {f_f:.2g}(F) / {f_m:.2g}(M)"
        return result

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
        # Don't automatically switch to deme tab - keep user on current view
        self._render_landscape()
        self._render_migration_panel()
        self._update_selected_deme()

    def _update_selected_deme(self) -> None:
        """Refresh the selected-deme detail panel."""
        deme = self.pop.deme(self.selected_deme_idx)
        state = deme.state
        self.lbl_selected_name.text = deme.name
        self.lbl_selected_name_detail.text = deme.name
        if self.pop.topology is not None:
            coord_text = str(self.pop.topology.from_index(self.selected_deme_idx))
        else:
            coord_text = "-"
        self.lbl_selected_coord.text = coord_text
        self.lbl_selected_coord_detail.text = coord_text

        total_text = str(deme.get_total_count())
        female_text = str(deme.get_female_count())
        male_text = str(deme.get_male_count())

        self.lbl_selected_total.text = total_text
        self.lbl_selected_females.text = female_text
        self.lbl_selected_males.text = male_text
        self.lbl_selected_total_detail.text = total_text
        self.lbl_selected_females_detail.text = female_text
        self.lbl_selected_males_detail.text = male_text
        self._render_selected_deme_summary(state)
        self._render_deme_genotypes(state)

    def _render_selected_deme_summary(self, state: PopulationState) -> None:
        """Render the selected-deme age summary card."""
        self.summary_age_container.clear()
        rows = self._selected_deme_age_rows(state)

        if not rows:
            self.age_summary_card.visible = False
            return

        self.age_summary_card.visible = True
        with self.summary_age_container:
            for row in rows:
                with ui.row().classes("w-full items-center justify-between py-1 border-b last:border-b-0"):
                    ui.label(f"Age {row['age']}").classes("font-semibold text-base text-gray-700 min-w-[5rem]")
                    with ui.row().classes("gap-3 text-sm font-mono"):
                        ui.label(f"F {row['female']:,}").classes("text-pink-600")
                        ui.label(f"M {row['male']:,}").classes("text-blue-600")
                        ui.label(f"T {row['total']:,}").classes("text-gray-800 font-semibold")

    def _render_deme_genotypes(self, state: PopulationState) -> None:
        """Render genotype cards for the selected deme."""
        self.genotype_container.clear()
        rows = self._selected_genotype_rows(state)

        with self.genotype_container:
            for row in rows:
                genotype = row["genotype"]
                with ui.card().classes("items-center p-3 border rounded shadow-sm w-44"):
                    ui.html(render_cell_svg(genotype, self.pop.species, size=72))
                    ui.label(str(genotype)).classes("text-sm font-bold text-center leading-tight text-gray-800")

                    fitness = row["fitness"]
                    if fitness:
                        with ui.column().classes("w-full items-center gap-0 my-1 bg-gray-50 rounded p-1"):
                            if "viability" in fitness:
                                ui.label(fitness["viability"]).classes("text-xs text-gray-600")
                            if "fecundity" in fitness:
                                ui.label(fitness["fecundity"]).classes("text-xs text-gray-600")

                    with ui.row().classes("w-full justify-between text-sm"):
                        ui.label(f"F: {row['female']}").classes("text-pink-600 font-semibold")
                        ui.label(f"M: {row['male']}").classes("text-blue-600 font-semibold")

                    age_rows = row["age_rows"]
                    if age_rows:
                        with ui.column().classes("w-full gap-0.5 text-xs text-gray-500"):
                            for age_row in age_rows:
                                with ui.row().classes("w-full justify-between leading-tight"):
                                    ui.label(f"A{age_row['age']}")
                                    ui.label(f"{age_row['female']}/{age_row['male']}")

    def reset_simulation(self) -> None:
        """Reset the spatial simulation and the dashboard session history."""
        self.pop.reset()
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

    def _on_show_numbers_change(self) -> None:
        """Handle checkbox change for show_numbers toggle."""
        self.show_numbers = self.chk_show_numbers.value
        self._render_landscape()

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

            ui.label("Landscape Display").classes("text-sm font-bold text-gray-400 uppercase mt-4 mb-2")
            self.chk_show_numbers = ui.checkbox(
                "Show numbers",
                value=False,
                on_change=lambda: self._on_show_numbers_change()
            ).classes("text-sm")
            if self._use_large_landscape_mode():
                self.chk_show_numbers.disable()

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
                ui.label("Coord").classes("font-bold text-gray-600")
                self.lbl_selected_coord = ui.label("").classes("text-right font-mono")
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
                        with ui.card().classes("flex-1 min-w-0 p-4 border rounded shadow-sm"):
                            ui.label("Landscape").classes("text-lg font-bold text-gray-700 mb-3")
                            self.landscape_container = ui.column().classes("w-full min-w-0 overflow-hidden gap-3")
                        with ui.card().classes("w-[26rem] p-4 border rounded shadow-sm"):
                            ui.label("Migration Rule").classes("text-lg font-bold text-gray-700 mb-3")
                            self.migration_container = ui.column().classes("w-full gap-2")

                with ui.tab_panel(tab_deme).classes("w-full"):
                    ui.label("Selected Deme State").classes("text-xl font-bold text-gray-700 mb-4")
                    with ui.row().classes("w-full gap-6 items-start"):
                        with ui.card().classes("w-[22rem] p-4 border rounded shadow-sm"):
                            ui.label("Overview").classes("text-lg font-bold text-gray-700 mb-3")
                            with ui.grid(columns=2).classes("w-full gap-y-2 gap-x-4"):
                                ui.label("Name").classes("font-bold text-gray-600")
                                self.lbl_selected_name_detail = ui.label("").classes("text-right font-mono")
                                ui.label("Coord").classes("font-bold text-gray-600")
                                self.lbl_selected_coord_detail = ui.label("").classes("text-right font-mono")
                                ui.label("Total").classes("font-bold text-gray-600")
                                self.lbl_selected_total_detail = ui.label("").classes("text-right font-mono")
                                ui.label("Females").classes("font-bold text-pink-600")
                                self.lbl_selected_females_detail = ui.label("").classes("text-right font-mono text-pink-600")
                                ui.label("Males").classes("font-bold text-blue-600")
                                self.lbl_selected_males_detail = ui.label("").classes("text-right font-mono text-blue-600")

                        self.age_summary_card = ui.card().classes("flex-1 p-4 border rounded shadow-sm")
                        with self.age_summary_card:
                            ui.label("Age Breakdown").classes("text-lg font-bold text-gray-700 mb-3")
                            self.summary_age_container = ui.column().classes("w-full gap-0")

                    with ui.card().classes("w-full p-4 border rounded shadow-sm mt-4"):
                        ui.label("Genotype Details").classes("text-lg font-bold text-gray-700 mb-3")
                        self.genotype_container = ui.row().classes("w-full flex-wrap gap-4")

        self._tick_timer = ui.timer(0.1, self._on_timer)
        self.refresh_ui()


def launch_spatial(
    population: SpatialPopulation,
    port: int = 8080,
    title: str = "NATAL Spatial Dashboard",
) -> None:
    """Launch the embedded spatial dashboard."""
    from importlib.resources import files

    # Get favicon path from package resources
    favicon_path = str(files('natal').joinpath('natal.svg'))

    @ui.page("/")
    def main_page() -> None:
        dashboard = SpatialDashboard(population)
        dashboard.build_layout()

    print(f"Starting Spatial Dashboard at http://localhost:{port}")
    ui.run(title=title, port=port, show=False, reload=False, favicon=favicon_path)
