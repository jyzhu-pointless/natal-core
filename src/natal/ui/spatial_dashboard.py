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
from natal.ui.dashboard_helpers import (
    ObservationPanel,
    get_hooks_data,
    get_unordered_genotype_labels,
    growth_mode_name,
    numpy_converter,
    render_single_hook,
)
from natal.visualization import get_allele_color, render_cell_svg


class SpatialDashboard:
    """Real-time dashboard for a ``SpatialPopulation``."""

    LARGE_LANDSCAPE_THRESHOLD = 400
    LARGE_LANDSCAPE_THRESHOLD_SQUARE = 150

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

        # Landscape cache for incremental (no-flicker) updates
        self._landscape_plot: Any = None
        self._landscape_fig: Any = None
        self._landscape_state: dict[str, Any] = {}  # structural hash

        # Observation panel (reusable component)
        # Wrap aggregate count in an object so render_observation_results can
        # detect it as raw state (not pre-applied observed data).
        class _StateRef:
            def __init__(self, pop):
                self._pop = pop

            @property
            def individual_count(self):
                return self._pop.aggregate_individual_count()

        self.obs_panel = ObservationPanel(
            genotype_labels=get_unordered_genotype_labels(
                self.pop.deme(0).registry.index_to_genotype
            ),
            get_state=lambda: _StateRef(self.pop),
            get_registry=lambda: self.pop.deme(0).registry,
        )

        self._rebuild_chart_history()
        self._record_snapshot()

    async def _run_step(self) -> None:
        """Execute one spatial simulation step."""
        if self.is_processing:
            return

        self.is_processing = True
        self.status_spinner.visible = True
        self.status_label.text = "Running..."

        try:
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
        except Exception as e:
            import traceback

            self.status_label.text = f"ERROR: {e}"
            traceback.print_exc()

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

    def _landscape_hover_label(self) -> tuple[str, callable]:
        """Return (label, format_fn) for the current landscape metric."""
        metric = getattr(self, "landscape_metric", None)
        if metric is None or metric.value == "total":
            return "Count", lambda v: f"{int(v):,}"
        return "Freq", lambda v: f"{float(v):.4f}"

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

    @staticmethod
    def _nice_ceil(value: float) -> float:
        """Round *value* up to a ``nice`` number for stable colorbar scaling.

        Always returns a number strictly greater than *value* by snapping
        the mantissa to the next step in the nice-step ladder.
        """
        import math

        if value <= 0:
            return 1.0
        exp = math.floor(math.log10(value))
        mantissa = value / (10 ** exp)
        for step in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
            if mantissa < step:
                return step * (10 ** exp)
        return 10.0 * (10 ** exp)

    def _use_large_landscape_mode(self) -> bool:
        """Return whether the landscape should use the scalable heatmap mode."""
        topology = self.pop.topology
        if topology is None:
            return False
        if isinstance(topology, HexGrid):
            return self.pop.n_demes > self.LARGE_LANDSCAPE_THRESHOLD
        # Square grids: polygon rendering is expensive with fewer cells
        return self.pop.n_demes > self.LARGE_LANDSCAPE_THRESHOLD_SQUARE

    def _build_large_landscape_figure(
        self,
        topology: Any,
        counts: list[float],
        max_count: float,
        colorbar_title: str = "Population Count",
        hover_label: str = "Count",
        format_val: object = None,
    ) -> Any:
        """Build one scalable row/col heatmap for large landscapes."""
        if format_val is None:

            def format_val(v): return f"{int(v):,}"

        z = np.zeros((topology.rows, topology.cols), dtype=np.float64)
        customdata = np.zeros((topology.rows, topology.cols), dtype=np.int64)
        text = np.empty((topology.rows, topology.cols), dtype=object)

        for row in range(topology.rows):
            for col in range(topology.cols):
                idx = topology.to_index((row, col))
                count = counts[idx]
                z[row, col] = count
                customdata[row, col] = idx
                text[row, col] = f"Index: {idx}<br>Coord: ({row}, {col})<br>{hover_label}: {format_val(count)}"

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
                        "title": colorbar_title,
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
            margin={"l": 10, "r": 10, "t": 40, "b": 60},
            uirevision="landscape",
        )
        return fig

    def _on_landscape_metric_change(self) -> None:
        """Update target dropdown options and re-render landscape."""
        metric = self.landscape_metric.value
        registry = self.pop.deme(0).registry
        if metric == "genotype":
            labels = get_unordered_genotype_labels(registry.index_to_genotype)
            options = {label: label for label in labels}
            self.landscape_target.set_options(options)
            if self.landscape_target.value not in options:
                self.landscape_target.value = next(iter(options), None)
            self.landscape_target.visible = True
        elif metric == "allele":
            options: dict[str, str] = {}
            for chrom in self.pop.species.chromosomes:
                for locus in chrom.loci:
                    for gene in locus.alleles:
                        options[gene.name] = gene.name
            self.landscape_target.set_options(options)
            if self.landscape_target.value not in options:
                self.landscape_target.value = next(iter(options), None)
            self.landscape_target.visible = True
        else:
            self.landscape_target.visible = False
        self._render_landscape()

    def _get_landscape_values(self) -> list[float]:
        """Compute per-deme values based on the selected landscape metric."""
        metric = self.landscape_metric.value if hasattr(self, "landscape_metric") else "total"

        if metric == "total":
            return [float(deme.state.individual_count.sum()) for deme in self.pop.demes]

        target_val = self.landscape_target.value if hasattr(self, "landscape_target") else None
        if target_val is None:
            return [float(deme.state.individual_count.sum()) for deme in self.pop.demes]

        if metric == "genotype":
            registry = self.pop.deme(0).registry
            # Resolve pattern (supports :: unordered, | ordered, * wildcards)
            from natal.genetic_patterns import GenotypeSelector

            selector = GenotypeSelector(self.pop.species)
            matched_indices = selector.resolve_genotype_indices(
                target_val, registry.index_to_genotype
            )
            if matched_indices:
                vals = []
                for deme in self.pop.demes:
                    ind = deme.state.individual_count
                    gt_count = float(sum(float(ind[:, :, i].sum()) for i in matched_indices))
                    total = float(ind.sum())
                    vals.append(gt_count / total if total > 0 else 0.0)
                return vals

        if metric == "allele":
            vals = []
            for deme in self.pop.demes:
                freqs = deme.compute_allele_frequencies()
                vals.append(float(freqs.get(target_val, 0.0)))
            return vals

        # Fallback
        return [float(deme.state.individual_count.sum()) for deme in self.pop.demes]

    def _render_landscape(self) -> None:
        """Render the clickable landscape geometry using plotly.

        On first render or structural change (metric, show_numbers, selected deme)
        the entire figure is rebuilt.  On data-only change (new counts), only trace
        colours and hover labels are updated in-place to avoid flicker.
        """
        topology = self.pop.topology
        counts = self._get_landscape_values()

        # Determine metric for colorbar range and title
        metric = self.landscape_metric.value if hasattr(self, "landscape_metric") else "total"
        target_val = self.landscape_target.value if hasattr(self, "landscape_target") else None

        if metric != "total":
            max_count = 1.0
        else:
            current_max = max(counts) if counts else 1.0
            if current_max > self._max_count_history:
                self._max_count_history = self._nice_ceil(current_max)
            max_count = self._max_count_history

        if metric == "genotype" and target_val:
            colorbar_title = f"Genotype Freq: {target_val}"
        elif metric == "allele" and target_val:
            colorbar_title = f"Allele Freq: {target_val}"
        else:
            colorbar_title = "Population Count"

        hover_label, format_val = self._landscape_hover_label()
        use_large = self._use_large_landscape_mode() if topology is not None else False

        # Structural state — full rebuild needed if any of these change
        struct_key = (
            metric,
            target_val,
            self.show_numbers,
            self.selected_deme_idx,
            use_large,
            type(topology).__name__ if topology else None,
            self.pop.n_demes,
        )

        if not HAS_PLOTLY:
            ui.label("Plotly is required for landscape visualization. Install with: pip install plotly").classes(
                "text-red-600 font-bold"
            )
            return

        needs_rebuild = (
            self._landscape_plot is None
            or self._landscape_fig is None
            or struct_key != self._landscape_state.get("struct_key")
        )

        if needs_rebuild:
            self.landscape_container.clear()
            self._landscape_state["struct_key"] = struct_key

            with self.landscape_container:
                if topology is None:
                    self._landscape_fig, self._landscape_plot = self._build_scatter_landscape(
                        counts, max_count, hover_label, format_val
                    )
                    return

                if use_large:
                    ui.label(
                        "Large landscape mode uses a scalable row/column heatmap for responsiveness."
                    ).classes("text-sm text-slate-500")
                    self._landscape_fig = self._build_large_landscape_figure(
                        topology=topology, counts=counts, max_count=max_count,
                        colorbar_title=colorbar_title,
                        hover_label=hover_label, format_val=format_val,
                    )
                    self._landscape_plot = ui.plotly(self._landscape_fig).classes(
                        "w-full border rounded"
                    ).props('style="height: 500px; width: 100%;"')
                    self._landscape_plot.on("plotly_click", self._on_landscape_click, ["points"])
                    return

                # Polygon (hex/square) mode — full build
                self._landscape_fig, self._landscape_plot = self._build_polygon_landscape(
                    topology, counts, max_count, colorbar_title, hover_label, format_val
                )
            return

        # --- Incremental update path ---
        if topology is None:
            self._update_scatter_landscape(counts, max_count, hover_label, format_val)
        elif use_large:
            self._update_heatmap_landscape(counts, max_count, colorbar_title, hover_label, format_val)
        else:
            self._update_polygon_landscape(counts, max_count, colorbar_title, hover_label, format_val)

        self._landscape_plot.update()

    # ------------------------------------------------------------------
    # Full-build helpers
    # ------------------------------------------------------------------

    def _build_scatter_landscape(
        self, counts: list[float], max_count: float, hover_label: str, format_val: Any
    ) -> tuple[Any, Any]:
        """Build non-topology scatter-marker landscape (first render)."""
        fig = go.Figure()
        n_demes = len(self.pop.demes)
        cols = int(np.ceil(np.sqrt(n_demes)))

        for idx, count in enumerate(counts):
            row = idx // cols
            col = idx % cols
            x, y = col, -row
            color = self._get_color_for_value(count, 0, max_count)
            hover_text = f"Index: {idx}<br>{hover_label}: {format_val(count)}"
            if self.show_numbers:
                hover_text += f"<br>{format_val(count)}"
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers",
                marker={"size": 15, "color": color},
                hovertext=hover_text, hoverinfo="text",
                customdata=[idx], showlegend=False,
            ))
            if self.show_numbers:
                fig.add_trace(go.Scatter(
                    x=[x], y=[y], mode="text",
                    text=[str(int(count))],
                    textposition="middle center",
                    textfont={"size": 10, "color": "#111827"},
                    hoverinfo="skip", showlegend=False,
                ))
            if idx == self.selected_deme_idx:
                fig.add_shape(
                    type="circle", x0=x - 0.6, y0=y - 0.6,
                    x1=x + 0.6, y1=y + 0.6,
                    line={"color": "#0f172a", "width": 3},
                )

        fig.update_layout(
            title="Landscape",
            xaxis={"showgrid": False, "zeroline": False},
            yaxis={"showgrid": False, "zeroline": False},
            hovermode="closest", height=400,
            uirevision="landscape",
        )
        plot = ui.plotly(fig).classes("w-full border rounded").props('style="height: 400px; width: 100%;"')
        plot.on("plotly_click", self._on_landscape_click, ["points"])
        return fig, plot

    def _build_polygon_landscape(
        self, topology: Any, counts: list[float], max_count: float,
        colorbar_title: str, hover_label: str, format_val: Any,
    ) -> tuple[Any, Any]:
        """Build hex/square polygon landscape (first render)."""
        is_hex = isinstance(topology, HexGrid)
        fig = go.Figure()
        hex_size = 1.0
        x_spacing = float(np.sqrt(3) * hex_size) if is_hex else 2.0
        y_spacing = 1.5 * hex_size if is_hex else 2.0

        # Per-deme traces: fill + hit_target (+ optional text)
        selected_xs: Any = None
        selected_ys: Any = None
        for row in range(topology.rows):
            for col in range(topology.cols):
                idx = topology.to_index((row, col))
                count = counts[idx]

                if is_hex:
                    x = col * x_spacing + row * (x_spacing / 2.0)
                    y = row * y_spacing
                    xs, ys = self._get_hex_vertices(x, y, hex_size)
                else:
                    x = col * x_spacing
                    y = row * y_spacing
                    xs, ys = self._get_square_vertices(x, y, hex_size)

                if idx == self.selected_deme_idx:
                    selected_xs, selected_ys = xs, ys

                color = self._get_color_for_value(count, 0, max_count)

                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines", fill="toself",
                    fillcolor=color,
                    line={"color": "rgba(100, 100, 100, 0.3)", "width": 1},
                    hoverinfo="skip", showlegend=False, name="",
                ))
                fig.add_trace(go.Scatter(
                    x=[x], y=[y], mode="markers",
                    marker={
                        "size": 64 if is_hex else 56,
                        "symbol": "hexagon" if is_hex else "square",
                        "opacity": 0.001, "color": "rgba(0,0,0,0)",
                        "line": {"width": 0},
                    },
                    hovertemplate=f"Index: {idx}<br>{hover_label}: {format_val(count)}<extra></extra>",
                    customdata=[idx], showlegend=False, name="",
                ))

                if self.show_numbers:
                    fig.add_trace(go.Scatter(
                        x=[x], y=[y], mode="text",
                        text=[str(int(count))],
                        textposition="middle center",
                        textfont={"size": 10, "color": "#111827"},
                        customdata=[idx], hoverinfo="skip", showlegend=False, name="",
                    ))

        # Selection outline — rendered after all fill traces so it stays on top.
        if selected_xs is not None:
            fig.add_trace(go.Scatter(
                x=selected_xs, y=selected_ys, fill=None,
                line={"color": "white", "width": 4},
                hoverinfo="skip", showlegend=False, name="",
            ))

        # Colorbar
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker={
                "colorscale": "Viridis", "cmid": max_count / 2,
                "colorbar": {
                    "title": colorbar_title, "thickness": 15, "len": 0.8,
                    "orientation": "h", "x": 0.5, "xanchor": "center",
                    "y": -0.15, "yanchor": "top",
                },
                "cmin": 0, "cmax": max_count, "size": 0,
            },
            showlegend=False,
        ))

        # Compute explicit axis ranges so colorbar changes cannot shift
        # the viewport through Plotly's domain-constrained autorange.
        pad = 0.05
        half = hex_size / np.sqrt(2) if not is_hex else 0.0
        x_max_data = (
            (topology.cols - 1) * x_spacing
            + (topology.rows - 1) * (x_spacing / 2.0)
        )
        y_max_data = (topology.rows - 1) * y_spacing
        margin_x = (hex_size if is_hex else half) * (1.0 + pad)
        margin_y = (hex_size if is_hex else half) * (1.0 + pad)

        fig.update_layout(
            title="Landscape",
            xaxis={
                "showgrid": False, "zeroline": False, "showticklabels": False,
                "range": [-margin_x, x_max_data + margin_x],
                "constrain": "range",
            },
            yaxis={
                "showgrid": False, "zeroline": False, "showticklabels": False,
                "range": [-margin_y, y_max_data + margin_y],
                "scaleanchor": "x", "scaleratio": 1, "constrain": "range",
            },
            hovermode="closest", height=500, transition={"duration": 0},
            margin={"l": 10, "r": 10, "t": 40, "b": 60},
            uirevision="landscape",
        )

        plot = ui.plotly(fig).classes("w-full border rounded").props('style="height: 500px; width: 100%;"')
        plot.on("plotly_click", self._on_landscape_click, ["points"])
        return fig, plot

    # ------------------------------------------------------------------
    # Incremental-update helpers
    # ------------------------------------------------------------------

    def _update_scatter_landscape(
        self, counts: list[float], max_count: float, hover_label: str, format_val: Any
    ) -> None:
        """Update scatter-marker landscape colours and hover text in-place."""
        fig = self._landscape_fig
        n_demes = len(self.pop.demes)
        trace_idx = 0
        for idx in range(n_demes):
            count = counts[idx]
            color = self._get_color_for_value(count, 0, max_count)
            hover_text = f"Index: {idx}<br>{hover_label}: {format_val(count)}"
            if self.show_numbers:
                hover_text += f"<br>{format_val(count)}"
            fig.data[trace_idx].marker.color = color
            fig.data[trace_idx].hovertext = hover_text
            trace_idx += 1
            if self.show_numbers:
                fig.data[trace_idx].text = [str(int(count))]
                trace_idx += 1
            # shapes and selection outlines are not updated incrementally;
            # a full rebuild happens when selected_deme_idx changes.

    def _update_heatmap_landscape(
        self, counts: list[float], max_count: float, colorbar_title: str,
        hover_label: str, format_val: Any,
    ) -> None:
        """Update heatmap landscape z-data and colorbar in-place."""
        fig = self._landscape_fig
        topology = self.pop.topology
        z = np.zeros((topology.rows, topology.cols), dtype=np.float64)
        text = np.empty((topology.rows, topology.cols), dtype=object)
        for row in range(topology.rows):
            for col in range(topology.cols):
                idx = topology.to_index((row, col))
                count = counts[idx]
                z[row, col] = count
                text[row, col] = f"Index: {idx}<br>Coord: ({row}, {col})<br>{hover_label}: {format_val(count)}"
        # Heatmap is the first trace
        fig.data[0].z = z
        fig.data[0].text = text
        fig.data[0].zmax = max_count
        # Colorbar is on the heatmap trace's marker
        if hasattr(fig.data[0], "colorbar") and fig.data[0].colorbar:
            fig.data[0].colorbar.title = colorbar_title
            fig.data[0].zmax = max_count

    def _update_polygon_landscape(
        self, counts: list[float], max_count: float, colorbar_title: str,
        hover_label: str, format_val: Any,
    ) -> None:
        """Update polygon landscape fill colours and hover text in-place."""
        fig = self._landscape_fig
        topology = self.pop.topology
        n_demes = topology.rows * topology.cols

        # Trace order per deme: fill, hit_target, [text if show_numbers].
        # After all demes: selection outline, colorbar dummy.
        trace_idx = 0
        for idx in range(n_demes):
            count = counts[idx]
            color = self._get_color_for_value(count, 0, max_count)

            # Fill trace
            fig.data[trace_idx].fillcolor = color
            trace_idx += 1

            # Hit target trace — update hovertemplate
            fig.data[trace_idx].hovertemplate = (
                f"Index: {idx}<br>{hover_label}: {format_val(count)}<extra></extra>"
            )
            trace_idx += 1

            # Optional: show_numbers text
            if self.show_numbers:
                fig.data[trace_idx].text = [str(int(count))]
                trace_idx += 1

        # Skip the selection outline (it was placed after all per-deme traces
        # at build time; its coords never change during the same session).
        trace_idx += 1

        # Last trace is the colorbar dummy
        cbar_idx = trace_idx
        if cbar_idx < len(fig.data):
            fig.data[cbar_idx].marker.cmid = max_count / 2
            fig.data[cbar_idx].marker.cmax = max_count
            if fig.data[cbar_idx].marker.colorbar:
                fig.data[cbar_idx].marker.colorbar.title = colorbar_title

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

    def _rebuild_chart_history(self) -> None:
        """Re-populate chart data from spatial population history.

        Reads stacked spatial history entries and computes aggregate population
        totals and allele frequencies for the global line charts.
        """
        self._chart_history = []
        self._allele_freq_history = {}
        self._last_chart_tick = -1

        if not self.pop.history:
            return

        first_deme = self.pop.deme(0)
        config = first_deme._config
        n_demes = self.pop.n_demes
        n_sexes = int(config.n_sexes)
        n_ages = int(config.n_ages)
        n_genotypes = int(config.n_genotypes)
        ind_per_deme = n_sexes * n_ages * n_genotypes
        registry = first_deme.registry

        # Collect known allele names for zero-filling
        known_alleles: set[str] = set()
        for chrom in self.pop.species.chromosomes:
            for locus in chrom.loci:
                for gene in locus.alleles:
                    known_alleles.add(gene.name)

        for tick, flat_state in self.pop.history:
            ind_size = n_demes * ind_per_deme
            ind_all = flat_state[1:1 + ind_size].reshape(n_demes, n_sexes, n_ages, n_genotypes)
            total_pop = float(ind_all.sum())

            # Aggregate genotype counts across all demes
            agg_genotype = ind_all.sum(axis=(0, 1, 2))

            # Compute allele frequencies from aggregate counts
            allele_counts: dict[str, float] = {}
            locus_totals: dict[str, float] = {}
            for g_idx, count in enumerate(agg_genotype):
                if count <= 0:
                    continue
                gt = registry.index_to_genotype[g_idx]
                for chrom in self.pop.species.chromosomes:
                    for locus in chrom.loci:
                        locus_totals.setdefault(locus.name, 0.0)
                        mat, pat = gt.get_alleles_at_locus(locus)
                        if mat:
                            allele_counts[mat.name] = allele_counts.get(mat.name, 0.0) + count
                            locus_totals[locus.name] += count
                        if pat:
                            allele_counts[pat.name] = allele_counts.get(pat.name, 0.0) + count
                            locus_totals[locus.name] += count

            freqs: dict[str, float] = {}
            for allele, count in allele_counts.items():
                gene = self.pop.species.gene_index.get(allele)
                if gene:
                    total = locus_totals.get(gene.locus.name, 0.0)
                    if total > 0:
                        freqs[allele] = count / total

            self._chart_history.append([float(tick), total_pop])

            for allele in known_alleles:
                freq = freqs.get(allele, 0.0)
                if allele not in self._allele_freq_history:
                    self._allele_freq_history[allele] = []
                self._allele_freq_history[allele].append([float(tick), freq])

            self._last_chart_tick = int(tick)

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

    def _create_meiosis_plots(self):
        """Create meiosis heatmap plots (shared across all demes)."""
        import plotly.express as px

        config = self.pop.deme(0).export_config()
        registry = self.pop.deme(0).registry
        g2g = config.genotype_to_gametes_map
        n_glabs = config.n_glabs
        genotypes = registry.index_to_genotype

        row_labels = [str(g) for g in genotypes]
        col_labels = []
        for hg_idx in range(config.n_haploid_genotypes):
            hg_obj = registry.index_to_haplo[hg_idx]
            for glab_idx in range(n_glabs):
                label = str(hg_obj)
                if n_glabs > 1:
                    label += f" [{registry.index_to_glab[glab_idx]}]"
                col_labels.append(label)

        figs = []
        for sex_idx in range(config.n_sexes):
            sex_label = "Female" if sex_idx == 0 else "Male"
            matrix = g2g[sex_idx]
            fig = px.imshow(
                matrix,
                labels={"x": "Gamete", "y": "Parent", "color": "Prob"},
                x=col_labels, y=row_labels,
                color_continuous_scale="Viridis",
                title=f"{sex_label} Meiosis",
            )
            fig.update_layout(margin={"l": 0, "r": 0, "t": 30, "b": 0}, height=300)
            figs.append(fig)
        return figs

    def _create_fertilization_plot(self):
        """Create fertilization heatmap (shared across all demes)."""
        import plotly.express as px

        config = self.pop.deme(0).export_config()
        registry = self.pop.deme(0).registry
        g2z = config.gametes_to_zygote_map
        n_hg_glabs = int(config.n_haploid_genotypes * config.n_glabs)
        genotypes = registry.index_to_genotype

        if n_hg_glabs > 40:
            return None

        labels = []
        for hg_idx in range(config.n_haploid_genotypes):
            hg_obj = registry.index_to_haplo[hg_idx]
            for glab_idx in range(config.n_glabs):
                label = str(hg_obj)
                if config.n_glabs > 1:
                    label += f" [{registry.index_to_glab[glab_idx]}]"
                labels.append(label)

        z_data = np.full((n_hg_glabs, n_hg_glabs), np.nan)
        text_data = np.full((n_hg_glabs, n_hg_glabs), "", dtype=object)

        for r in range(n_hg_glabs):
            for c in range(n_hg_glabs):
                probs = g2z[r, c, :]
                if probs.sum() < 1e-9:
                    continue
                indices = np.argsort(-probs)
                primary_idx = indices[0]
                z_data[r, c] = primary_idx
                outcomes = []
                for idx in indices:
                    p = probs[idx]
                    if p < 0.01:
                        break
                    gt_str = str(genotypes[idx])
                    outcomes.append(f"{gt_str}<br>({p:.0%})")
                text_data[r, c] = "<br>".join(outcomes)

        fig = px.imshow(
            z_data,
            labels={"x": "Paternal Gamete", "y": "Maternal Gamete", "color": "Primary Zygote"},
            x=labels, y=labels,
            color_continuous_scale="Viridis",
            title="Fertilization (Gametes → Zygote)",
        )
        fig.update_traces(
            customdata=text_data,
            hovertemplate="%{customdata}<extra></extra>",
        )
        fig.update_layout(margin={"l": 0, "r": 0, "t": 30, "b": 0}, height=500)
        return fig

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

    MAX_CHART_POINTS = 500

    def _update_charts(self) -> None:
        """Update global population and allele-frequency charts.

        Applies stride-based downsampling when history exceeds ``MAX_CHART_POINTS``.
        """
        max_pts = self.MAX_CHART_POINTS

        pop_data = self._chart_history
        if len(pop_data) > max_pts:
            stride = max(1, len(pop_data) // max_pts)
            pop_data = pop_data[::stride]
        self.chart_pop.options["series"][0]["data"] = pop_data
        self.chart_pop.update()

        series_map = {series["name"]: series for series in self.chart_allele.options["series"]}
        for allele, data in self._allele_freq_history.items():
            if len(data) > max_pts:
                stride = max(1, len(data) // max_pts)
                data = data[::stride]
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
        self._render_deme_config()

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

    def _render_deme_config(self) -> None:
        """Render configuration for the currently selected deme."""
        if not hasattr(self, "config_container") or self.config_container is None:
            return
        self.config_container.clear()

        deme = self.pop.deme(self.selected_deme_idx)
        config = deme.export_config()
        registry = deme.registry
        genotypes = registry.index_to_genotype

        with self.config_container:
            with ui.card().classes("p-4 border rounded shadow-sm"):
                ui.label(f"Deme {self.selected_deme_idx} Configuration").classes("text-lg font-bold text-gray-700 mb-4")

                with ui.row().classes("w-full gap-8"):
                    with ui.column().classes("w-1/3"):
                        ui.label("Scalar Parameters").classes("font-bold text-gray-600 mb-2")
                        with ui.grid(columns=2).classes("w-full gap-y-1 gap-x-4"):
                            for label_text, value in [
                                ("Carrying Capacity", float(config.carrying_capacity)),
                                ("Eggs per Female", float(config.expected_eggs_per_female)),
                                ("Growth Mode", growth_mode_name(int(config.juvenile_growth_mode))),
                                ("Stochastic", bool(config.is_stochastic)),
                                ("Sex Ratio", float(config.sex_ratio)),
                                ("Low Density Growth", float(config.low_density_growth_rate)),
                                ("Population Scale", float(config.population_scale)),
                                ("New Adult Age", int(config.new_adult_age)),
                                ("Sperm Displacement", float(config.sperm_displacement_rate)),
                                ("N Sexes", int(config.n_sexes)),
                                ("N Ages", int(config.n_ages)),
                                ("N Genotypes", int(config.n_genotypes)),
                            ]:
                                ui.label(label_text).classes("font-bold text-gray-500 text-sm")
                                ui.label(str(value)).classes("text-right font-mono text-sm")

                    with ui.column().classes("flex-grow"):
                        target_age = max(0, int(config.new_adult_age) - 1)

                        # Viability fitness table
                        via_rows = []
                        for g_idx, g_obj in enumerate(genotypes):
                            f_val = float(config.viability_fitness[0, target_age, g_idx])
                            m_val = float(config.viability_fitness[1, target_age, g_idx])
                            if f_val != 1.0 or m_val != 1.0:
                                via_rows.append({"Genotype": str(g_obj), "Female": f_val, "Male": m_val})
                        if via_rows:
                            ui.label("Viability Fitness").classes("font-bold text-gray-600 mb-2")
                            ui.table(
                                columns=[
                                    {"name": "Genotype", "label": "Genotype", "field": "Genotype"},
                                    {"name": "Female", "label": "Female", "field": "Female"},
                                    {"name": "Male", "label": "Male", "field": "Male"},
                                ],
                                rows=via_rows,
                            ).props("dense flat").classes("mb-4 w-full")

                        # Fecundity fitness table
                        fec_rows = []
                        for g_idx, g_obj in enumerate(genotypes):
                            f_val = float(config.fecundity_fitness[0, g_idx])
                            m_val = float(config.fecundity_fitness[1, g_idx])
                            if f_val != 1.0 or m_val != 1.0:
                                fec_rows.append({"Genotype": str(g_obj), "Female": f_val, "Male": m_val})
                        if fec_rows:
                            ui.label("Fecundity Fitness").classes("font-bold text-gray-600 mb-2")
                            ui.table(
                                columns=[
                                    {"name": "Genotype", "label": "Genotype", "field": "Genotype"},
                                    {"name": "Female", "label": "Female", "field": "Female"},
                                    {"name": "Male", "label": "Male", "field": "Male"},
                                ],
                                rows=fec_rows,
                            ).props("dense flat").classes("w-full")

    def _render_hooks_panel(self) -> None:
        """Display all hooks registered on the spatial population."""
        if not hasattr(self, "hooks_container") or self.hooks_container is None:
            return
        self.hooks_container.clear()

        with self.hooks_container:
            # Collect per-deme hooks
            per_deme_hooks: list[tuple[int, Any]] = []
            for deme_id, deme in enumerate(self.pop.demes):
                for desc in deme.get_compiled_hooks():
                    per_deme_hooks.append((deme_id, desc))

            if not per_deme_hooks:
                ui.label("No hooks registered.").classes("text-gray-500 italic")
                return

            # Identify global (registered on all demes with "*" selector) vs local
            global_hooks: dict[str, list[tuple[int, Any]]] = {}
            local_hooks: list[tuple[int, Any]] = []

            for d_id, desc in per_deme_hooks:
                # A hook is global if its deme_selector is "*"
                if getattr(desc, "deme_selector", None) == "*":
                    global_hooks.setdefault(desc.name, []).append((d_id, desc))
                else:
                    local_hooks.append((d_id, desc))

            if global_hooks:
                ui.label("Global Hooks (applied to all demes)").classes("text-lg font-bold text-green-700 mt-2")
                seen: set[str] = set()
                for _, desc in per_deme_hooks:
                    if getattr(desc, "deme_selector", None) == "*" and desc.name not in seen:
                        seen.add(desc.name)
                        render_single_hook(desc, is_global=True)

            if local_hooks:
                ui.label("Per-Deme Hooks").classes("text-lg font-bold text-blue-700 mt-2")
                # Group by deme
                by_deme: dict[int, list[Any]] = {}
                for d_id, desc in local_hooks:
                    by_deme.setdefault(d_id, []).append(desc)
                for d_id in sorted(by_deme):
                    ui.label(f"Deme {d_id}").classes("text-md font-semibold text-gray-700 mt-2")
                    for desc in by_deme[d_id]:
                        render_single_hook(desc, is_global=False)

    def show_export_dialog(self) -> None:
        """Open export dialog for spatial data."""
        with ui.dialog() as self.export_dialog, ui.card():
            ui.label("Select items to export").classes("text-lg font-bold")
            self.cb_config = ui.checkbox("Configuration & Fitness", value=True)
            self.cb_history = ui.checkbox("Population History", value=True)
            self.cb_hooks = ui.checkbox("Hooks", value=True)
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Export", on_click=self._do_export)
                ui.button("Cancel", on_click=self.export_dialog.close).props("flat")
        self.export_dialog.open()

    def _do_export(self) -> None:
        """Handle export button click in the dialog."""
        include_config = self.cb_config.value
        include_history = self.cb_history.value
        include_hooks = self.cb_hooks.value
        self.export_dialog.close()
        self._do_export_logic(include_config, include_history, include_hooks)
        ui.notify("Export started...")

    def _do_export_logic(
        self, include_config: bool, include_history: bool, include_hooks: bool
    ) -> None:
        """Core export logic for spatial data."""
        import json

        export_content = {"population_name": self.pop.name}

        if include_history:
            from natal.state_translation import spatial_population_output_history

            history_payload = spatial_population_output_history(self.pop)
            export_content["history"] = history_payload

        if include_config:
            first_config = self.pop.deme(0).export_config()
            export_content["configuration"] = {
                "parameters": {
                    "n_demes": self.pop.n_demes,
                    "migration_mode": self.pop.migration_mode,
                    "migration_rate": float(self.pop.migration_rate),
                    "carrying_capacity": float(first_config.carrying_capacity),
                    "n_sexes": int(first_config.n_sexes),
                    "n_ages": int(first_config.n_ages),
                    "n_genotypes": int(first_config.n_genotypes),
                    "sex_ratio": float(first_config.sex_ratio),
                    "eggs_per_female": float(first_config.expected_eggs_per_female),
                    "growth_mode": growth_mode_name(int(first_config.juvenile_growth_mode)),
                },
            }

        if include_hooks:
            export_content["hooks"] = get_hooks_data(self.pop)

        try:
            json_str = json.dumps(export_content, default=numpy_converter)
            ui.download(
                json_str.encode("utf-8"),
                filename=f"natal_spatial_export_{self.pop.name}_tick{self.pop.tick}.json",
                media_type="application/json",
            )
        except Exception as e:
            ui.notify(f"Export failed: {e}", type="negative")

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

                with ui.row().classes("w-full gap-2"):
                    ui.button("Reset", on_click=self.reset_simulation).props("icon=restart_alt flat color=grey").classes("flex-grow")
                    ui.button("Export", on_click=self.show_export_dialog).props("icon=download flat color=grey").classes("flex-grow")

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
                tab_config = ui.tab(name="config", label="Config", icon="settings")
                tab_hooks = ui.tab(name="hooks", label="Hooks", icon="extension")
                tab_genetics = ui.tab(name="genetics", label="Genetics", icon="biotech")
                tab_observation = ui.tab(name="observation", label="Observation", icon="visibility")

            with ui.tab_panels(tabs, value="overview").classes("w-full bg-transparent p-0"):
                with ui.tab_panel(tab_overview).classes("w-full"):
                    with ui.row().classes("w-full gap-6 items-start"):
                        with ui.card().classes("flex-1 min-w-0 p-4 border rounded shadow-sm"):
                            with ui.row().classes("items-center gap-4 mb-3"):
                                ui.label("Landscape").classes("text-lg font-bold text-gray-700")
                                self.landscape_metric = ui.select(
                                    label="Metric",
                                    options={
                                        "total": "Total Population",
                                        "genotype": "Genotype Frequency",
                                        "allele": "Allele Frequency",
                                    },
                                    value="total",
                                    on_change=self._on_landscape_metric_change,
                                ).classes("w-48")
                                self.landscape_target = ui.select(
                                    label="Target",
                                    options={},
                                    value=None,
                                    on_change=self._on_landscape_metric_change,
                                    new_value_mode="add",
                                    key_generator=lambda x: x,
                                ).classes("w-48")
                                self.landscape_target.visible = False
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

                with ui.tab_panel(tab_config).classes("w-full"):
                    self.config_container = ui.column().classes("w-full gap-6")
                    self._render_deme_config()

                with ui.tab_panel(tab_hooks).classes("w-full"):
                    self.hooks_container = ui.column().classes("w-full gap-4")
                    self._render_hooks_panel()

                with ui.tab_panel(tab_genetics).classes("w-full"):
                    with ui.column().classes("w-full gap-6"):
                        ui.label("Meiosis (Genotype → Gametes)").classes("font-bold text-gray-700 text-xl")
                        figs = self._create_meiosis_plots()
                        with ui.row().classes("w-full gap-4"):
                            for fig in figs:
                                ui.plotly(fig).classes("flex-1 h-[600px] border rounded")

                        ui.label("Fertilization (Gametes → Zygote)").classes("font-bold text-gray-700 text-xl mt-4")
                        fig_fert = self._create_fertilization_plot()
                        if fig_fert:
                            ui.plotly(fig_fert).classes("w-full border rounded").props('style="height: 600px;"')
                        else:
                            ui.label("Fertilization matrix too large to display.").classes("text-orange-500 italic")

                with ui.tab_panel(tab_observation).classes("w-full"):
                    self.obs_panel.build(ui.column())

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
