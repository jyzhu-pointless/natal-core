"""
NiceGUI-based Dashboard for NATAL populations.

This module provides a web-based control panel that can be launched directly
from a simulation script. It runs in a separate thread (or manages the main loop)
and accesses the population object directly in memory.
"""

import threading
import time
import webbrowser
from typing import Optional, TYPE_CHECKING, List, Dict, Tuple
import pandas as pd
import numpy as np
import inspect
import json
import bisect

try:
    from nicegui import ui, app, run
    HAS_NICEGUI = True
except ImportError:
    HAS_NICEGUI = False

from natal.visualization import render_cell_svg, get_allele_color
from natal.index_registry import decompress_hg_glab
from natal.population_state import parse_flattened_state, parse_flattened_discrete_state, PopulationState


if TYPE_CHECKING:
    from natal.base_population import BasePopulation

class Dashboard:
    """
    A real-time dashboard for controlling and visualizing a NATAL population.
    """
    
    def __init__(self, population: 'BasePopulation'):
        if not HAS_NICEGUI:
            raise ImportError("NiceGUI is required. Please install it with: pip install nicegui")
        
        self.pop = population
        self.is_running = False
        self.is_processing = False
        self._tick_timer = None
        
        # UI Elements state
        self._chart_history: List[Dict] = []
        self._allele_freq_history: Dict[str, List[List]] = {}
        self.max_chart_points = 500  # Control total points for sparse display
        
        # Reconstruct chart history from existing population history (fixes data loss on reload)
        self.view_min: Optional[float] = None
        self.view_max: Optional[float] = None
        
        self._last_chart_tick = -1
        self._rebuild_chart_history()
        self.inspected_tick: Optional[int] = None
        self.inspection_mode = False # False = Current, True = History
        
    async def _run_step(self):
        """Execute one simulation step."""
        if self.is_processing:
            return
            
        self.is_processing = True
        self.status_spinner.visible = True
        self.status_label.text = "Compiling/Running..."
        
        # If we were inspecting history, switch back to live view on step
        if self.inspection_mode:
            self.inspection_mode = False
            self.inspected_tick = None
            self.tabs_main.set_value('inspection')
        
        if not self.pop.is_finished:
            # Run blocking simulation step in a separate thread executor
            # This allows the UI loop (including spinner) to keep running
            if self.slider_speed.value <= 0:
                # Turbo mode: Run in batches (render periodically)
                # Execute ticks for ~100ms before yielding back to UI
                def run_batch():
                    start = time.time()
                    # Run until 100ms passed, finished, or batch limit reached
                    ticks = 0
                    while time.time() - start < 0.1 and not self.pop.is_finished and ticks < 50:
                        self.pop.run_tick()
                        ticks += 1
                
                await run.io_bound(run_batch)
            else:
                # Normal mode: Single tick
                await run.io_bound(self.pop.run_tick)
            
            self.refresh_ui()
        
        if self.pop.is_finished:
            self.is_running = False
            self.btn_play.props('icon=play_arrow')
            self.btn_play.text = "Play"
            self.status_label.text = "Finished"

        if not self.pop.is_finished:
            self.status_label.text = "Ready"
            
        self.status_spinner.visible = False
        self.is_processing = False
            
    def _toggle_play(self):
        """Toggle auto-play."""
        self.is_running = not self.is_running
        if self.is_running:
            # If starting play, exit inspection mode
            self.inspection_mode = False
            self.inspected_tick = None
            self.tabs_main.set_value('inspection')

    async def _on_timer(self):
        """Called periodically by UI loop."""
        if self.is_running:
            await self._run_step()

    def _update_timer_interval(self):
        """Update timer interval based on slider value."""
        val = self.slider_speed.value
        # If 0 (Turbo), run timer frequently to drive the batch loop
        # If > 0, use value as delay
        self._tick_timer.interval = 0.01 if val <= 0 else val

    def _compute_metrics_from_flat(self, tick: int, flat_state: np.ndarray) -> Tuple[int, Dict[str, float]]:
        """Extract total population and allele frequencies from a flattened state array."""
        config = self.pop._config
        registry = self.pop.registry
        n_sexes = config.n_sexes
        n_ages = config.n_ages
        n_genotypes = config.n_genotypes
        
        # Determine indices
        # flat_state structure: [tick, individual_count(flattened), sperm_storage(flattened)...]
        start_idx = 1
        end_idx = 1 + n_sexes * n_ages * n_genotypes
        
        # Extract individual_count
        if flat_state.size < end_idx:
            return 0, {} # Should not happen
            
        ind_flat = flat_state[start_idx:end_idx]
        total_pop = int(np.sum(ind_flat))
        
        # Compute allele frequencies
        # Reshape to (n_sexes, n_ages, n_genotypes) then sum to (n_genotypes,)
        ind_reshaped = ind_flat.reshape((n_sexes, n_ages, n_genotypes))
        genotype_counts = ind_reshaped.sum(axis=(0, 1))
        
        freqs = {}
        # We can reuse BasePopulation logic but optimized for numpy
        # Since we don't have easy access to gene maps here without looping, 
        # we use the registry. This is slower than pure C but acceptable for UI.
        # To optimize, we could cache gene-to-genotype maps.
        
        # Re-use the pop's logic by temporarily setting state? No, that's unsafe.
        # For now, let's use the BasePopulation.compute_allele_frequencies logic 
        # but operating on our extracted genotype_counts.
        
        # For performance in "Turbo" mode, we might skip detailed allele freq 
        # on every single intermediate point if it's too slow, but let's try full fidelity first.
        # Calling the population method requires state injection. Let's reimplement lightweight version.
        
        # Optimization: Just return total_pop for now, and rely on `pop.compute_allele_frequencies`
        # for the 'current' state in _update_charts if history parsing is too slow.
        # BUT the user wants ZOOM. So we must compute it.
        
        # Let's map genotype_counts to allele counts
        allele_counts = {}
        locus_totals = {}
        
        for g_idx, count in enumerate(genotype_counts):
            if count <= 0: continue
            gt = registry.index_to_genotype[g_idx]
            
            # We need to access alleles. 
            # This implies object access.
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

        for allele, count in allele_counts.items():
            # Find locus total. We need to know which locus this allele belongs to.
            # We can find it via species.
            gene = self.pop.species.gene_index.get(allele)
            if gene:
                total = locus_totals.get(gene.locus.name, 0.0)
                if total > 0:
                    freqs[allele] = count / total
        
        return total_pop, freqs

    def _rebuild_chart_history(self):
        """Re-populate chart data structures from population history."""
        # Clear current
        self._chart_history = []
        self._allele_freq_history = {}
        self._last_chart_tick = -1
        
        if not self.pop.history:
            return
            
        # We need to compute total counts and allele freqs from history snapshots.
        # This can be expensive if history is huge, but max_history limits it.
        # Only process every Nth point if history is huge to save UI load time
        # But max_history is usually small (5000), so process all.
        
        for tick, flat_state in self.pop.history:
            total_pop, freqs = self._compute_metrics_from_flat(tick, flat_state)
            self._chart_history.append([tick, total_pop])
            
            # Allele Frequencies
            for allele, freq in freqs.items():
                if allele not in self._allele_freq_history:
                    self._allele_freq_history[allele] = []
                self._allele_freq_history[allele].append([tick, freq])
            
            if tick > self._last_chart_tick:
                self._last_chart_tick = tick

    def refresh_ui(self):
        """Update reactive UI elements."""
        # Determine which state to show
        if self.inspection_mode and self.inspected_tick is not None:
            # Inspecting history - handled by inspect_tick, but we might need to update static labels if we want
            return

        # Showing current state
        self._update_inspection_view(self.pop.state, self.pop.tick, is_history=False)
        self._update_charts()

    def _update_charts(self):
        """Update chart data from current population state."""
        # Initialize allele series structures if missing
        # (We do this early to ensure series order is stable)
        for allele in self._allele_freq_history.keys():
            series = next((s for s in self.chart_allele.options['series'] if s['name'] == allele), None)
            if not series:
                self.chart_allele.options['series'].append({
                    'name': allele, 'data': [], 'color': get_allele_color(allele)
                })

        # 1. Collect all NEW snapshots from population history into local full-resolution buffers
        # pop.history is a list of (tick, flat_state)
        # We need to find where new data starts.
        # Optim: check last item. If matching, verify backwards? 
        # Simple approach: iterate history.
        
        new_points = []
        
        # Access private history for speed/consistency
        history = self.pop._history 
        
        # Optimization: start search from the end or assume append-only
        # If history was cleared/rotated (max_history), we might need to handle gaps or full refresh
        # For now, just scan for tick > last.
        
        for tick, flat_state in history:
            if tick > self._last_chart_tick:
                total_pop, freqs = self._compute_metrics_from_flat(tick, flat_state)
                new_points.append((tick, total_pop, freqs))
                self._last_chart_tick = tick
        
        # If current live state is newer than anything in history (e.g. record_every > 1), add it too
        if self.pop.tick > self._last_chart_tick:
            freqs = self.pop.compute_allele_frequencies()
            new_points.append((self.pop.tick, self.pop.get_total_count(), freqs))
            self._last_chart_tick = self.pop.tick

        # 2. Append new points to internal full-history buffers
        if new_points:
            for tick, total, freqs in new_points:
                self._chart_history.append([tick, total])
                
                for allele, freq in freqs.items():
                    if allele not in self._allele_freq_history:
                        self._allele_freq_history[allele] = []
                        # Add series if it appeared mid-simulation
                        self.chart_allele.options['series'].append({
                            'name': allele, 'data': [], 'color': get_allele_color(allele)
                        })
                    self._allele_freq_history[allele].append([tick, freq])

        # 3. Downsample and update charts
        # We always replace the chart data with a strided view of the full history
        # This ensures the UI remains responsive (sparse display) even with huge datasets
        
        if not self._chart_history:
            return

        # Determine slice indices based on zoom
        if self.view_min is not None and self.view_max is not None:
            # Zoomed-in view: slice the history and show more detail
            idx_start = bisect.bisect_left(self._history_ticks, self.view_min)
            idx_end = bisect.bisect_right(self._history_ticks, self.view_max)
            # Cap the number of points in a zoomed view to avoid freezing
            count = idx_end - idx_start
            stride = max(1, count // 2000) # Allow up to 2000 points in a zoomed view
        else:
            # Full view: downsample to max_chart_points
            idx_start = 0
            idx_end = len(self._chart_history)
            count = idx_end
            stride = max(1, count // self.max_chart_points)
            
        # Ensure valid range
        idx_start = max(0, idx_start)
        idx_end = min(len(self._chart_history), idx_end)
        
        # Update Population Chart
        # Slicing with stride: [start:end:step]
        self.chart_pop.options['series'][0]['data'] = self._chart_history[idx_start:idx_end:stride]
        self.chart_pop.update()
        
        # Update Allele Freq Chart
        for s in self.chart_allele.options['series']:
            allele = s['name']
            if allele in self._allele_freq_history:
                # Ensure the allele history is also sliced and strided correctly
                full_allele_data = self._allele_freq_history[allele]
                # We need to find the corresponding slice for allele data.
                # This is tricky if alleles appear/disappear. A safer way is to rebuild.
                # For simplicity, we assume the history lists are aligned.
                s['data'] = full_allele_data[idx_start:idx_end:stride]
        self.chart_allele.update()
        
    def _update_record_every(self, e):
        """Update population record_every setting."""
        if e.value is not None:
            self.pop.record_every = int(e.value)
            
    def _update_max_history(self, e):
        """Update population max_history setting."""
        if e.value is not None:
            self.pop.max_history = int(e.value)

    def _update_inspection_view(self, state: 'PopulationState', tick: int, is_history: bool = False):
        """Update the Inspection tab with details from a specific state."""
        # Update Header Stats
        total = int(state.individual_count.sum())
        females = int(state.individual_count[0].sum())
        males = int(state.individual_count[1].sum())
        
        self.lbl_tick.text = f"{tick}"
        self.lbl_total.text = f"{total}"
        self.lbl_females.text = f"{females}"
        self.lbl_males.text = f"{males}"
        self.lbl_history_count.text = f'(Current: {len(self.pop.history)} snapshots)'
        
        if is_history:
            self.lbl_status_mode.text = f"INSPECTING HISTORY (Tick {tick})"
            self.lbl_status_mode.classes(add='text-orange-600', remove='text-green-600')
        else:
            self.lbl_status_mode.text = "LIVE VIEW"
            self.lbl_status_mode.classes(add='text-green-600', remove='text-orange-600')

        # Update Genotype Cards
        # Clear existing
        self.genotype_container.clear()
        
        config = self.pop.export_config()
        registry = self.pop.registry
        genotypes = registry.index_to_genotype
        
        # Calculate counts for this specific state
        # individual_count: (n_sex, n_ages, n_genotypes)
        ind_count = state.individual_count
        n_ages = ind_count.shape[1]
        
        # For fitness display
        conf = self.pop.export_config()
        target_age_fit = max(0, int(conf.new_adult_age) - 1)

        with self.genotype_container:
            for i, gt in enumerate(genotypes):
                # Compute total for card summary
                total_f = int(ind_count[0, :, i].sum())
                total_m = int(ind_count[1, :, i].sum())
                
                with ui.card().classes('items-center p-2 border rounded shadow-sm w-40'):
                    # SVG
                    svg = render_cell_svg(gt, self.pop.species, size=80)
                    ui.html(svg)
                    # Name
                    ui.label(str(gt)).classes('text-xs font-bold mt-1 text-center leading-tight text-gray-800')
                    
                    # Fitness (NEW)
                    fit_info = self._get_genotype_fitness(i, target_age_fit)
                    if fit_info:
                        with ui.column().classes('w-full items-center gap-0 my-1 bg-gray-50 rounded p-1'):
                            if 'via' in fit_info:
                                ui.label(fit_info['via']).classes('text-xs text-gray-600')
                            if 'fec' in fit_info:
                                ui.label(fit_info['fec']).classes('text-xs text-gray-600')

                    # Counts
                    with ui.row().classes('w-full justify-between px-1 mt-1'):
                        ui.label(f"F: {total_f}").classes('text-sm font-bold text-pink-600')
                        ui.label(f"M: {total_m}").classes('text-sm font-bold text-blue-600')
                    
                    # Detailed age breakdown (skipping Age 0)
                    if n_ages > 1:
                        with ui.column().classes('w-full gap-0 mt-1'):
                            # Iterate ages starting from 1
                            for age in range(1, n_ages):
                                af = int(ind_count[0, age, i])
                                am = int(ind_count[1, age, i])
                                if af > 0 or am > 0:
                                    with ui.row().classes('w-full justify-between text-xs text-gray-500 leading-tight'):
                                        ui.label(f"A{age}")
                                        ui.label(f"{af}/{am}")

    def _get_viability_data(self):
        config = self.pop.export_config()
        registry = self.pop.registry
        genotypes = registry.index_to_genotype
        # Typically viability selection happens at new_adult_age - 1 (late juvenile)
        target_age = max(0, int(config.new_adult_age) - 1)
        
        data = []
        for g_idx, g_obj in enumerate(genotypes):
            f_val = config.viability_fitness[0, target_age, g_idx]
            m_val = config.viability_fitness[1, target_age, g_idx]
            if f_val != 1.0 or m_val != 1.0 or "Dr" in str(g_obj) or "Drive" in str(g_obj):
                data.append({
                    "Genotype": str(g_obj),
                    "Female": f"{f_val:.3g}",
                    "Male": f"{m_val:.3g}"
                })
        return data

    def _get_fecundity_data(self):
        config = self.pop.export_config()
        registry = self.pop.registry
        genotypes = registry.index_to_genotype
        
        data = []
        for g_idx, g_obj in enumerate(genotypes):
            f_val = config.fecundity_fitness[0, g_idx]
            m_val = config.fecundity_fitness[1, g_idx]
            if f_val != 1.0 or m_val != 1.0 or "Dr" in str(g_obj) or "Drive" in str(g_obj):
                data.append({
                    "Genotype": str(g_obj),
                    "Female": f"{f_val:.3g}",
                    "Male": f"{m_val:.3g}"
                })
        return data

    def _create_meiosis_plots(self):
        import plotly.express as px
        config = self.pop.export_config()
        registry = self.pop.registry
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
            fig = px.imshow(matrix,
                            labels=dict(x="Gamete", y="Parent", color="Prob"),
                            x=col_labels, y=row_labels,
                            color_continuous_scale="Viridis",
                            title=f"{sex_label} Meiosis")
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300)
            figs.append(fig)
        return figs

    def _create_fertilization_plot(self):
        import plotly.express as px
        config = self.pop.export_config()
        registry = self.pop.registry
        g2z = config.gametes_to_zygote_map
        n_hg_glabs = int(config.n_haploid_genotypes * config.n_glabs)
        genotypes = registry.index_to_genotype
        
        if n_hg_glabs > 40:
            return None

        # Prepare labels for gametes (axes)
        labels = []
        for hg_idx in range(config.n_haploid_genotypes):
            hg_obj = registry.index_to_haplo[hg_idx]
            for glab_idx in range(config.n_glabs):
                label = str(hg_obj)
                if config.n_glabs > 1:
                    label += f" [{registry.index_to_glab[glab_idx]}]"
                labels.append(label)

        # Build matrices for heatmap
        # z_data: numeric index of the primary zygote (for coloring)
        # text_data: formatted string of all zygote outcomes (for display)
        z_data = np.full((n_hg_glabs, n_hg_glabs), np.nan)
        text_data = np.full((n_hg_glabs, n_hg_glabs), "", dtype=object)
        
        for r in range(n_hg_glabs): # Maternal gamete
            for c in range(n_hg_glabs): # Paternal gamete
                probs = g2z[r, c, :]
                if probs.sum() < 1e-9:
                    continue
                
                # Sort outcomes by probability
                indices = np.argsort(-probs)
                primary_idx = indices[0]
                z_data[r, c] = primary_idx
                
                outcomes = []
                for idx in indices:
                    p = probs[idx]
                    if p < 0.01: break # Skip <1% outcomes
                    gt_str = str(genotypes[idx])
                    outcomes.append(f"{gt_str}<br>({p:.0%})")
                
                text_data[r, c] = "<br>".join(outcomes)

        fig = px.imshow(
            z_data, 
            x=labels, 
            y=labels,
            labels=dict(x="Paternal Gamete", y="Maternal Gamete", color="Zygote ID"),
            color_continuous_scale="Viridis",
            title="Fertilization Outcomes (Maternal x Paternal)"
        )
        # Show full text inside grid cells
        fig.update_traces(text=text_data, texttemplate="%{text}")
        fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0), 
            height=max(500, n_hg_glabs * 50),
            xaxis_tickangle=-45
        )
        return fig

    def reset_simulation(self):
        """Reset the population and UI state."""
        self.pop.reset()
        self.is_running = False
        self.inspection_mode = False
        self.inspected_tick = None
        self._last_chart_tick = -1
        self.tabs_main.set_value('inspection')
        
        # Reset UI controls
        if hasattr(self, 'btn_play'):
            self.btn_play.props('icon=play_arrow')
            self.btn_play.text = "Play"
        if hasattr(self, 'status_label'):
            self.status_label.text = "Ready"
        
        # Reset charts
        self.chart_pop.options['series'][0]['data'] = []
        self.chart_allele.options['series'] = []
        self._chart_history = []
        self._allele_freq_history = {}
        self.chart_pop.update()
        self.chart_allele.update()
        
        # Refresh to show initial state
        self.refresh_ui()
        ui.notify("Population reset to initial state.")

    def export_data(self):
        """Export full config and current state to JSON."""

        def numpy_converter(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        def semanticize_state(state):
            registry = self.pop.registry
            genotypes = [str(g) for g in registry.index_to_genotype]
            
            # Handle discrete (2D) vs age-structured (3D)
            if state.individual_count.ndim == 3:
                n_ages = state.individual_count.shape[1]
                
                state_dict = {
                    "tick": int(state.n_tick),
                    "individual_count": { "female": {}, "male": {} }
                }

                for age in range(n_ages):
                    female_counts = {}
                    male_counts = {}
                    for g_idx, g_str in enumerate(genotypes):
                        f_count = state.individual_count[0, age, g_idx]
                        m_count = state.individual_count[1, age, g_idx]
                        if f_count > 0:
                            female_counts[g_str] = f_count
                        if m_count > 0:
                            male_counts[g_str] = m_count
                    if female_counts:
                        state_dict["individual_count"]["female"][age] = female_counts
                    if male_counts:
                        state_dict["individual_count"]["male"][age] = male_counts
                return state_dict
            else:
                # Fallback for unexpected shapes, though usually it's 3D even for discrete
                return {"tick": int(state.n_tick), "raw_shape": state.individual_count.shape}
        
        # Reconstruct history
        history_list = []
        n_sexes = self.pop._config.n_sexes
        n_ages = self.pop._config.n_ages
        n_genotypes = self.pop._config.n_genotypes
        expected_ind_size = int(n_sexes * n_ages * n_genotypes)

        for tick, flat_state in self.pop.history:
            if flat_state.size == 1 + expected_ind_size:
                state_obj = parse_flattened_discrete_state(flat_state, n_sexes, n_ages, n_genotypes, copy=False)
            else:
                state_obj = parse_flattened_state(flat_state, n_sexes, n_ages, n_genotypes, copy=False)
            
            history_list.append(semanticize_state(state_obj))
        
        # Also add current state if not in history (e.g. at tick 0 before first record)
        if not history_list or history_list[-1]["tick"] != self.pop.tick:
             history_list.append(semanticize_state(self.pop.state))

        export_content = {
            "population_name": self.pop.name,
            "current_state": semanticize_state(self.pop.state),
            "history": history_list,
            "configuration": {
                "parameters": self._get_config_scalars(),
                "fitness": {
                    "viability": self._get_viability_data(),
                    "fecundity": self._get_fecundity_data()
                }
            },
        }

        try:
            json_str = json.dumps(export_content, indent=2, default=numpy_converter)
            ui.download(json_str.encode('utf-8'),
                        filename=f"natal_export_{self.pop.name}_{self.pop.tick}.json",
                        media_type='application/json')
        except Exception as e:
            ui.notify(f"Export failed: {e}", type='negative')

    def handle_chart_click(self, e):
        """Handle click on chart points to inspect history."""
        # e.point_x contains the tick value
        if e.point_x is None:
            return
            
        tick = int(e.point_x)
        self.inspect_tick(tick)

    def handle_tick_input(self, e):
        """Handle manual tick input."""
        if e.value is None:
            return
        self.inspect_tick(int(e.value))

    def _get_config_scalars(self):
        conf = self.pop._config
        return {
            "is_stochastic": conf.is_stochastic,
            "use_dirichlet_sampling": conf.use_dirichlet_sampling,
            "n_sexes": conf.n_sexes,
            "n_ages": conf.n_ages,
            "n_genotypes": conf.n_genotypes,
            "new_adult_age": conf.new_adult_age,
            "sperm_displacement_rate": conf.sperm_displacement_rate,
            "expected_eggs_per_female": conf.expected_eggs_per_female,
            "use_fixed_egg_count": conf.use_fixed_egg_count,
            "carrying_capacity": conf.carrying_capacity,
            "sex_ratio": conf.sex_ratio,
            "low_density_growth_rate": conf.low_density_growth_rate,
            "juvenile_growth_mode": conf.juvenile_growth_mode,
        }

    def _get_genotype_fitness(self, g_idx: int, target_age: int) -> dict:
        """Helper to get formatted fitness strings for a genotype."""
        config = self.pop.export_config()
        
        v_f = config.viability_fitness[0, target_age, g_idx]
        v_m = config.viability_fitness[1, target_age, g_idx]
        f_f = config.fecundity_fitness[0, g_idx]
        f_m = config.fecundity_fitness[1, g_idx]
        
        res = {}
        if v_f != 1.0 or v_m != 1.0:
            res['via'] = f"V: {v_f:.2g}(F)/{v_m:.2g}(M)"
        if f_f != 1.0 or f_m != 1.0:
            res['fec'] = f"F: {f_f:.2g}(F)/{f_m:.2g}(M)"
        return res

    def handle_chart_zoom(self, e):
        """Handle Highcharts 'selection' event to update view window."""
        # e.args format for selection event:
        # { 'xAxis': [ {'min': float, 'max': float, ...} ], ... } if zooming
        # { ... } (no xAxis) if resetting zoom
        
        # Highcharts on client side handles the visual zoom.
        # We update our internal view state and trigger a data refresh
        # to load higher resolution data for the selected range.
        if e.args and 'xAxis' in e.args and e.args['xAxis']:
            self.view_min = e.args['xAxis'][0]['min']
            self.view_max = e.args['xAxis'][0]['max']
        else:
            self.view_min = None
            self.view_max = None
        self._update_charts()

    def inspect_tick(self, tick: int):
        """Handle click on chart points to inspect history."""
        
        # Case 1: Inspecting the current live state (e.g. Tick 0 after reset, or latest tick)
        if tick == self.pop.tick:
            self.inspection_mode = False
            self.inspected_tick = None
            self._update_inspection_view(self.pop.state, tick, is_history=False)
            self.tabs_main.set_value('inspection')
            ui.notify(f"Inspecting Current State (Tick {tick})")
            return

        # Case 2: Inspecting historical state
        self.inspected_tick = tick
        self.inspection_mode = True
        
        # Try to find state in history
        # BasePopulation stores history as [(tick, flattened_array), ...]
        # We need to find the one matching tick
        history_record = next((rec for rec in self.pop._history if rec[0] == tick), None)
        
        if history_record:
            flat_state = history_record[1]
            # Parse state
            n_sexes = self.pop._config.n_sexes
            n_ages = self.pop._config.n_ages
            n_genotypes = self.pop._config.n_genotypes
            
            # Auto-detect state type based on flattened size
            expected_ind_size = int(n_sexes * n_ages * n_genotypes)
            if flat_state.size == 1 + expected_ind_size:
                state_obj = parse_flattened_discrete_state(flat_state, n_sexes, n_ages, n_genotypes, copy=False)
            else:
                state_obj = parse_flattened_state(flat_state, n_sexes, n_ages, n_genotypes, copy=False)
            
            self._update_inspection_view(state_obj, tick, is_history=True)
            self.tabs_main.set_value('inspection')
            ui.notify(f"Inspecting Tick {tick}")
        else:
            ui.notify(f"No history found for Tick {tick}", type="warning")

    def build_layout(self):
        """Construct the NiceGUI layout."""
        
        # --- Header ---
        with ui.header().classes('items-center justify-between bg-slate-900 text-white'):
            ui.label('🧬 NATAL Dashboard').classes('text-2xl font-bold')
            ui.label(f'Population: {self.pop.name}').classes('text-base opacity-80')

        # --- Left Drawer (Controls) ---
        with ui.left_drawer(value=True).classes('bg-gray-50 p-4 shadow-lg border-r').props('width=300'):
            ui.label('Control Panel').classes('text-xl font-bold text-gray-700 mb-4')
            
            # Status
            with ui.row().classes('items-center gap-2 mb-4 p-2 bg-white rounded border'):
                self.status_spinner = ui.spinner(size='sm').props('color=primary')
                self.status_label = ui.label('Ready').classes('text-base font-medium text-gray-600')
                self.status_spinner.visible = False
            
            # Live Stats
            ui.label('Current State').classes('text-sm font-bold text-gray-400 uppercase mb-2')
            with ui.grid(columns=2).classes('w-full gap-y-2 gap-x-4 mb-6'):
                ui.label('Tick:').classes('font-bold text-gray-600 text-lg')
                self.lbl_tick = ui.label(str(self.pop.tick)).classes('text-right font-mono text-lg')
                
                ui.label('Total:').classes('font-bold text-gray-600 text-lg')
                self.lbl_total = ui.label(str(self.pop.get_total_count())).classes('text-right font-mono text-lg')
                
                ui.label('Females:').classes('font-bold text-pink-600 text-lg')
                self.lbl_females = ui.label(str(self.pop.get_female_count())).classes('text-right font-mono text-pink-600 text-lg')
                
                ui.label('Males:').classes('font-bold text-blue-600 text-lg')
                self.lbl_males = ui.label(str(self.pop.get_male_count())).classes('text-right font-mono text-blue-600 text-lg')

            # Speed Control
            ui.label('Interval (s) (0=Unlimited)').classes('text-sm font-bold text-gray-400 uppercase mt-4 mb-2')
            self.slider_speed = ui.slider(min=0.0, max=0.2, value=0.05, step=0.005).props('label-always')
            self.slider_speed.on_value_change(self._update_timer_interval)
            
            # History Control
            ui.label('History Settings').classes('text-sm font-bold text-gray-400 uppercase mt-4 mb-2')
            with ui.grid(columns=2).classes('w-full gap-2'):
                ui.number(label='Record Every', value=self.pop.record_every, min=1, precision=0, on_change=self._update_record_every).classes('w-full')
                ui.number(label='Max History', value=self.pop.max_history, min=10, precision=0, on_change=self._update_max_history).classes('w-full')
            self.lbl_history_count = ui.label(f'(Current: {len(self.pop.history)} snapshots)').classes('text-sm text-gray-400 italic')

            ui.separator().classes('mb-4')
            
            # Control Buttons
            with ui.column().classes('w-full gap-2'):
                with ui.row().classes('w-full gap-2'):
                    ui.button('Step', on_click=self._run_step).props('icon=skip_next outline').classes('flex-grow')
                    
                    def update_play_state(e):
                        self._toggle_play()
                        icon = "pause" if self.is_running else "play_arrow"
                        text = "Pause" if self.is_running else "Play"
                        e.sender.props(f'icon={icon}')
                        e.sender.text = text

                    self.btn_play = ui.button('Play', on_click=update_play_state).props('icon=play_arrow').classes('flex-grow')

                with ui.row().classes('w-full gap-2 mt-2'):
                    ui.button('Reset', on_click=self.reset_simulation).props('icon=restart_alt flat color=grey').classes('flex-grow')
                    ui.button('Export', on_click=self.export_data).props('icon=download flat color=grey').classes('flex-grow')
                
                def reset_zoom_and_update():
                    self.reset_zoom()
                    ui.notify("Zoom reset.")
                ui.button('Reset Zoom', on_click=reset_zoom_and_update).props('icon=zoom_out_map flat color=grey').classes('w-full mt-1')

        # --- Main Content ---
        with ui.column().classes('w-full p-4 gap-6'):
            
            # --- Charts Row ---
            with ui.card().classes('w-full p-0 gap-0 border-none shadow-sm'):
                with ui.row().classes('w-full no-wrap'):
                    # Population Chart
                    self.chart_pop = ui.highchart({
                        'title': {'text': 'Population Size'},
                        'chart': {'type': 'line', 'animation': False, 'height': 300, 'zoomType': 'x'},
                        'xAxis': {'title': {'text': 'Tick'}},
                        'yAxis': {'title': {'text': 'Count'}},
                        'series': [{'name': 'TotalPop', 'data': []}],
                        'plotOptions': {
                            'series': {
                                'dataGrouping': {'enabled': False},
                                'marker': {'enabled': False},
                                'cursor': 'pointer',
                                'events': {'click': True} # Enable click events
                            }
                        }
                    }, on_point_click=self.handle_chart_click).classes('w-1/2 h-80') \
                    .on('selection', self.handle_chart_zoom, ['xAxis'])
                    
                    # Allele Freq Chart
                    self.chart_allele = ui.highchart({
                        'title': {'text': 'Allele Frequencies'},
                        'chart': {'type': 'line', 'animation': False, 'height': 300, 'zoomType': 'x'},
                        'xAxis': {'title': {'text': 'Tick'}},
                        'yAxis': {'title': {'text': 'Freq'}, 'max': 1.0, 'min': 0.0},
                        'series': [],
                        'plotOptions': {
                            'series': {
                                'dataGrouping': {'enabled': False},
                                'marker': {'enabled': False},
                                'cursor': 'pointer',
                                'events': {'click': True}
                            }
                        }
                    }, on_point_click=self.handle_chart_click).classes('w-1/2 h-80') \
                    .on('selection', self.handle_chart_zoom, ['xAxis'])
            
            # --- Tabs Section ---
            with ui.tabs().classes('w-full justify-start border-b') as tabs:
                self.tabs_main = tabs
                tab_inspect = ui.tab(name='inspection', label='Inspection', icon='search')
                tab_config = ui.tab('Configuration', icon='settings')
                tab_hooks = ui.tab('Hooks', icon='extension')
                tab_rules = ui.tab('Genetics', icon='biotech')
            
            with ui.tab_panels(tabs, value='inspection').classes('w-full bg-transparent p-0'):
                
                # --- Tab 1: State Inspection ---
                with ui.tab_panel(tab_inspect).classes('w-full'):
                    with ui.row().classes('items-center justify-between mb-4'):
                        with ui.row().classes('items-center gap-4'):
                            ui.label('Population State Inspection').classes('text-xl font-bold text-gray-700')
                            ui.number(label='Go to Tick', value=None, min=0, precision=0, on_change=self.handle_tick_input).props('dense outlined').classes('w-32')
                        self.lbl_status_mode = ui.label('LIVE VIEW').classes('font-bold text-green-600 text-lg')

                    # Container for Genotype Cards
                    self.genotype_container = ui.row().classes('w-full flex-wrap gap-4')
                    
                # --- Tab 2: Configuration ---
                with ui.tab_panel(tab_config):
                    with ui.row().classes('w-full gap-8'):
                        # Scalar Configs
                        with ui.column().classes('w-1/3'):
                            ui.label('Parameters').classes('text-xl font-bold text-gray-700')
                            conf = self.pop._config
                            ui.label(f"Carrying Capacity: {conf.carrying_capacity}").classes('text-base')
                            ui.label(f"Eggs/Female: {conf.expected_eggs_per_female}").classes('text-base')
                            ui.label(f"Growth Mode: {conf.juvenile_growth_mode}").classes('text-base')
                            ui.label(f"Stochastic: {conf.is_stochastic}").classes('text-base')
                        
                        # Fitness Tables
                        with ui.column().classes('flex-grow'):
                            with ui.expansion('Fitness Tables', icon='fitness_center').classes('w-full border rounded mb-2'):
                                with ui.column().classes('p-2 w-full'):
                                    ui.label('Viability Fitness').classes('font-bold text-gray-700 text-lg')
                                    ui.table(columns=[
                                        {'name': 'Genotype', 'label': 'Genotype', 'field': 'Genotype'},
                                        {'name': 'Female', 'label': 'Female', 'field': 'Female'},
                                        {'name': 'Male', 'label': 'Male', 'field': 'Male'},
                                    ], rows=self._get_viability_data()).props('dense flat').classes('mb-4 w-full')
                                    
                                    ui.label('Fecundity Fitness').classes('font-bold text-gray-700 text-lg')
                                    ui.table(columns=[
                                        {'name': 'Genotype', 'label': 'Genotype', 'field': 'Genotype'},
                                        {'name': 'Female', 'label': 'Female', 'field': 'Female'},
                                        {'name': 'Male', 'label': 'Male', 'field': 'Male'},
                                    ], rows=self._get_fecundity_data()).props('dense flat').classes('w-full')

                # --- Tab 3: Hooks ---
                def format_op(op) -> str:
                    """Format a declarative HookOp into a human-readable string."""
                    from natal.hook.declarative import OpType
                    
                    type_name = str(op.op_type).split('.')[-1]
                    
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

                with ui.tab_panel(tab_hooks):
                    hooks = self.pop.get_compiled_hooks()
                    if not hooks:
                        ui.label("No hooks registered.").classes('text-gray-500 italic')
                    else:
                        for desc in hooks:
                            with ui.expansion(f"{desc.name} ({desc.event})", icon='code').classes('w-full border rounded mb-2'):
                                with ui.column().classes('p-2'):
                                    ui.label(f"Priority: {desc.priority}").classes('text-xs text-gray-500')
                                    
                                    # --- Declarative Hook Plan Visualization ---
                                    if desc.plan:
                                        # Declarative hook
                                        ui.label("Declarative Operations:").classes('font-bold text-base')
                                        
                                        # We need the original Ops list if available, but CompiledHookDescriptor 
                                        # only stores the compiled plan arrays by default.
                                        # However, if it was created via register_declarative_hook or @hook, 
                                        # the source function usually returns the ops list.
                                        # If we have the source wrapper, we can't easily execute it to get ops again safely.
                                        # BUT, `compile_declarative_hook` consumes ops.
                                        # 
                                        # Best effort: If it's a wrapper, try to show source. 
                                        # If it's pure data plan (no wrapper), we can't reverse engineer easily without stored metadata.
                                        # Let's check if we have the original Ops attached.
                                        # (Currently we don't attach them to the descriptor, but we could or just show source)
                                        
                                        # If it has a python wrapper (e.g. from @hook that returned ops), show its source
                                        if desc.py_wrapper:
                                            try:
                                                code = inspect.getsource(desc.py_wrapper)
                                                # Try to extract the return list if simple
                                                ui.code(code, language='python').classes('w-full text-sm')
                                            except OSError:
                                                ui.label("(Source code unavailable)").classes('italic')
                                        elif desc.njit_fn:
                                             # Njit hook
                                            try:
                                                code = inspect.getsource(desc.njit_fn)
                                                ui.code(code, language='python').classes('w-full text-sm')
                                            except OSError:
                                                ui.label("(Njit Source unavailable)").classes('italic')
                                        else:
                                            # Just plan data available.
                                            ui.label("Compiled Plan (Low-level arrays)").classes('text-sm text-gray-400')
                                    
                                    # --- Custom Function Hook ---
                                    else:
                                        target_fn = desc.njit_fn or desc.py_wrapper
                                        if target_fn:
                                            # Try to resolve original function if it's a wrapper
                                            if hasattr(target_fn, '__wrapped__'):
                                                target_fn = target_fn.__wrapped__
                                                
                                            ui.label(f"Function: {target_fn.__name__}").classes('font-mono text-sm mb-1')
                                            if target_fn.__doc__:
                                                ui.markdown(f"_{target_fn.__doc__.strip()}_").classes('text-sm text-gray-600 mb-2 block')
                                                
                                            try:
                                                code = inspect.getsource(target_fn)
                                                ui.code(code, language='python').classes('w-full text-sm')
                                            except OSError:
                                                ui.label("(Source code not available)").classes('italic')

                # --- Tab 4: Genetics ---
                with ui.tab_panel(tab_rules):
                    with ui.column().classes('w-full gap-6'):
                        ui.label('Meiosis (Genotype -> Gametes)').classes('font-bold text-gray-700 text-xl')
                        figs = self._create_meiosis_plots()
                        with ui.row().classes('w-full gap-4'):
                            for fig in figs:
                                ui.plotly(fig).classes('flex-1 h-[600px] border rounded')
                        
                        ui.label('Fertilization (Gametes -> Zygote)').classes('font-bold text-gray-700 text-xl mt-4')
                        fig_fert = self._create_fertilization_plot()
                        if fig_fert:
                            ui.plotly(fig_fert).classes('w-full border rounded').props('style="height: 600px;"')
                        else:
                            ui.label("Fertilization matrix too large to display.").classes('text-orange-500 italic')

        # Timer for loop
        self._tick_timer = ui.timer(0.1, self._on_timer)

        # Initial refresh to show state 0
        self.refresh_ui()

    def reset_zoom(self):
        """Reset zoom on both charts."""
        if self.view_min is None and self.view_max is None:
            return # No zoom to reset
        self.view_min = None
        self.view_max = None
        self._update_charts()
        self.chart_pop.run_method('zoomOut')
        self.chart_allele.run_method('zoomOut')


def launch(population: 'BasePopulation', port: int = 8080, title: str = "NATAL"):
    """
    Launch the embedded dashboard.
    
    Args:
        population: The population object to visualize.
        port: Web server port.
    """
    # Reset NiceGUI state to avoid conflicts if re-run in same process
    # Note: NiceGUI is singleton-based, so multiple launches might need care.
    
    @ui.page('/')
    def main_page():
        dashboard = Dashboard(population)
        dashboard.build_layout()
        
    # Start server
    # In a script usage, we typically want this to block so the script keeps running
    # and serving the UI.
    print(f"🚀 Starting Dashboard at http://localhost:{port}")
    ui.run(title=title, port=port, show=False, reload=False)