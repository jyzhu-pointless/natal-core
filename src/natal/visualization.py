"""
Visualization utilities for genetic entities.

Provides helper functions to render genetic entities (Genotypes, HaploidGenotypes)
as visual representations (SVG, colors, etc.).
"""

import hashlib
from typing import Any

__all__ = ["get_allele_color", "render_cell_svg"]

def get_allele_color(allele_name: str) -> str:
    """Determine a display color for an allele based on naming conventions.
    
    Args:
        allele_name: The name of the allele.
        
    Returns:
        Hex color string (e.g., "#ff0000").
    """
    name = allele_name.lower()
    # Default color scheme
    if "wt" in name or "+" in name or "wild" in name:
        return "#3b82f6"  # Blue (WT)
    if "drive" in name or "dr" in name:
        return "#ef4444"  # Red (Drive)
    if "r1" in name or "functional" in name:
        return "#a855f7"  # Purple (Functional R1)
    if "r2" in name or "resistance" in name:
        return "#eab308"  # Yellow (R2 / Resistance)
    if "rescue" in name:
        return "#22c55e"  # Green (Rescue)

    # Fallback: deterministic random color based on name hash
    h = hashlib.md5(allele_name.encode('utf-8')).hexdigest()
    return f"#{h[:6]}"

def render_cell_svg(entity: Any, species_def: Any, size: int = 100) -> str:
    """Generate an SVG string representing a cell's genotype.
    
    Draws a cell circle containing chromosome bars. Can render both diploid 
    Genotypes and HaploidGenotypes.
    
    Args:
        entity: Genotype or HaploidGenotype instance.
        species_def: Species instance defining the chromosome structure.
        size: Width/Height of the SVG in pixels.
        
    Returns:
        String containing the SVG XML.
    """
    # Determine ploidy based on attributes
    is_diploid = hasattr(entity, 'maternal') and hasattr(entity, 'paternal')
    
    chromosomes = species_def.chromosomes
    n_chroms = len(chromosomes)
    
    # SVG container and cell membrane
    svg = [f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">']
    svg.append('<circle cx="50" cy="50" r="48" fill="#f8fafc" stroke="#334155" stroke-width="2"/>')
    
    # Layout calculations
    padding_x = 20
    avail_width = 100 - 2 * padding_x
    col_width = avail_width / max(1, n_chroms)
    
    bar_width = 6
    bar_height = 50
    bar_y_start = (100 - bar_height) / 2
    
    for i, chrom in enumerate(chromosomes):
        cx = padding_x + i * col_width + col_width / 2
        loci = chrom.loci
        n_loci = len(loci)
        seg_height = bar_height / max(1, n_loci)
        
        def draw_chrom_bar(x: float, source_obj: Any) -> None:
            # Get haplotype for this chromosome
            if source_obj is None: # Missing chromosome
                return
            
            # Try to get haplotype (works for both Genotype via helper or HaploidGenotype direct access)
            # For Genotype, source_obj is a HaploidGenotype
            # For HaploidGenotype, source_obj is self
            haplo = source_obj.get_haplotype_for_chromosome(chrom)
            if haplo is None: return

            for l_idx, locus in enumerate(loci):
                gene = haplo.get_gene_at_locus(locus)
                color = get_allele_color(gene.name) if gene else "#cbd5e1"
                
                y = bar_y_start + l_idx * seg_height
                # Draw segment (rounded if single, or ends)
                # Simplified rounding for visual cleanliness
                radius = 3 if n_loci == 1 else 1
                svg.append(f'<rect x="{x - bar_width/2}" y="{y}" width="{bar_width}" height="{seg_height}" '
                           f'fill="{color}" rx="{radius}" stroke="none"/>')
                # Separator line between loci
                if l_idx > 0:
                    svg.append(f'<line x1="{x-bar_width/2}" y1="{y}" x2="{x+bar_width/2}" y2="{y}" stroke="white" stroke-width="1"/>')

        if is_diploid:
            draw_chrom_bar(cx - 5, entity.maternal) # Maternal (Left)
            draw_chrom_bar(cx + 5, entity.paternal) # Paternal (Right)
        else:
            draw_chrom_bar(cx, entity) # Haploid (Center)

    svg.append('</svg>')
    return "".join(svg)