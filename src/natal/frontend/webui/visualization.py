"""Visualization helpers for the web UI serialization layer.

Pure display utilities: allele display colors, genotype cell SVG rendering,
and unordered genotype label construction.  These are internal to the webui
package — nothing here is re-exported at the ``natal`` top level.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from natal.frontend.genetics import HaploidGenotype, Haplotype, Locus, Species

__all__ = ["get_allele_color", "get_unordered_genotype_labels", "render_cell_svg"]


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


def get_unordered_genotype_labels(genotypes: list[Any]) -> list[str]:
    """Generate unique unordered (``::``) genotype labels from a genotype list.

    For each genotype, builds a label in the form ``hapstrA::hapstrB``
    (sorted alphabetically so ``WT|Dr`` and ``Dr|WT`` both become ``WT::Dr``).
    Multi-chromosome: ``hapA::hapB; hapC::hapC``.

    Returns:
        Sorted unique labels suitable for dropdown options.
    """
    seen: set[str] = set()
    labels: list[str] = []
    for gt in genotypes:
        chrom_pairs: list[str] = []
        for chrom in gt.species.chromosomes:
            mat_hap = gt.maternal.get_haplotype_for_chromosome(chrom)
            pat_hap = gt.paternal.get_haplotype_for_chromosome(chrom)

            def _hap_str(hap: "Haplotype", loci: "list[Locus]") -> str:
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


def render_cell_svg(entity: Any, species_def: "Species", size: int = 100) -> str:
    """Generate an SVG string representing a cell's genotype.

    Draws a cell circle containing chromosome bars. Can render both diploid
    Genotypes and HaploidGenotypes.

    Args:
        entity: Genotype or HaploidGenotype instance (duck-typed).
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

        def draw_chrom_bar(x: float, source_obj: "HaploidGenotype | None") -> None:
            # Get haplotype for this chromosome; ``None`` means the chromosome
            # is missing from this genome, so nothing is drawn.
            if source_obj is None:
                return

            haplo = source_obj.get_haplotype_for_chromosome(chrom)

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
            draw_chrom_bar(cx - 5, entity.maternal)  # Maternal (Left)
            draw_chrom_bar(cx + 5, entity.paternal)  # Paternal (Right)
        else:
            draw_chrom_bar(cx, entity)  # Haploid (Center)

    svg.append('</svg>')
    return "".join(svg)
