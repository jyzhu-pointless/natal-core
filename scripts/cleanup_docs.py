#!/usr/bin/env python3
"""
Clean up unused documentation files

This script removes the old API documentation files that are no longer needed
since we're using mkdocstrings for automatic API documentation generation.
"""

import os
import shutil
from pathlib import Path

def main():
    """Main function"""
    
    docs_dir = Path(__file__).parent.parent / 'docs'
    
    # Files and directories to remove
    items_to_remove = [
        # Old API directories
        docs_dir / 'api',
        docs_dir / 'enhanced_api',
        
        # Old Sphinx directories
        docs_dir / 'source',
        docs_dir / 'build',
        
        # Old configuration files
        docs_dir / 'Makefile',
        docs_dir / 'conf.py',
    ]
    
    print("Cleaning up old documentation files...")
    
    for item in items_to_remove:
        if item.exists():
            if item.is_dir():
                print(f"Removing directory: {item}")
                shutil.rmtree(item)
            else:
                print(f"Removing file: {item}")
                os.remove(item)
        else:
            print(f"Not found (skipping): {item}")
    
    # Keep only the essential files
    essential_files = [
        '01_quickstart.md',
        '02_genetic_structures.md', 
        '03_simulation_kernels.md',
        '04_population_state_config.md',
        '05_index_core.md',
        '06_modifiers.md',
        '07_hooks.md',
        '08_numba_optimization.md',
        '09_api_reference.md',
        '10_samplers_observation.md',
        '11_allele_conversion_rules.md',
        '12_genotype_filter_implementation.md',
        '13_filtering_api_reference.md',
        '14_genotype_pattern_matching_design.md',
        '15_genetic_presets_guide.md',
        '16_spatial_simulation_guide.md',
        'README.md'
    ]
    
    # Create new api directory for mkdocstrings files
    new_api_dir = docs_dir / 'api'
    new_api_dir.mkdir(exist_ok=True)
    print(f"Created directory: {new_api_dir}")
    
    print("\nCleanup complete!")
    print("The documentation structure is now simplified:")
    print("- docs/ - User guide Markdown files")
    print("- docs/api/ - Auto-generated API documentation")
    print("- mkdocs.yml - MkDocs configuration")

if __name__ == "__main__":
    main()