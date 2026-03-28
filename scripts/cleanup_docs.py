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
    # Note: essential_files variable is defined for documentation purposes but not used in current implementation

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
