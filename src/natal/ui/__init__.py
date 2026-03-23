"""
NATAL UI Module
===============

Provides user interface components for visualizing and controlling simulations.
Requires `nicegui` to be installed (`pip install nicegui`).
"""

from .dashboard import Dashboard, launch

__all__ = ["Dashboard", "launch"]
