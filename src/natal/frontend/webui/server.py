"""Entry point for the Vue-based web dashboard.

``launch_vue`` is the NiceGUI ``launch`` replacement during the migration
window; both stay available so existing demos keep working until parity is
accepted.
"""

from __future__ import annotations

import uvicorn

from .app import create_app
from .types import DashboardPopulation


def launch_vue(
    population: DashboardPopulation,
    port: int = 8000,
    title: str = "NATAL Dashboard",
) -> None:
    """Serve the Vue dashboard for *population* and block.

    Production mode serves the built ``frontend/dist`` at the root URL.  In
    frontend development, run the Vite dev server (``npm run dev`` inside
    ``frontend/``) instead; it proxies ``/api`` and ``/ws`` to this server.

    Args:
        population: A built panmictic or spatial population.
        port: TCP port for the HTTP + WebSocket server.
        title: Dashboard window title.
    """
    app = create_app(population, title=title)
    print(f"🚀 NATAL Vue dashboard at http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
