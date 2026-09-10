"""FastAPI application factory for the NATAL Vue web UI.

The app exposes two channel families:

- ``/api/*`` REST endpoints for snapshots and queries (request/response).
- ``/ws`` a WebSocket carrying control commands (client -> server) and live
  updates / log frames (server -> client).

Handlers are module-level functions reading session state from
``app.state`` (FastAPI convention): this keeps them referenceable for the
type checker and directly testable without closures.

When ``frontend/dist`` exists (production build), it is served at ``/`` so a
single ``launch_vue(pop)`` call is self-contained.  During frontend
development the dist directory is absent and the Vite dev server proxies
``/api`` and ``/ws`` to this app instead.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .rest import ROUTES
from .session import SimulationSession
from .types import DashboardPopulation
from .ws import session_from_app, websocket_endpoint

#: Location of the built frontend, relative to this file:
#: ``<repo-root>/frontend/dist`` (parents: webui -> frontend -> natal -> src -> root).
_DIST_DIR = Path(__file__).resolve().parents[4] / "frontend" / "dist"


async def _get_meta(request: Request) -> dict[str, object]:  # object: heterogeneous JSON meta payload
    """Return static dashboard metadata plus the live tick."""
    session = session_from_app(request.app)
    return {
        "app": "natal-webui",
        "title": request.app.state.title,
        "dashboard_type": session.dashboard_type,
        "backend": session.backend,
        "tick": session.tick,
        "status": session.status,
        "interval_ms": session.interval_ms,
        "population_name": session.population.name,
    }


async def _index_fallback() -> HTMLResponse:
    """Serve a hint page when the frontend build is not present."""
    return HTMLResponse(
        "<h1>NATAL Vue dashboard</h1><p>Frontend build not found. "
        "Either run <code>corepack pnpm build</code> in "
        "<code>frontend/</code> or start the Vite dev server "
        "(<code>corepack pnpm dev</code>) which proxies to this app.</p>"
    )


def create_app(
    population: DashboardPopulation, *, title: str = "NATAL Dashboard"
) -> FastAPI:
    """Create the FastAPI application bound to *population*.

    Args:
        population: A built panmictic or spatial population.
        title: Window title reported through ``/api/meta``.

    Returns:
        The configured application (not yet running).  The engine loop
        starts with the app's lifespan and stops at shutdown.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        session = session_from_app(app)
        session.start()
        yield
        session.shutdown()

    app = FastAPI(title=title, lifespan=lifespan)
    app.state.session = SimulationSession(population)
    app.state.title = title

    app.add_api_route("/api/meta", _get_meta, methods=["GET"])
    for path, handler, methods in ROUTES:
        app.add_api_route(path, handler, methods=list(methods))
    app.add_api_websocket_route("/ws", websocket_endpoint)

    if _DIST_DIR.is_dir():
        app.mount("/", StaticFiles(directory=_DIST_DIR, html=True), name="static")
    else:
        app.add_api_route(
            "/", _index_fallback, methods=["GET"], response_class=HTMLResponse
        )

    return app
