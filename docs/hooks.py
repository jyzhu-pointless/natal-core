"""mkdocs hook — monkey-patch pygments for Python 3.13 compatibility.

The HtmlFormatter.__init__ receives ``filename=None`` from pymdownx
when a fenced code block has no title.  Python 3.13's html.escape
does not tolerate None, causing ``mkdocs build`` to crash.
"""

from __future__ import annotations


def _apply_pygments_fix() -> None:
    import pygments.formatters.html

    _original_init = pygments.formatters.html.HtmlFormatter.__init__

    def _patched_init(self: object, **options: object) -> None:
        if "filename" in options and options["filename"] is None:
            options["filename"] = ""
        _original_init(self, **options)  # type: ignore[arg-type]

    pygments.formatters.html.HtmlFormatter.__init__ = _patched_init  # type: ignore[method-assign]


def on_startup(*, command: str, **kwargs: object) -> None:  # noqa: ARG001
    """Apply the pygments fix before mkdocs starts building."""
    _apply_pygments_fix()


def on_pre_build(*, config: object, **kwargs: object) -> None:  # noqa: ARG001
    """Ensure the fix is applied before every build."""
    _apply_pygments_fix()
