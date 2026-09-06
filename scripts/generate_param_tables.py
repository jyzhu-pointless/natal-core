"""Generate the Rust wire tables for runtime-mutable ecology parameters.

RUST_ONLY_REFACTOR_PLAN.md section 5.4: the parameter inventory and
bounds must not be hand-written in more than one place.
``src/natal/parameters.jsonc`` is the single source; this script derives
the Rust mirror (``rust/src/eco_param_wire.rs``) from it, in the fixed
``ECO_PARAM_NAMES`` wire order shared with the Python hook compiler.

Usage::

    python scripts/generate_param_tables.py           # regenerate
    python scripts/generate_param_tables.py --check   # fail on drift

The pytest suite pins freshness (a test runs ``--check``), so an edit to
the jsonc without a regeneration fails the gates instead of silently
desynchronizing the Rust commit validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from natal.frontend.utils.parameters import ParamDescriptor

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "rust" / "src" / "eco_param_wire.rs"

HEADER = """\
//! GENERATED FILE — DO NOT EDIT.
//!
//! Produced by ``scripts/generate_param_tables.py`` from
//! ``src/natal/parameters.jsonc`` (single source of truth, plan
//! section 5.4).  Wire order follows ``ECO_PARAM_NAMES`` in
//! ``natal/frontend/hooks/types.py``.  Run the generator after editing
//! the jsonc; the pytest suite fails on drift.
"""


def _wire_rows() -> list[tuple[str, float, float]]:
    """Return (name, lo, hi) rows in ECO_PARAM_NAMES order."""
    from natal.frontend.hooks.types import ECO_PARAM_NAMES
    from natal.frontend.utils.parameters import ALL_PARAMETERS

    by_name = {d.name: d for d in ALL_PARAMETERS.values()}
    rows: list[tuple[str, float, float]] = []
    for name in ECO_PARAM_NAMES:
        desc = by_name[name]
        bounds = desc.bounds
        if bounds is None:
            raise SystemExit(f"parameter {name!r} has no bounds in the jsonc")
        lo, hi = float(bounds[0]), float(bounds[1])
        rows.append((name, lo, hi))
    return rows


def _require_jsonc_names(
    by_name: dict[str, ParamDescriptor], names: tuple[str, ...]
) -> None:
    """Fail closed when a layout-appended name disappears from the jsonc."""
    for name in names:
        if name not in by_name:
            raise SystemExit(
                f"parameter {name!r} is part of the Rust layout contract but "
                "is missing from src/natal/parameters.jsonc"
            )


def _layout_names(
    by_name: dict[str, ParamDescriptor],
) -> tuple[list[str], list[str]]:
    """Return the params-contract and checkpoint scalar column lists.

    Both lists extend the fixed wire order with layout-appended scalar
    names that exist in the jsonc: the params contract carries
    ``external_expected_eggs`` last; the memory checkpoint additionally
    carries ``growth_mode`` before it.
    """
    from natal.frontend.hooks.types import ECO_PARAM_NAMES

    appended = ("external_expected_eggs", "growth_mode")
    _require_jsonc_names(by_name, appended)
    contract_columns = list(ECO_PARAM_NAMES) + ["external_expected_eggs"]
    checkpoint_columns = (
        list(ECO_PARAM_NAMES) + ["growth_mode", "external_expected_eggs"]
    )
    return contract_columns, checkpoint_columns


def _render(
    rows: list[tuple[str, float, float]],
    contract_columns: list[str],
    checkpoint_columns: list[str],
) -> str:
    """Render the Rust module text for the wire rows and layout lists."""
    n = len(rows)
    # Trailing commas keep rustfmt happy (the generated file must pass
    # `cargo fmt --check` untouched).
    names = "\n    ".join(f'"{name}",' for name, _, _ in rows)
    bounds = "\n    ".join(f"({lo!r}, {hi!r})," for _, lo, hi in rows)
    contract_list = "\n    ".join(f'"{c}",' for c in contract_columns)
    checkpoint_list = "\n    ".join(f'"{c}",' for c in checkpoint_columns)
    n_contract = len(contract_columns)
    n_checkpoint = len(checkpoint_columns)
    return (
        f"{HEADER}\n"
        f"/// Wire names of the runtime-mutable ecology parameters, in the\n"
        f"/// fixed order shared with the Python hook compiler.\n"
        f"pub const ECO_PARAM_COLUMNS: [&str; {n}] = [\n    {names}\n];\n\n"
        f"/// Number of wire columns.\n"
        f"pub const N_ECO_PARAMS: usize = ECO_PARAM_COLUMNS.len();\n\n"
        f"/// Validity bounds per column — the generated mirror of the jsonc\n"
        f"/// scalar ``bounds`` (same order as ``ECO_PARAM_COLUMNS``).\n"
        f"pub const ECO_PARAM_BOUNDS: [(f64, f64); N_ECO_PARAMS] = [\n    {bounds}\n];\n\n"
        f"/// Ecology scalar channel names of the params contract: the wire\n"
        f"/// order plus ``external_expected_eggs`` appended last.\n"
        f"pub const ECOLOGY_SCALAR_COLUMNS: [&str; {n_contract}] = [\n    {contract_list}\n];\n\n"
        f"/// Ecology scalar names carried by a memory checkpoint: the wire\n"
        f"/// order plus ``growth_mode`` and ``external_expected_eggs``.\n"
        f"pub const ECOLOGY_SCALARS: [&str; {n_checkpoint}] = [\n    {checkpoint_list}\n];\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Regenerate the Rust wire tables or verify their freshness.

    Args:
        argv: Optional CLI arguments (``--check``).

    Returns:
        Process exit code: 0 on success or (with ``--check``) on a fresh
        file; 1 on drift or missing bounds.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the file would change",
    )
    args = parser.parse_args(argv)

    from natal.frontend.utils.parameters import ALL_PARAMETERS

    by_name = {d.name: d for d in ALL_PARAMETERS.values()}
    rows = _wire_rows()
    contract_columns, checkpoint_columns = _layout_names(by_name)
    rendered = _render(rows, contract_columns, checkpoint_columns)
    if args.check:
        current = (
            TARGET.read_text(encoding="utf-8") if TARGET.is_file() else ""
        )
        if current != rendered:
            print(
                "DRIFT: rust/src/eco_param_wire.rs is stale against "
                "src/natal/parameters.jsonc — run "
                "python scripts/generate_param_tables.py",
                file=sys.stderr,
            )
            return 1
        print("rust/src/eco_param_wire.rs is fresh")
        return 0

    TARGET.write_text(rendered, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
