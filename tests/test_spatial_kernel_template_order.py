#!/usr/bin/env python3

from pathlib import Path


def test_spatial_template_stage_order_is_strict():
    template_path = Path(__file__).resolve().parents[1] / "src" / "natal" / "kernels" / "templates" / "spatial_kernel_wrappers.py.tmpl"
    source = template_path.read_text(encoding="utf-8")

    tick_def = source.index("def __RUN_SPATIAL_TICK_NAME__")
    tick_body = source[tick_def:]

    idx_tick_with_migration = tick_body.index("run_spatial_tick_with_migration")
    idx_adjacency = tick_body.index("adjacency=adjacency")
    idx_migration_rate = tick_body.index("migration_rate=migration_rate")

    assert idx_tick_with_migration < idx_adjacency < idx_migration_rate
