#!/usr/bin/env python3

from pathlib import Path


def test_spatial_template_stage_order_is_strict():
    template_path = Path(__file__).resolve().parents[1] / "src" / "natal" / "kernels" / "templates" / "spatial_kernel_wrappers.py.tmpl"
    source = template_path.read_text(encoding="utf-8")

    tick_def = source.index("def __RUN_SPATIAL_TICK_NAME__")
    tick_body = source[tick_def:]

    idx_first = tick_body.index("_first_event")
    idx_repro = tick_body.index("run_spatial_reproduction")
    idx_early = tick_body.index("_early_event")
    idx_surv = tick_body.index("run_spatial_survival")
    idx_late = tick_body.index("_late_event")
    idx_aging = tick_body.index("run_spatial_aging")
    idx_mig = tick_body.index("run_spatial_migration")

    assert idx_first < idx_repro < idx_early < idx_surv < idx_late < idx_aging < idx_mig
