#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from natal.hook_dsl import CompiledEventHooks, HookProgram  # noqa: E402


def _empty_program() -> HookProgram:
    return HookProgram(
        n_events=np.int32(4),
        n_hooks=np.int32(0),
        hook_offsets=np.array([0, 0, 0, 0, 0], dtype=np.int32),
        n_ops_list=np.array([], dtype=np.int32),
        op_offsets=np.array([0], dtype=np.int32),
        op_types_data=np.array([], dtype=np.int32),
        gidx_offsets_data=np.array([0], dtype=np.int32),
        gidx_data=np.array([], dtype=np.int32),
        age_offsets_data=np.array([0], dtype=np.int32),
        age_data=np.array([], dtype=np.int32),
        sex_masks_data=np.array([], dtype=np.bool_),
        params_data=np.array([], dtype=np.float64),
        condition_offsets_data=np.array([0], dtype=np.int32),
        condition_types_data=np.array([], dtype=np.int32),
        condition_params_data=np.array([], dtype=np.int32),
        deme_selector_types=np.array([], dtype=np.int32),
        deme_selector_offsets=np.array([0], dtype=np.int32),
        deme_selector_data=np.array([], dtype=np.int32),
    )


def test_compiled_event_hooks_exposes_spatial_wrappers():
    hooks = CompiledEventHooks.from_compiled_hooks([], registry=_empty_program())
    assert hooks.run_spatial_tick_fn is not None
    assert hooks.run_spatial_fn is not None
