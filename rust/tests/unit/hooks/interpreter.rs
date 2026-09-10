//! Unit tests for the OP_SET_PARAM RPN interpreter and the
//! OP_CONVERT migration, mirroring the Python-side invariants.

use super::*;

/// Build a minimal one-hook program with the given op payload.
fn single_op_program(op_type: i64, param: f64) -> HookProgram {
    // One hook on event 1, one op with all-deme selectors, a dummy
    // ztype/age span, and an always-true condition.
    HookProgram {
        n_events: 4,
        n_hooks: 1,
        hook_offsets: vec![0, 0, 1, 1, 1],
        op_offsets: vec![0, 1],
        op_types: vec![op_type],
        params: vec![param],
        zidx_offsets: vec![0, 1],
        zidx_data: vec![0],
        age_offsets: vec![0, 1],
        age_data: vec![0],
        sex_masks: vec![false, false],
        condition_offsets: vec![0, 1],
        condition_types: vec![COND_ALWAYS],
        condition_params: vec![0],
        deme_selector_types: vec![0],
        deme_selector_offsets: vec![0, 0],
        ..HookProgram::default()
    }
}

fn seeded() -> SessionRng {
    crate::kernels::rng::new_rng(42)
}

/// The generated wire tables are internally consistent: one bounds
/// row per column, and every row is a real interval.  Value-level
/// freshness against ``parameters.jsonc`` is enforced by the Python
/// generator's ``--check`` test — no second hand-written copy here.
#[test]
fn eco_param_wire_tables_are_consistent() {
    assert_eq!(
        ECO_PARAM_BOUNDS.len(),
        crate::generated::ecology_parameters::ECO_PARAM_COLUMNS.len()
    );
    // Compile-time completeness: the generated wire table must never
    // be empty (a const assert avoids clippy's const_is_empty lint on
    // the runtime form).
    const _: () = assert!(!crate::generated::ecology_parameters::ECO_PARAM_COLUMNS.is_empty());
    for (id, (lo, hi)) in ECO_PARAM_BOUNDS.iter().enumerate() {
        assert!(lo <= hi, "bounds row {id} is not an interval");
    }
}

/// Validation accepts boundary values and rejects non-finite /
/// out-of-bounds ones with a message naming the parameter.
#[test]
fn validate_eco_param_accepts_bounds_and_rejects_inf_nan() {
    assert!(validate_eco_param(0, 0.0).is_ok());
    assert!(validate_eco_param(0, 1e12).is_ok());
    assert!(validate_eco_param(2, 1.0).is_ok());
    let inf_err = validate_eco_param(0, f64::INFINITY).unwrap_err();
    assert!(
        inf_err.contains("carrying_capacity") && inf_err.contains("inf"),
        "message names the parameter and the offending value: {inf_err}"
    );
    assert!(validate_eco_param(1, f64::NAN).is_err());
    assert!(validate_eco_param(2, 1.5).is_err());
    assert!(validate_eco_param(3, -0.1).is_err());
    assert!(validate_eco_param(4, 2e6).is_err());
}

#[test]
fn rpn_evaluates_expressions_against_current_values() {
    // "K * 0.95" with K = 200 -> 190.  Tokens: [param0, lit0, mul].
    let mut program = single_op_program(OP_SET_PARAM, 0.0);
    program.sp_param_ids = vec![0];
    program.sp_every = vec![1];
    program.sp_start = vec![0];
    program.rpn_offsets = vec![0, 3];
    program.rpn_kinds = vec![RPN_PARAM, RPN_LITERAL, RPN_MUL];
    program.rpn_payload = vec![0, 0, 0];
    program.sp_literals = vec![0.95];

    let mut eco = [200.0, 0.0, 0.0, 0.0, 0.0];
    let mut rng = seeded();
    let mut ind = vec![0.0; 2];
    let result = program
        .execute_event(
            &mut rng,
            1,
            &mut ind,
            &mut [],
            2,
            1,
            1,
            5,
            false,
            false,
            0,
            &mut eco,
            &mut None,
        )
        .unwrap();
    assert_eq!(result, RESULT_CONTINUE);
    assert!((eco[0] - 190.0).abs() < 1e-12, "K * 0.95 with K=200 -> 190");
}

#[test]
fn rpn_division_by_zero_follows_ieee_semantics() {
    // "1 / (K - K)" with K = 50 -> 1 / 0 -> +inf.
    let mut program = single_op_program(OP_SET_PARAM, 0.0);
    program.sp_param_ids = vec![0];
    program.sp_every = vec![1];
    program.sp_start = vec![0];
    program.rpn_offsets = vec![0, 5];
    program.rpn_kinds = vec![RPN_LITERAL, RPN_PARAM, RPN_PARAM, RPN_SUB, RPN_DIV];
    program.rpn_payload = vec![0, 0, 0, 0, 0];
    program.sp_literals = vec![1.0];

    let mut eco = [50.0, 0.0, 0.0, 0.0, 0.0];
    let mut rng = seeded();
    let mut ind = vec![0.0; 2];
    program
        .execute_event(
            &mut rng,
            1,
            &mut ind,
            &mut [],
            2,
            1,
            1,
            0,
            false,
            false,
            0,
            &mut eco,
            &mut None,
        )
        .unwrap();
    assert!(
        eco[0].is_infinite() && eco[0] > 0.0,
        "1/0 -> +inf, got {}",
        eco[0]
    );
}

#[test]
fn set_param_respects_every_and_start_schedule() {
    // every=10, start=5: fires at ticks 5, 15, 25 — not 0..4 or 10.
    let mut program = single_op_program(OP_SET_PARAM, 0.0);
    program.sp_param_ids = vec![0];
    program.sp_every = vec![10];
    program.sp_start = vec![5];
    program.rpn_offsets = vec![0, 1];
    program.rpn_kinds = vec![RPN_LITERAL];
    program.rpn_payload = vec![0];
    program.sp_literals = vec![-1.0];

    for tick in [0_i64, 4, 5, 10, 14, 15, 24, 25] {
        let mut eco = [7.0, 0.0, 0.0, 0.0, 0.0];
        let mut rng = seeded();
        let mut ind = vec![0.0; 2];
        program
            .execute_event(
                &mut rng,
                1,
                &mut ind,
                &mut [],
                2,
                1,
                1,
                tick,
                false,
                false,
                0,
                &mut eco,
                &mut None,
            )
            .unwrap();
        let expected = if tick >= 5 && (tick - 5) % 10 == 0 {
            -1.0
        } else {
            7.0
        };
        assert_eq!(eco[0], expected, "tick {tick}");
    }
}

#[test]
fn deterministic_convert_conserves_totals_with_sperm() {
    // 2 ages, 3 ztypes, sperm (age, female_z, male_z).
    let mut program = single_op_program(OP_CONVERT, 0.25);
    program.convert_source_z = vec![0];
    program.convert_target_z = vec![1];

    let n_ages = 2;
    let n_ztypes = 3;
    let mut ind = vec![0.0; 2 * n_ages * n_ztypes];
    let mut sperm = vec![0.0; n_ages * n_ztypes * n_ztypes];
    // Female A|A: 20 per age (6 mated via sperm buckets, 14 virgin).
    // Male A|A: 15 per age.
    for age in 0..n_ages {
        ind[age * n_ztypes] = 20.0;
        ind[(n_ages + age) * n_ztypes] = 15.0;
        sperm[age * n_ztypes * n_ztypes + 1] = 3.0;
        sperm[age * n_ztypes * n_ztypes + 2] = 3.0;
    }
    let ind_before = ind.clone();
    let sperm_before = sperm.clone();
    let total_before: f64 = ind.iter().sum::<f64>() + sperm.iter().sum::<f64>();

    let mut rng = seeded();
    let mut eco = [0.0; N_ECO_PARAMS];
    let result = program
        .execute_event(
            &mut rng, 1, &mut ind, &mut sperm, 2, n_ages, n_ztypes, 0, false, false, 0, &mut eco,
            &mut None,
        )
        .unwrap();
    assert_eq!(result, RESULT_CONTINUE);

    let total_after: f64 = ind.iter().sum::<f64>() + sperm.iter().sum::<f64>();
    assert!(
        (total_before - total_after).abs() < 1e-12,
        "total conserved"
    );

    // Males: 15 -> 11.25 per age (25 % deterministic migration).
    for age in 0..n_ages {
        assert!(
            (ind[(n_ages + age) * n_ztypes] - 11.25).abs() < 1e-12,
            "male deterministic migration"
        );
        assert!(
            (ind[(n_ages + age) * n_ztypes + 1] - 3.75).abs() < 1e-12,
            "male target gains exactly the moved count"
        );
        // Females: virgins 14 and each 3-count sperm bucket migrate at
        // 25 % -> 0.75 + 0.75 + 3.5 = 5.0 moved, 15.0 remain.
        assert!(
            (ind[age * n_ztypes] - 15.0).abs() < 1e-12,
            "female source after migration"
        );
        assert!(
            (ind[age * n_ztypes + 1] - 5.0).abs() < 1e-12,
            "female target after migration"
        );
    }
    // Per-bucket atomicity: source bucket + target bucket unchanged sum.
    for age in 0..n_ages {
        for mz in 0..n_ztypes {
            let src = age * n_ztypes * n_ztypes + mz;
            let dst = (age * n_ztypes + 1) * n_ztypes + mz;
            assert!(
                (sperm[src] + sperm[dst] - (sperm_before[src] + sperm_before[dst])).abs() < 1e-12,
                "bucket ({age}, 0, {mz}) conserved"
            );
        }
    }
    // The male axis of every sperm row is untouched.
    for mz in 0..n_ztypes {
        let col_before: f64 = (0..n_ages)
            .map(|age| sperm_before[age * n_ztypes * n_ztypes + mz])
            .sum();
        let col_after: f64 = (0..n_ages)
            .map(|age| {
                sperm[age * n_ztypes * n_ztypes + mz]
                    + sperm[(age * n_ztypes + 1) * n_ztypes + mz]
                    + sperm[(age * n_ztypes + 2) * n_ztypes + mz]
            })
            .sum();
        assert!(
            (col_before - col_after).abs() < 1e-12,
            "male sperm axis {mz} frozen"
        );
    }
    let _ = ind_before;
}

#[test]
fn stochastic_convert_conservs_expectation() {
    let mut program = single_op_program(OP_CONVERT, 0.5);
    program.convert_source_z = vec![0];
    program.convert_target_z = vec![2];

    let n_ztypes = 3;
    let trials = 400;
    let mut moved_sum = 0.0;
    // Distinct seed per trial: re-using one seed would repeat the
    // same draw and make the mean a single sample.
    for trial in 0..trials {
        let mut ind = vec![0.0; 2 * n_ztypes];
        ind[0] = 100.0; // female A|A, age 0, no sperm
        let mut rng = crate::kernels::rng::new_rng(1_000 + trial as u64);
        let mut eco = [0.0; N_ECO_PARAMS];
        program
            .execute_event(
                &mut rng,
                1,
                &mut ind,
                &mut [],
                2,
                1,
                n_ztypes,
                0,
                true,
                false,
                0,
                &mut eco,
                &mut None,
            )
            .unwrap();
        moved_sum += ind[2];
    }
    let mean = moved_sum / trials as f64;
    // 100 females * 0.5 = 50 expected; 3-sigma of Binomial(100, .5)
    // averaged over 400 trials is ~0.06 * 5 = loose bound 2.5.
    assert!(
        (mean - 50.0).abs() < 2.5,
        "expected 50 moved on average, got {mean}"
    );
}

/// Fixture: event 0 carries one CSR hook, event 1 carries two; each hook
/// owns exactly one op (op offsets 0|1|3), all selectors wildcard.
fn two_event_csr_program() -> HookProgram {
    HookProgram {
        n_events: 4,
        n_hooks: 3,
        hook_offsets: vec![0, 1, 3, 3, 3],
        op_offsets: vec![0, 1, 2, 3],
        op_types: vec![OP_ADD, OP_ADD, OP_ADD],
        params: vec![1.0, 1.0, 1.0],
        zidx_offsets: vec![0, 1, 2, 3],
        zidx_data: vec![0, 0, 0],
        age_offsets: vec![0, 1, 2, 3],
        age_data: vec![0, 0, 0],
        sex_masks: vec![false; 6],
        condition_offsets: vec![0, 1, 2, 3],
        condition_types: vec![COND_ALWAYS; 3],
        condition_params: vec![0; 3],
        deme_selector_types: vec![0; 3],
        deme_selector_offsets: vec![0, 0, 0, 0],
        python_callback_slots: vec![-1, -1, -1],
        ..HookProgram::default()
    }
}

fn none_callbacks(counts: [usize; 4]) -> Vec<Vec<pyo3::Py<pyo3::PyAny>>> {
    pyo3::prepare_freethreaded_python();
    pyo3::Python::with_gil(|py| {
        counts
            .iter()
            .map(|n| (0..*n).map(|_| py.None()).collect())
            .collect()
    })
}

/// Installing callback lists on a CSR-only program appends paired zero-op
/// wildcard slots at each event segment end and keeps every CSR column
/// consistent (offsets shifted, n_hooks grown, slot column interleaved).
#[test]
fn install_callback_lists_appends_paired_slots() {
    let mut program = two_event_csr_program();
    program.install_callback_lists(none_callbacks([1, 0, 2, 0]));

    assert_eq!(program.n_hooks, 6);
    assert_eq!(program.hook_offsets, vec![0, 2, 4, 6, 6]);
    // Slot column: event 0 keeps its CSR slot then the callback; event 1
    // keeps both CSR slots; event 2 gains two callback slots.
    assert_eq!(program.python_callback_slots, vec![-1, 0, -1, -1, 0, 1]);
    // New slots own empty op ranges at the segment boundaries.
    assert_eq!(program.op_offsets, vec![0, 1, 1, 2, 3, 3, 3]);
    assert_eq!(program.op_offsets[2], program.op_offsets[1]);
    assert_eq!(program.op_offsets[6], program.op_offsets[5]);
    // Wildcard deme selectors keep every segment executable.
    assert_eq!(program.deme_selector_types, vec![0; 6]);
    assert_eq!(program.deme_selector_offsets.len(), 7);
    assert_eq!(program.event_callback_count(0), 1);
    assert_eq!(program.event_callback_count(1), 0);
    assert_eq!(program.event_callback_count(2), 2);
    // The CSR ops themselves are untouched.
    assert_eq!(program.op_types.len(), 3);
}

/// Clearing demotes callback slots to inert CSR slots; re-installing on a
/// mixed (CSR + demoted) program appends fresh slots at the segment ends
/// instead of resurrecting the demoted indexes.
#[test]
fn clear_then_reinstall_keeps_pairing_repairable() {
    let mut program = two_event_csr_program();
    program.install_callback_lists(none_callbacks([1, 0, 2, 0]));
    program.clear_callbacks();

    // Mixed program (CSR ops present): slots demote, lists empty.
    assert_eq!(program.python_callback_slots, vec![-1; 6]);
    assert!(program.python_callbacks.iter().all(|l| l.is_empty()));
    assert_eq!(program.n_hooks, 6);

    program.install_callback_lists(none_callbacks([0, 0, 1, 0]));
    // The re-registered callback lands after the demoted slots.
    assert_eq!(program.python_callback_slots[6], 0);
    assert_eq!(program.hook_offsets, vec![0, 2, 4, 7, 7]);
    assert_eq!(program.n_hooks, 7);
    assert_eq!(program.event_callback_count(2), 1);
}

/// A callbacks-only program (no CSR ops) resets to the empty default on
/// clear, so repeated set/clear cycles cannot grow the program forever.
#[test]
fn clear_resets_callbacks_only_program() {
    let mut program = HookProgram::default();
    program.install_callback_lists(none_callbacks([2, 0, 0, 0]));
    assert_eq!(program.n_hooks, 2);

    program.clear_callbacks();
    assert_eq!(program.n_hooks, 0);
    assert!(program.python_callback_slots.is_empty());

    program.install_callback_lists(none_callbacks([1, 0, 0, 0]));
    assert_eq!(program.n_hooks, 1);
    assert_eq!(program.python_callback_slots, vec![0]);
}

/// Shrinking an event's callback list demotes the surplus slots to inert
/// ``-1`` entries in place: CSR columns (offsets, n_hooks, segment bounds)
/// stay untouched and no slot references the shorter list out of range.
#[test]
fn install_callback_lists_shrink_demotes_surplus_slots() {
    let mut program = two_event_csr_program();
    program.install_callback_lists(none_callbacks([1, 0, 2, 0]));
    let hook_offsets_before = program.hook_offsets.clone();
    let op_offsets_before = program.op_offsets.clone();
    let n_hooks_before = program.n_hooks;

    // Event 2 shrinks from two callbacks to none; event 0 keeps its one.
    program.install_callback_lists(none_callbacks([1, 0, 0, 0]));

    assert_eq!(program.n_hooks, n_hooks_before);
    assert_eq!(program.hook_offsets, hook_offsets_before);
    assert_eq!(program.op_offsets, op_offsets_before);
    assert_eq!(program.python_callback_slots, vec![-1, 0, -1, -1, -1, -1]);
    assert_eq!(program.event_callback_count(0), 1);
    assert_eq!(program.event_callback_count(2), 0);

    // Growing back appends fresh slots (the demoted ones stay inert);
    // the event's callbacks remain addressable with indexes 0 and 1.
    program.install_callback_lists(none_callbacks([1, 0, 2, 0]));
    assert_eq!(program.python_callback_slots[6..=7], vec![0, 1]);
    assert_eq!(program.n_hooks, 8);
    assert_eq!(program.event_callback_count(2), 2);
}
