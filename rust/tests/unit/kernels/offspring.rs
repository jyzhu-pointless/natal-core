use super::compute_offspring_tensor_flat;

/// Hand-computed Mendelian case: every homozygote cross yields only
/// its own genotype; mixed crosses yield nothing (the fusion map
/// only joins matching haplotypes here); skip-zero must not skip
/// nonzero contributions.
#[test]
fn offspring_kernel_matches_hand_computed_mendelian_cells() {
    // z = 2 (A|A, a|a), g = 2 (A, a); both sexes segregate purely.
    let meiosis = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]; // (2, 2, 2)
    let fusion = [
        1.0, 0.0, // A,A -> (A|A, a|a)
        0.0, 0.0, // A,a
        0.0, 0.0, // a,A
        0.0, 1.0, // a,a
    ];
    let out = compute_offspring_tensor_flat(&meiosis, &fusion, 2, 2);
    // Cell order ((gf,gm) pairs x 2 offspring columns):
    // (A|A,A|A)=[1,0], (A|A,a|a)=[0,0], (a|a,A|A)=[0,0], (a|a,a|a)=[0,1].
    assert_eq!(out, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
}

/// Statement-order parity: the accumulated sum must follow
/// (hf, hm) lexicographic order with zero-terms skipped — pinned by
/// a table where different orderings give different last-bits.
#[test]
fn offspring_kernel_accumulates_in_pinned_order() {
    // One (gf, gm, go) cell fed by two nonzero (hf, hm) pairs whose
    // contributions do not commute exactly in f64.
    let z = 1;
    let g = 2;
    let meiosis = [1.0, 1.0, 1.0, 1.0]; // both sexes use both gtypes
    let fusion = [
        0.1, // (0,0) -> go 0
        0.0,
        0.0,       // (0,1), (1,0) unused at go 1 (z=1 → fusion is (g,g,1))
        1.0 / 3.0, // (1,1) -> go 0
    ];
    // fusion flat (2,2,1): [f(0,0), f(0,1), f(1,0), f(1,1)] = [0.1, 0, 0, 1/3]
    let expect: f64 = 1.0 * 0.1 + 1.0 * (1.0f64 / 3.0);
    let out = compute_offspring_tensor_flat(&meiosis, &fusion, z, g);
    assert_eq!(out[0].to_bits(), expect.to_bits());
}
