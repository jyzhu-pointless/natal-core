// Minimal C-ABI bridge used to validate that Rust can call a Numba `@cfunc`
// function pointer while passing raw NumPy array pointers and flat config
// buffers.
//
// This is a research spike, not part of the production backend.

/// Call a Numba `@cfunc` hook with the flat-buffer ABI explored for P5.
///
/// The callee is expected to have the signature:
///
/// ```c
/// int64_t hook(
///     double *individual_count,
///     double *sperm_storage,
///     int64_t tick,
///     int64_t deme_id,
///     double *config_f64,
///     int64_t *config_i64);
/// ```
///
/// # Parameters
/// - `hook`: Address of the Numba `@cfunc` adapter.
/// - `individual_count`: Mutable pointer to the individual-count array data.
/// - `sperm_storage`: Mutable pointer to the sperm-storage array data.
/// - `tick`: Current tick.
/// - `deme_id`: Current deme id.
/// - `config_f64`: Pointer to the flat float64 config buffer.
/// - `config_i64`: Pointer to the flat int64 config buffer.
///
/// # Returns
/// The hook's `int64` result code (`0` continue, `1` stop).
///
/// # Safety
/// The caller must guarantee that `hook` points to a valid C-ABI function with
/// exactly the signature above, and that all pointers are valid for the duration
/// of the call.  This function is only intended for the P5 research spike.
#[no_mangle]
pub extern "C" fn call_hook(
    hook: usize,
    individual_count: *mut f64,
    sperm_storage: *mut f64,
    tick: i64,
    deme_id: i64,
    config_f64: *const f64,
    config_i64: *const i64,
) -> i64 {
    let hook_fn: extern "C" fn(
        *mut f64,
        *mut f64,
        i64,
        i64,
        *const f64,
        *const i64,
    ) -> i64 = unsafe { std::mem::transmute(hook) };
    hook_fn(individual_count, sperm_storage, tick, deme_id, config_f64, config_i64)
}

/// A tiny no-op used to validate the loader without Numba.
///
/// # Parameters
/// - `x`: Input value.
///
/// # Returns
/// `x + 1`.
#[no_mangle]
pub extern "C" fn add_one(x: i64) -> i64 {
    x + 1
}
