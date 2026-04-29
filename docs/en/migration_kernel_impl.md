# Migration Kernel Internal Implementation

This article covers the underlying mechanisms of kernel-based migration, including the conversion process from kernel to spatial offsets, boundary handling strategies, and routing for heterogeneous kernels.

It is suitable for developers who need to understand the internal implementation or debug migration behavior.

## 1. Overall Data Flow

```
Input kernel (3x3 / 5x5 matrix)
        │
        ▼
_build_kernel_offset_table()     ← one-time: kernel → compact offset table
        │
        ▼
for each source deme in prange:
    _build_source_kernel_sparse_row()  ← per deme: apply offset table, handle boundaries
    migrate_scalar_bucket()            ← per bucket: discretize emigration amounts
    migrate_sperm_bucket()
```

## 2. Kernel to Offset Table Conversion

`_build_kernel_offset_table()` is executed once at the start of the entire migration call, and the result is reused by all source demes.

### 2.1 Input

- `migration_kernel`: an odd-dimension 2D array whose center corresponds to the source deme itself
- `kernel_include_center`: whether to retain the center cell (self-loop)

### 2.2 Conversion Process

```python
# 3x3 kernel example
# [[0.0, 1.0, 0.0],
#  [1.0, 0.0, 1.0],    ← center (1,1) weight = 0 (excluding self)
#  [0.0, 1.0, 0.0]]

center_row = 3 // 2 = 1
center_col = 3 // 2 = 1

# Iterate over all kernel coordinates, convert to source-relative offsets:
for kernel_row in [0,1,2]:
    for kernel_col in [0,1,2]:
        weight = kernel[kernel_row, kernel_col]
        if weight <= 0: continue          # skip zero weight
        if (not include_center) and is_center: continue  # skip center

        d_row = kernel_row - center_row   # row offset (relative to source)
        d_col = kernel_col - center_col   # column offset (relative to source)
```

### 2.3 Output Format

| Array | Type | Length | Meaning |
|------|------|--------|---------|
| `d_row` | `int64[:]` | nnz | Row offset for each valid neighbor |
| `d_col` | `int64[:]` | nnz | Column offset for each valid neighbor |
| `weights` | `float64[:]` | nnz | Original weight for each valid neighbor |
| `nnz` | `int` | — | Total number of valid neighbors |
| `kernel_total_sum` | `float` | — | Sum of all positive kernel weights (normalization baseline) |

For the above 3x3 von Neumann kernel (center excluded), the output is:

```
d_row  = [-1,  0,  0,  1]
d_col  = [ 0, -1,  1,  0]
weights = [1.0, 1.0, 1.0, 1.0]
nnz = 4
kernel_total_sum = 4.0
```

## 3. Single Source Migration Row Construction

`_build_source_kernel_sparse_row()` generates the emigration probability distribution for each source deme.

### 3.1 Input

- `source_idx`: flat index of the source deme
- `topology_rows / topology_cols`: grid dimensions
- `topology_wrap`: whether boundary wrapping is enabled
- Offset table (d_row, d_col, weights, nnz, kernel_total_sum)
- `adjust_on_edge`: boundary adjustment mode

### 3.2 Coordinate Decoding

```python
src_row = source_idx // topology_cols
src_col = source_idx % topology_cols
```

### 3.3 Neighbor Generation and Boundary Handling

```python
for idx in range(nnz):
    dst_row = src_row + d_row[idx]
    dst_col = src_col + d_col[idx]

    if topology_wrap:
        # Periodic boundary: wrap around to the opposite side
        dst_row %= topology_rows
        dst_col %= topology_cols
    elif dst_row < 0 or dst_row >= topology_rows or dst_col < 0 or dst_col >= topology_cols:
        # Hard boundary: discard out-of-range neighbors
        continue

    # Write valid destination
    row_dst_idx[count] = dst_row * topology_cols + dst_col
    row_dst_prob[count] = weights[idx]
    total += weights[idx]
    count += 1
```

### 3.4 Probability Normalization

Two modes with diverging logic:

```python
if adjust_on_edge:
    # Mode 1: Normalize to 1.0 — all demes emigrate the same total amount
    inv_total = 1.0 / total
    for idx in range(count):
        row_dst_prob[idx] *= inv_total
else:
    # Mode 2 (default): Scale by kernel_total_sum — boundary naturally emigrates less
    inv_kernel_sum = 1.0 / kernel_total_sum
    for idx in range(count):
        row_dst_prob[idx] *= inv_kernel_sum
```

### 3.5 Mode Comparison

Using a 3x3 von Neumann kernel (`kernel_total_sum = 4.0`) as an example:

| Scenario | adjust_on_edge | Valid Neighbors | Probability per Neighbor | Total Migration Rate |
|----------|----------------|-----------------|--------------------------|---------------------|
| Interior deme | — | 4 | `1.0 / 4.0 = 0.25` | `rate * 1.0` |
| Corner deme | `False` | 2 | `1.0 / 4.0 = 0.25` | `rate * 0.5` |
| Corner deme | `True` | 2 | `1.0 / 2.0 = 0.50` | `rate * 1.0` |

When `topology_wrap=True`, all demes have a full set of neighbors, so both modes are equivalent.

### 3.6 Non-Uniform Weight Kernels

For kernels where weights are not all 1 (e.g., Gaussian kernels), `kernel_total_sum` preserves the relative weight structure:

```
5x5 Gaussian kernel:
  kernel_total_sum = Σ all positive weights ≈ 20.3

Interior deme (25 neighbors all valid):
  neighbor_prob = weight / 20.3           # preserves Gaussian shape
  total_rate = rate * (20.3 / 20.3) = rate  # equivalent to rate * 1.0

Boundary deme (15 neighbors valid):
  neighbor_prob = weight / 20.3           # relative weights unchanged
  total_rate = rate * (effective_sum / 20.3)  # < rate
```

## 4. Main Migration Function

`apply_spatial_kernel_migration()` processes all source demes in parallel within `prange`.

### 4.1 Thread-Local Buffers

To avoid write contention in `prange`, each thread holds private output buffers:

```python
# Thread-local output tensors
out_ind_by_thread   = np.zeros((n_threads, n_demes, n_sexes, n_ages, n_genotypes))
out_sperm_by_thread = np.zeros((n_threads, n_demes, n_ages, n_genotypes, n_genotypes))

# Thread-local sparse row buffer (size = max kernel nnz)
row_dst_idx_by_thread  = np.full((n_threads, max_nnz), -1)
row_dst_prob_by_thread = np.zeros((n_threads, max_nnz))

# Thread-local distribution scratch buffer
distributed_by_thread  = np.zeros((n_threads, max_nnz))
```

After `prange` completes, all thread-local tensors are merged deterministically:

```python
out_ind = np.zeros_like(ind_count_all)
out_sperm = np.zeros_like(sperm_store_all)
for thread_id in range(n_threads):
    out_ind += out_ind_by_thread[thread_id]
    out_sperm += out_sperm_by_thread[thread_id]
```

### 4.2 Complete Flow

```
For each source deme (prange parallel):
  1. kid = deme_kernel_ids[src]          ← select the kernel for this deme
  2. Get thread-local buffers for thread_id
  3. _build_source_kernel_sparse_row()   ← build emigration probability distribution
  4. for each age × genotype:
       migrate_scalar_bucket()           ← emigrate virgin female / male etc.
       migrate_sperm_bucket()            ← emigrate sperm-coupled female + sperm
                                          (destinations are identical, guaranteeing synchronicity)

Merge all thread outputs (deterministic addition)
```

## 5. Heterogeneous Kernel Routing

### 5.1 Prerequisites

Pass `kernel_bank` + `deme_kernel_ids`:

```python
SpatialPopulation(
    demes=demes,
    topology=SquareGrid(rows=1, cols=3),
    kernel_bank=(right_only, left_only),         # 2 different kernels
    deme_kernel_ids=np.array([0, 1, 0]),          # deme 0→kernel[0], deme 1→kernel[1], deme 2→kernel[0]
    migration_rate=1.0,
)
```

### 5.2 Initialization Phase (Python Layer)

`_build_heterogeneous_kernel_arrays()` pre-builds offset tables for each unique kernel in the kernel bank:

```python
n_kernels = len(kernel_bank)
kernel_d_row = np.zeros((n_kernels, max_kernel_size))
kernel_d_col = np.zeros((n_kernels, max_kernel_size))
kernel_weights = np.zeros((n_kernels, max_kernel_size))
kernel_nnzs = np.zeros(n_kernels)
kernel_total_sums = np.zeros(n_kernels)

for k in range(n_kernels):
    d_r, d_c, w, nnz, total = _build_kernel_offset_table(kernel_bank[k])
    kernel_d_row[k, :nnz] = d_r[:nnz]
    # ...
```

These pre-built arrays are passed into `apply_spatial_kernel_migration` during migration.

### 5.3 Migration Phase (Inside Numba prange)

```python
for src in prange(n_demes):
    kid = deme_kernel_ids[src]       # table lookup
    nnz_k = kernel_nnzs[kid]         # number of valid neighbors for this kernel
    total_k = kernel_total_sums[kid] # total weight for this kernel

    # Build migration row using this kernel's offset table
    _build_source_kernel_sparse_row(
        ...,
        d_row=kernel_d_row[kid],
        d_col=kernel_d_col[kid],
        weights=kernel_weights[kid],
        kernel_nnz=nnz_k,
        kernel_total_sum=total_k,
        ...
    )
```

This approach does not pre-build an O(n_demes²) dense adjacency matrix; each source deme constructs its sparse migration row on demand within `prange`.

## 6. Comparison with Adjacency Mode

| Dimension | Adjacency Mode | Kernel Mode |
|-----------|---------------|-------------|
| Data structure | Dense `(n, n)` matrix or CSR | Compact offset table `(nnz,)` |
| Space complexity | O(n²) or O(nnz) | O(kernel_nnz) |
| Migration row construction | Pre-built for all | On-demand (inside prange) |
| Topology awareness | Indirectly via adjacency matrix | Directly using grid coordinates + offsets |
| Boundary handling | Pre-encoded in the matrix | Runtime decision (wrap/clip) |
| Heterogeneous kernels | Not supported (or requires pre-built n² dense matrix) | Per-kernel grouped offset tables |

For large-scale grids (e.g., 501×501 = 251001 demes), kernel mode avoids the storage and access overhead of an O(n²) adjacency matrix.

## 7. Key Decisions and Edge Cases

### 7.1 `kernel_include_center`

- `True`: the source deme itself is also a migration target (self-loop)
- `False` (default): the kernel center is excluded, migration only goes to neighbors

### 7.2 Zero Weight Handling

Entries in the kernel with `weight <= 0` are skipped during offset table construction and do not participate in subsequent calculations.

### 7.3 Early Return for `rate <= 0`

When `migration_rate <= 0`, `apply_spatial_kernel_migration` directly returns the input state without performing any computation.

### 7.4 Thread Safety

All mutable intermediate results (output tensors, sparse row buffers, distribution arrays) are allocated per thread. They are merged via deterministic addition (not atomic operations), ensuring reproducibility.

### 7.5 Floating Point Drift Handling

Virgin female count = female_total - stored_total, both from `float64` arrays; subtraction can produce very small negative values. Truncation with tolerance ensures non-negativity:

```python
virgin_count = female_total - stored_total
if virgin_count < 0.0 and abs(virgin_count) < 1e-10:
    virgin_count = 0.0
```

## 8. Related Files

| File | Content |
|------|---------|
| `src/natal/kernels/migration/kernel.py` | `_build_kernel_offset_table`, `_build_source_kernel_sparse_row`, `apply_spatial_kernel_migration` |
| `src/natal/kernels/migration/adjacency.py` | `migrate_scalar_bucket`, `migrate_sperm_bucket` |
| `src/natal/kernels/spatial_migration_kernels.py` | `run_spatial_migration`, `apply_spatial_adjacency_migration` (dispatch entry points) |
| `src/natal/spatial_population.py` | `_build_heterogeneous_kernel_arrays`, runtime scheduling |
| `src/natal/kernels/templates/spatial_lifecycle_*.tmpl.py` | Code generation templates (calling `run_spatial_migration` in the njit path) |
