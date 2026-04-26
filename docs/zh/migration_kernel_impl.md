# Migration Kernel 底层实现

本文梳理 kernel 模式迁移的底层机制，涵盖 kernel 到空间偏移的转换过程、边界处理策略、以及异构 kernel 的路由方式。

适合需要理解内部实现或调试迁移行为的开发者阅读。

## 1. 整体数据流

```
传入 kernel (3x3 / 5x5 矩阵)
        │
        ▼
_build_kernel_offset_table()     ← 一次性：kernel → 紧凑偏移表
        │
        ▼
for each source deme in prange:
    _build_source_kernel_sparse_row()  ← 按 deme：应用偏移表，处理边界
    migrate_scalar_bucket()            ← 按 bucket：离散化迁出量
    migrate_sperm_bucket()
```

## 2. Kernel → 偏移表转换

`_build_kernel_offset_table()` 在整个迁移调用开始时执行一次，结果被所有 source deme 复用。

### 2.1 输入

- `migration_kernel`：奇数维二维数组，中心对应 source deme 自身
- `kernel_include_center`：是否保留中心格（self-loop）

### 2.2 转换过程

```python
# 3x3 kernel 示例
# [[0.0, 1.0, 0.0],
#  [1.0, 0.0, 1.0],    ← 中心 (1,1) 权重 = 0（不包含自己）
#  [0.0, 1.0, 0.0]]

center_row = 3 // 2 = 1
center_col = 3 // 2 = 1

# 遍历所有 kernel 坐标，转换为 source-relative 偏移：
for kernel_row in [0,1,2]:
    for kernel_col in [0,1,2]:
        weight = kernel[kernel_row, kernel_col]
        if weight <= 0: continue          # 跳过零权重
        if (not include_center) and is_center: continue  # 跳过中心

        d_row = kernel_row - center_row   # 行偏移（相对于 source）
        d_col = kernel_col - center_col   # 列偏移（相对于 source）
```

### 2.3 输出格式

| 数组 | 类型 | 长度 | 含义 |
|------|------|------|------|
| `d_row` | `int64[:]` | nnz | 每个有效邻居的行偏移 |
| `d_col` | `int64[:]` | nnz | 每个有效邻居的列偏移 |
| `weights` | `float64[:]` | nnz | 每个有效邻居的原始权重 |
| `nnz` | `int` | — | 有效邻居总数 |
| `kernel_total_sum` | `float` | — | 所有正向 kernel 权重之和（归一化基准） |

对上述 3x3 von Neumann kernel（中心排除），输出为：

```
d_row  = [-1,  0,  0,  1]
d_col  = [ 0, -1,  1,  0]
weights = [1.0, 1.0, 1.0, 1.0]
nnz = 4
kernel_total_sum = 4.0
```

## 3. 单源迁移行构建

`_build_source_kernel_sparse_row()` 为每个 source deme 生成其迁出概率分布。

### 3.1 输入

- `source_idx`：source deme 的扁平索引
- `topology_rows / topology_cols`：网格尺寸
- `topology_wrap`：是否周期边界
- 偏移表（d_row, d_col, weights, nnz, kernel_total_sum）
- `adjust_on_edge`：边界调整模式

### 3.2 坐标解码

```python
src_row = source_idx // topology_cols
src_col = source_idx % topology_cols
```

### 3.3 邻居生成与边界处理

```python
for idx in range(nnz):
    dst_row = src_row + d_row[idx]
    dst_col = src_col + d_col[idx]

    if topology_wrap:
        # 周期边界：超出范围的对侧折回
        dst_row %= topology_rows
        dst_col %= topology_cols
    elif dst_row < 0 or dst_row >= topology_rows or dst_col < 0 or dst_col >= topology_cols:
        # 硬边界：丢弃超出范围的邻居
        continue

    # 写入有效目的地
    row_dst_idx[count] = dst_row * topology_cols + dst_col
    row_dst_prob[count] = weights[idx]
    total += weights[idx]
    count += 1
```

### 3.4 概率归一化

两种模式的分叉逻辑：

```python
if adjust_on_edge:
    # 模式 1：归一化到 1.0 — 所有 deme 迁出相同总量
    inv_total = 1.0 / total
    for idx in range(count):
        row_dst_prob[idx] *= inv_total
else:
    # 模式 2（默认）：按 kernel_total_sum 缩放 — 边界自然迁出更少
    inv_kernel_sum = 1.0 / kernel_total_sum
    for idx in range(count):
        row_dst_prob[idx] *= inv_kernel_sum
```

### 3.5 模式对比

以 3x3 von Neumann kernel（`kernel_total_sum = 4.0`）为例：

| 场景 | adjust_on_edge | 有效邻居数 | 每个邻居概率 | 总迁移率 |
|------|---|---|---|---|
| 内部 deme | — | 4 | `1.0 / 4.0 = 0.25` | `rate * 1.0` |
| 角落 deme | `False` | 2 | `1.0 / 4.0 = 0.25` | `rate * 0.5` |
| 角落 deme | `True` | 2 | `1.0 / 2.0 = 0.50` | `rate * 1.0` |

当 `topology_wrap=True` 时，所有 deme 都有完整邻居数，两种模式等价。

### 3.6 非均匀权重 Kernel

对于权重不全是 1 的 kernel（如高斯核），`kernel_total_sum` 保留了相对权重结构：

```
5x5 高斯核:
  kernel_total_sum = Σ 所有正向权重 ≈ 20.3

内部 deme (25 邻居全有效):
  neighbor_prob = weight / 20.3           # 保留了高斯形状
  total_rate = rate * (20.3 / 20.3) = rate  # 等价于 rate * 1.0

边界 deme (15 邻居有效):
  neighbor_prob = weight / 20.3           # 相对权重不变
  total_rate = rate * (effective_sum / 20.3)  # < rate
```

## 4. 主迁移函数

`apply_spatial_kernel_migration()` 在 `prange` 中并行处理所有 source deme。

### 4.1 线程本地缓冲区

为避免 `prange` 中的写竞争，每个线程持有私有的输出缓冲区：

```python
# 线程本地输出张量
out_ind_by_thread   = np.zeros((n_threads, n_demes, n_sexes, n_ages, n_genotypes))
out_sperm_by_thread = np.zeros((n_threads, n_demes, n_ages, n_genotypes, n_genotypes))

# 线程本地的稀疏行缓冲区（大小 = 最大 kernel nnz）
row_dst_idx_by_thread  = np.full((n_threads, max_nnz), -1)
row_dst_prob_by_thread = np.zeros((n_threads, max_nnz))

# 线程本地的分发 scratch buffer
distributed_by_thread  = np.zeros((n_threads, max_nnz))
```

`prange` 结束后，所有线程的局部张量确定性合并：

```python
out_ind = np.zeros_like(ind_count_all)
out_sperm = np.zeros_like(sperm_store_all)
for thread_id in range(n_threads):
    out_ind += out_ind_by_thread[thread_id]
    out_sperm += out_sperm_by_thread[thread_id]
```

### 4.2 完整流程

```
对每个 source deme（prange 并行）:
  1. kid = deme_kernel_ids[src]          ← 选择该 deme 的 kernel
  2. 获取 thread_id 对应的本地缓冲区
  3. _build_source_kernel_sparse_row()   ← 构建迁出概率分布
  4. for each age × genotype:
       migrate_scalar_bucket()           ← 迁出 virgin female / male 等
       migrate_sperm_bucket()            ← 迁出 sperm-coupled female + sperm
                                          （destination 完全一致，保证同步）

合并所有线程的输出（确定性加法）
```

## 5. 异构 Kernel 路由

### 5.1 前置条件

传入 `kernel_bank` + `deme_kernel_ids`：

```python
SpatialPopulation(
    demes=demes,
    topology=SquareGrid(rows=1, cols=3),
    kernel_bank=(right_only, left_only),         # 2 个不同的 kernel
    deme_kernel_ids=np.array([0, 1, 0]),          # deme 0→kernel[0], deme 1→kernel[1], deme 2→kernel[0]
    migration_rate=1.0,
)
```

### 5.2 初始化阶段（Python 层）

`_build_heterogeneous_kernel_arrays()` 为 kernel_bank 中每个唯一 kernel 预构建偏移表：

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

这些预构建的数组在迁移时传入 `apply_spatial_kernel_migration`。

### 5.3 迁移阶段（Numba prange 内）

```python
for src in prange(n_demes):
    kid = deme_kernel_ids[src]       # 查表
    nnz_k = kernel_nnzs[kid]         # 该 kernel 的有效邻居数
    total_k = kernel_total_sums[kid] # 该 kernel 的总权重

    # 使用该 kernel 的偏移表构建迁移行
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

这种方式不会预构建 O(n_demes²) 的稠密邻接矩阵，每个 source deme 在 `prange` 内按需构建稀疏迁移行。

## 6. 与 Adjacency 模式的对比

| 维度 | Adjacency 模式 | Kernel 模式 |
|------|---------------|------------|
| 数据结构 | 稠密 `(n, n)` 矩阵或 CSR | 紧凑偏移表 `(nnz,)` |
| 空间复杂度 | O(n²) 或 O(nnz) | O(kernel_nnz) |
| 迁移行构建 | 预构建全部 | 按需构建（prange 内） |
| 拓扑感知 | 通过 adjacency 矩阵间接体现 | 直接使用 grid 坐标 + 偏移 |
| 边界处理 | 预先编码在矩阵中 | 运行时判断（wrap/clip） |
| 异构 kernel | 不支持（或需预构建 n² 稠密矩阵） | 按 kernel 分组偏移表 |

对于大规模网格（如 501×501 = 251001 demes），kernel 模式避免了 O(n²) 邻接矩阵的存储和访问开销。

## 7. 关键决策与边界条件

### 7.1 `kernel_include_center`

- `True`：source deme 自身也作为迁出目标（self-loop）
- `False`（默认）：kernel 中心被排除，迁移只到邻居

### 7.2 零权重处理

kernel 中 `weight <= 0` 的条目在偏移表构建时被跳过，不参与后续计算。

### 7.3 `rate <= 0` 的提前返回

当 `migration_rate <= 0` 时，`apply_spatial_kernel_migration` 直接返回输入状态，不做任何计算。

### 7.4 线程安全

所有可变的中间结果（输出张量、稀疏行缓冲区、分发数组）都按线程分配。最终通过确定性加法合并（而非原子操作），保证可复现性。

### 7.5 浮点漂移处理

virgin female 数量 = female_total - stored_total，两者均来自 `float64` 数组，减法可能产生极小的负值。通过容差截断保证非负：

```python
virgin_count = female_total - stored_total
if virgin_count < 0.0 and abs(virgin_count) < 1e-10:
    virgin_count = 0.0
```

## 8. 相关文件

| 文件 | 内容 |
|------|------|
| `src/natal/kernels/migration/kernel.py` | `_build_kernel_offset_table`、`_build_source_kernel_sparse_row`、`apply_spatial_kernel_migration` |
| `src/natal/kernels/migration/adjacency.py` | `migrate_scalar_bucket`、`migrate_sperm_bucket` |
| `src/natal/kernels/spatial_migration_kernels.py` | `run_spatial_migration`、`apply_spatial_adjacency_migration`（分发入口） |
| `src/natal/spatial_population.py` | `_build_heterogeneous_kernel_arrays`、运行时调度 |
| `src/natal/kernels/templates/spatial_lifecycle_*.tmpl.py` | 代码生成模板（在 njit 路径中调用 `run_spatial_migration`） |
