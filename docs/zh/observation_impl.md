# Observation 历史记录：开发者实现文档

本文档面向 NATAL Core 的维护者和贡献者，详细说明 observation-based history recording 的实现原理、数据流和各模块职责。

## 整体架构

Observation 历史记录涉及四个层次，并通过 NamedTuple 参数束在 Kernel 层传递：

```
用户层 API          →  BasePopulation / SpatialPopulation 的 record_observation 属性
                       SpatialPopulation 的 _spatial_topo / _migration_params / _compact_meta
Observation 层     →  observation.py: Observation, ObservationFilter, build_mask
                       observation_record.py: CompactMeta, build_observation_row_spatial
Kernel 层          →  spatial_simulation_kernels.py 中的 recording 通路
                       spatial_lifecycle_*.tmpl.py 代码生模板
导出层             →  state_translation.py: output_history / *observation_history_to_readable_dict
```

## NamedTuple 参数束

为避免散落的 13+ 个参数在多层函数间传递，空间内核调度使用四个 NamedTuple 收敛参数：

| NamedTuple | 定义位置 | 收敛的参数 | 生命周期 |
|-----------|---------|-----------|---------|
| `SpatialTopology` | `spatial_topology.py` | `rows`, `cols`, `wrap` | 构造时固化 |
| `MigrationParams` | `spatial_topology.py` | `kernel`, `include_center`, `rate`, `adjust_on_edge` | 构造时固化 |
| `HeterogeneousKernelParams` | `spatial_topology.py` | `deme_kernel_ids`, `d_row`, `d_col`, `weights`, `nnzs`, `total_sums`, `max_nnz` | 每次 `run()` 重建 |
| `CompactMeta` | `observation_record.py` | `offsets`, `deme_map`, `n_demes_per_group`, `selected_n`, `mode_aggregate`, `row_size` | `record_observation` 设置时重建 |

`migration_mode` 和 `adjacency` 不包含在 `MigrationParams` 中，因为它们由 `_effective_migration_route()` 在每次调用时动态解析（可选择 adjacency / kernel / auto），属于**路由策略**而非**迁移参数**。

## 数据流

```
用户定义 groups
       ↓
ObservationFilter.build_filter(groups) → Observation 对象（含 specs、labels、collapse_age）
       ↓
BasePopulation._build_observation_mask(obs) → 4D float64 mask (n_groups, n_sexes, n_ages, n_genotypes)
       ↓
_build_deme_info() → demean_modes dict → build_compact_metadata() → CompactMeta
       ↓
mask + CompactMeta 传入 Numba kernel（observation_mask + compact_meta 参数）
       ↓
kernel 每个 record_every 步：
  build_observation_row_spatial(ind, mask, compact_meta) → 紧凑 flat row
  row = [tick, compact_row]
       ↓
Python 层：_process_kernel_history() → history.append((tick, row.copy()))
       ↓
导出：output_history() 自动检测 record_observation →  dispatch 到 *observation_history_to_readable_dict
```

## 数据格式对比

### 原始模式（record_observation=None）

**Panmictic** 每行：
```
[tick, ind[0,0,0], ind[0,0,1], ..., ind[n_sexes-1, n_ages-1, n_genotypes-1],
 sperm[0,0,0,0], ..., sperm[n_ages-1, n_genotypes-1, n_genotypes-1]]
```
行大小 = `1 + n_sexes × n_ages × n_genotypes + n_ages × n_genotypes²`

**Spatial** 每行：
```
[tick, flat_ind_deme_0, ..., flat_ind_deme_n-1, flat_sperm_deme_0, ..., flat_sperm_deme_n-1]
```
其中 `flat_ind_deme_d = ind_d[d].ravel()`（每个 deme 的行大小 = `n_sexes × n_ages × n_genotypes`）

### 观测模式（record_observation 已设置）

**Panmictic** 每行：
```
[tick, observed[0,0,0], observed[0,0,1], ..., observed[n_groups-1, n_sexes-1, n_ages-1]]
```
行大小 = `1 + n_groups × n_sexes × n_ages`

**Spatial** 每行：
```
[tick, observed[0,0,0,0], ..., observed[n_demes-1, n_groups-1, n_sexes-1, n_ages-1]]
```
行大小 = `1 + n_demes × n_groups × n_sexes × n_ages`

## 核心模块详解

### 1. Observation 系统（observation.py）

#### Observation 对象

`Observation` 是不可变 dataclass，包含：
- `filter`: 创建它的 `ObservationFilter` 引用
- `diploid_genotypes`: 用于解析 genotype selector 的基因型序列
- `specs`: 标准化后的分组规格 `[(name, {key: value}), ...]`
- `labels`: 分组标签元组 `(name_0, name_1, ...)`
- `collapse_age`: 是否压缩年龄维度

关键方法：
- `apply(individual_count)` → `(n_groups, n_sexes, n_ages)`：对给定的 count 数组执行观测投影
- `build_mask(n_sexes, n_ages, n_genotypes)` → `(n_groups, n_sexes, n_ages, n_genotypes)`：编译 4D 二进制掩码，供 kernel 使用。注意该方法始终返回 4D 掩码（`collapse_age=False`），因为 kernel 需要完整的第 3 维
- `to_dict()` → metadata dict：序列化 labels、collapse_age、n_groups 等元数据

#### ObservationFilter

`ObservationFilter(registry)` 负责将用户定义的 group specs 编译为 Observation：
- `build_filter(diploid_genotypes, groups, collapse_age)` → `Observation`：完整编译流程
- `create_observation(...)`：`build_filter` 的别名
- `build_mask_from_specs(...)` → 4D float64 mask：核心编译函数，循环填充 mask：
  ```python
  for gi in range(n_groups):
      for gidx in per_genotypes[gi]:
          for s in per_sexes[gi]:
              for a in range(n_ages):
                  if per_age_preds[gi](a):
                      mask[gi, s, a, gidx] = 1.0
  ```

#### apply_rule

纯函数 `apply_rule(individual_count, rule)` → `observed`：
- 3D count × 4D mask：`sum(mask * count[None, :, :, :], axis=-1)`
- 2D count × 3D mask（discrete generation + collapse_age）：类似 broadcast + sum

### 2. 人口模型层（base_population.py）

#### record_observation 属性

```python
@property
def record_observation(self) -> Optional[Observation]: ...

@record_observation.setter
def record_observation(self, obs: Optional[Observation]) -> None:
    self._observation = obs
    if obs is not None:
        self._observation_mask = self._build_observation_mask(obs)
```

Setter 在设置 observation 的同时编译 4D 二进制掩码。`_observation_mask` 的类型是 `NDArray[np.float64]`，shape `(n_groups, n_sexes, n_ages, n_genotypes)`。

#### set_observations 快捷方式

```python
def set_observations(self, groups, *, collapse_age=False):
    obs_filter = ObservationFilter(self.index_registry)
    self._observation = obs_filter.build_filter(
        diploid_genotypes=self.species,
        groups=groups,
        collapse_age=collapse_age,
    )
    self._observation_mask = self._build_observation_mask(self._observation)
```

内部流程：创建 `ObservationFilter` → `build_filter` → 编译 mask → 分别存储 `_observation` 和 `_observation_mask`。

#### _clone 的兼容性

`_clone()` 被 `SpatialBuilder` 用于高效克隆 deme。克隆时 `_observation` 和 `_observation_mask` 被重置为 `None`（第 281-282 行），因为每个克隆需独立设置观测——observation mask 取决于 deme 的 state shape（虽然通常一样，但需要显式设置）。

### 3. Kernel 集成

#### Panmictic 内核（simulation_kernels.py）

`run_with_hooks` / `run_discrete_with_hooks` 内部调用 `build_observation_row_panmictic()`：

```python
if observation_mask is not None:
    flat_state[1:] = build_observation_row_panmictic(ind_count, observation_mask)
else:
    flat_state[1:1+ind_size] = ind_count.ravel()
    flat_state[1+ind_size:] = sperm_store.ravel()  # 如有
```

关键参数：
- `observation_mask: Optional[NDArray[np.float64]]` — 4D 掩码
- `build_observation_row_panmictic` 是 `observation_record.py` 中的独立 njit 函数

#### Spatial 内核（observation_record.py）

`build_observation_row_spatial()` 负责紧凑行的构建，替代了原先内联的 broadcast + `deme_selector` 掩码方式：

```python
# observation_record.py
@njit_switch(cache=True)
def build_observation_row_spatial(
    individual_count: NDArray[np.float64],  # (n_demes, n_sexes, n_ages, n_genotypes)
    observation_mask: NDArray[np.float64],  # (n_groups, n_sexes, n_ages, n_genotypes)
    compact: CompactMeta,
) -> NDArray[np.float64]:
    for gi in range(len(compact.offsets)):
        if compact.mode_aggregate[gi]:
            # aggregate: 选中 deme 求和为一个 chunk
            agg = sum(observation_mask[gi] * individual_count[d] for d in selected)
            result[offset:offset+sex_ages] = agg.ravel()
        else:
            for di in range(nd):
                if di < compact.selected_n[gi]:
                    # 选中 deme → 真实数据
                    result[...] = (observation_mask * ind).sum(axis=-1).ravel()
                else:
                    # 未选中 deme → -1.0 sentinel
                    result[...] = -1.0
```

#### Deme 选择：三种模式

`CompactMeta` 内置三种 per-group 模式，由 group spec 中的 `"deme"` 键控制：

| 模式 | spec 格式 | 记录行为 | 导出行为 |
|------|---------|---------|---------|
| `mask`（默认） | `"deme": [0, 2]` 或 `list` | 全 `n_demes` 写入，未选中 = `-1.0` | 未选中显示 `"masked"`，不参与 aggregate |
| `expand` | `"deme": {"demes": [0,2], "mode": "expand"}` | 仅写入选中 deme | 仅展示选中 deme |
| `aggregate` | `"deme": {"demes": [0,2], "mode": "aggregate"}` | 求和为一个 chunk | 单个统计量 |

-1.0 sentinel 确保"deme 真正 0 只"和"被 mask 掉"可区分，避免了旧 `deme_selector` 归零方式的歧义。

#### Codegen 传递路径与模板签名

`_run_codegen_wrapper_steps()` 使用 NamedTuple 束传递到模板：

```python
run_fn(
    ind_all, sperm_all,
    config_bank, deme_config_ids, registry, tick, n_steps,
    adjacency, migration_mode,
    self._spatial_topo,         # SpatialTopology (rows, cols, wrap)
    self._migration_params,     # MigrationParams (kernel, include_center, rate, adjust_on_edge)
    het,                        # HeterogeneousKernelParams | None
    record_interval, observation_mask, compact_meta,
)
```

模板 `RUN_FN_NAME` 签名从原先 35 个参数精简为 17 个：

```python
def RUN_FN_NAME(
    ind, sperm, config_bank, deme_config_ids, registry, tick, n_steps,
    adjacency, migration_mode,
    spatial_topo: SpatialTopology,
    migration: MigrationParams,
    het_kernel: HeterogeneousKernelParams | None,
    record_interval: int,
    observation_mask: Optional[np.ndarray],
    compact_meta: Optional[CompactMeta],
) -> ...:
```

其中 `migration_mode` 和 `adjacency` 不纳入 `MigrationParams`，因为它们属于**路由策略**（由 `_effective_migration_route()` 动态解析），而非**迁移参数**本身。

### 4. Spatial 特有路径（spatial_population.py）

#### record_observation 属性

设置时自动调用 `_build_deme_info()` 解析 `"deme"` spec，再调用 `build_compact_metadata()` 构建 `CompactMeta`：

```python
@record_observation.setter
def record_observation(self, obs: Optional[Observation]) -> None:
    self._observation = obs
    if obs is not None:
        ref_deme = self._demes[0]
        state = ref_deme.state
        self._observation_mask = obs.build_mask(...)
        self._rebuild_compact_meta()   # → _build_deme_info() + build_compact_metadata()
```

`_build_deme_info()` 解析 group spec 中的 `"deme"` 键，支持三种格式：
- 缺失 / `"all"` → 不在 dict 中（默认全 deme）
- `list[int | (row, col)]` → `("mask", flat_indices)`（向后兼容）
- `{"demes": [...], "mode": "aggregate" | "expand" | "mask"}` → dict 格式新语义

#### Python Dispatch 路径

当使用 Python dispatch（hook-aware 回退路径）时，recording 在 Python 层手动打点，复用同一 `build_observation_row_spatial()`：

```python
# run() → _should_use_python_dispatch() → True
if record_every > 0 and (self._tick % record_every == 0):
    self._record_snapshot()
for _ in range(n_steps):
    if self._run_python_dispatch_tick():
        was_stopped = True
        break
    if record_every > 0 and (self._tick % record_every == 0):
        self._record_snapshot()
```

`_record_snapshot()` 在观测模式下调用独立 njit 函数：

```python
if self._observation_mask is not None and self._compact_meta is not None:
    row = build_observation_row_spatial(
        ind_all, self._observation_mask, self._compact_meta,
    )
    flat = np.empty(1 + self._compact_meta.row_size, dtype=np.float64)
    flat[0] = float(self._tick)
    flat[1:] = row
```

### 5. 导出层（state_translation.py）

#### 自动分发逻辑

```python
def output_history(population, observation=None, groups=None, ...):
    pop_obs = getattr(population, "record_observation", None)
    if pop_obs is not None and observation is None and groups is None:
        # 观测模式：直接解析压缩快照
        payload = population_observation_history_to_readable_dict(...)
    else:
        # 原始模式或 post-hoc 观测：按每个快照重新解析
        payload = _build_history_observation_payload(...)
```

Spatial 版本类似：

```python
def spatial_population_output_history(spatial_population, ...):
    obs = getattr(spatial_population, "record_observation", None)
    if obs is not None:
        payload = spatial_population_observation_history_to_readable_dict(...)
    else:
        payload = spatial_population_history_to_readable_dict(...)
```

#### population_observation_history_to_readable_dict

这个函数解析观测模式压缩后的历史数组。流程：
1. 获取 `record_observation` 的 labels 和 collapse_age
2. 对每一行 `[tick, observed.ravel()]`：
   - 按 `(n_groups, n_sexes, n_ages)` 做 reshape
   - 用 `_build_observation_payload()` 将 observed 数组转为 `{sex: {age: {label: count}}}` 嵌套字典
3. 返回带 labels 和 snapshots 的结构

#### spatial_population_observation_history_to_readable_dict

类似 panmictic 版本，但多了一个 deme 维度：
1. 对每一行 reshape 为 `(n_demes, n_groups, n_sexes, n_ages)`
2. 按 deme 展开 per-deme payload
3. 跨 deme 求和得到 aggregate

#### _build_observation_payload 工具函数

```python
def _build_observation_payload(observed, labels, sex_labels, include_zero_counts):
    """将 observed 数组转为嵌套字典 {sex: {age: {label: count}}}"""
```

这个函数是所有观测导出路径的公共工具，将数字数组转为人类可读的嵌套字典。

#### 回退机制

观测历史的导出函数（`population_observation_history_to_readable_dict`、`spatial_population_observation_history_to_readable_dict`）都包含回退逻辑：

```python
obs = getattr(population, "record_observation", None)
if obs is None:
    return population_history_to_readable_dict(population, ...)
```

这意味着即使在没有 observation 的旧数据上调用这些函数也不会崩溃——它们会回退到原始历史解析路径。

### 6. Post-hoc 观测路径

post-hoc 观测（不修改 recording 模式，仅在导出时应用观测）通过 `_build_history_observation_payload` 实现：

```python
def _build_history_observation_payload(population, history, observation, groups, collapse_age, ...):
    # 对历史数组的每一行：
    # 1. 解析出 tick、individual_count、sperm_storage
    # 2. 用 observation 或临时构建的 Observation 做 apply
    # 3. 组装为观测格式的 payload
```

这个路径的性能开销较大——需要对每个历史快照重建 state 并重新 apply observation。适用于历史较短或事后需要不同分组视角的场景。

## 关键设计决策

### 1. 统一 _history 存储

无论原始还是观测模式，`_history` 始终是 `List[Tuple[int, NDArray[np.float64]]]`。区别仅在于 array 的内容（原始扁平 state vs 观测聚合数组）。这样 `get_history()` 的接口保持一致。

### 2. 4D Mask 始终完整

`Observation.build_mask()` 始终返回 4D mask `(n_groups, n_sexes, n_ages, n_genotypes)`，即使是 discrete generation（年龄=1）或 `collapse_age=True` 的场景。`collapse_age` 仅作为 metadata 存储在 Observation 中，在导出时被 `_build_observation_payload` 等函数读取。

原因是 kernel 需要统一的 memory layout——在 Numba njit 函数中根据 `collapse_age` 动态切换维度会导致类型不稳定。

### 3. Kernel Observation 只做聚合，不做选择

Observation mask 在 kernel 内部只用于 genotype 维度聚合（`sum(axis=-1)`），不用于筛选 deme 或 sex。Deme 筛选由 `deme_selector` 完成（spatial 特有），sex 和 age 轴不做压缩——`observed` 数组保留 `(n_sexes, n_ages)` 维度。

### 4. Spatial 的 _observation_mask 共享

所有 deme 共享同一个 `_observation_mask`（因为 genotype 数量和基因型名称在所有 deme 中一致）。只有 `deme_selector` 是 per-deme 的。

## 验证要点

修改 observation recording 相关代码时，需要确保以下行为不变：

1. **无 observation 时记录原始数据**：`record_observation = None` 时，`run()` 的行为与之前完全一致
2. **观测模式的回退兼容**：观测模式的历史数据可以用 `output_history()` 正确导出
3. **Post-hoc 观测正确性**：`output_history(observation=obs)` 在原始历史和观测历史两种模式下都返回一致的结果
4. **Spatial 聚合验证**：观测模式的 spatial aggregate 等于所有 deme 按分组求和的结果
5. **Clone 兼容**：`SpatialBuilder` 克隆后的 deme 能独立设置 observation
6. **Python dispatch 路径**：`_should_use_python_dispatch()` 回退路径下的 recording 同样正确

测试命令：
```bash
pytest                                       # 所有测试
pyright src/natal/                            # 类型检查
ruff check src/natal/                         # lint
```
