# Observation 历史记录：开发者实现文档

本文档面向 NATAL Core 的维护者和贡献者，详细说明 observation-based history recording 的实现原理、数据流和各模块职责。

## 整体架构

Observation 历史记录涉及四个层次：

```
用户层 API          →  BasePopulation / SpatialPopulation 的 record_observation 属性
Observation 层     →  observation.py: Observation, ObservationFilter, build_mask
Kernel 层          →  simulation_kernels.py / spatial_simulation_kernels.py 中的 recording 通路
导出层             →  state_translation.py: output_history / *observation_history_to_readable_dict
```

## 数据流

```
用户定义 groups
       ↓
ObservationFilter.build_filter(groups) → Observation 对象（含 specs、labels、collapse_age）
       ↓
BasePopulation._build_observation_mask(obs) → 4D float64 mask (n_groups, n_sexes, n_ages, n_genotypes)
       ↓
mask 传入 Numba kernel（observation_mask 参数）
       ↓
kernel 每个 record_every 步：
  observed = sum(mask[None, :, :, :, :] * ind[:, None, :, :, :], axis=-1)
  row = [tick, observed.ravel()]
       ↓
Python 层：history.append((tick, row.copy()))  →  _process_kernel_history()
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

`run_simulation` 函数的 `record_single_tick()` 内部：

```python
if observation_mask is not None:
    observed = np.sum(observation_mask * ind_count[None, :, :, :], axis=-1)
    row[1:] = observed.ravel()
else:
    row[1:1+ind_size] = ind_count.ravel()
    row[1+ind_size:] = sperm_storage.ravel()  # 如有
```

关键参数：
- `observation_mask: Optional[NDArray[np.float64]]` — 4D 掩码
- `n_obs_groups: int` — 观测分组数，决定 row 大小

#### Spatial 内核（spatial_simulation_kernels.py）

`spatial_run_simulation` 的 `record_single_tick()` 内部：

```python
if observation_mask is not None:
    observed = np.sum(observation_mask[None, :, :, :, :] * ind[:, None, :, :, :], axis=-1)
    if deme_selector is not None:
        observed = observed * deme_selector.T[:, :, None, None]
    row[1:] = observed.ravel()
else:
    # 拼接所有 deme 的 ind 和 sperm
```

注意 spatial 的 broadcast 多了一个 `n_demes` 维度：
- `ind`: `(n_demes, n_sexes, n_ages, n_genotypes)`
- `mask`: `(n_groups, n_sexes, n_ages, n_genotypes)`
- `mask[None, :, :, :, :] * ind[:, None, :, :, :]` → `(n_demes, n_groups, n_sexes, n_ages, n_genotypes)`
- `sum(axis=-1)` → `(n_demes, n_groups, n_sexes, n_ages)`

#### Deme Selector（spatial 特有）

`deme_selector: Optional[NDArray[np.float64]]` 的 shape 为 `(n_groups, n_demes)`。其作用是在观测聚合后对 deme 维度做逐分组的掩码：
- 值为 1.0 表示该 deme 计入当前分组
- 值为 0.0 表示该 deme 被排除

实现：`observed = observed * deme_selector.T[:, :, None, None]` — 利用 broadcast 在 n_demes 维度上施加掩码。

`_build_deme_selector()` 从 group spec 中的 `"deme"` 键构建该数组:
- `"deme": "all"` 或省略 → selector 全 1.0
- `"deme": [0, 2, 4]` → 对应列设为 1.0
- `"deme": [(1, 1)]` → 通过 topology 解析坐标后设为 1.0

当所有分组都是 `"all"` 时，`deme_selector` 为 `None`（kernel 跳过乘法以优化性能）。

#### Codegen 传递路径

`_run_codegen_wrapper_steps()` 将 observation 参数传递给 wrapper：

```python
obs_mask = self._observation_mask
n_obs = len(self._observation.labels) if self._observation is not None else 0
demean_sel = self._deme_selector
final_state_tuple, history_new, was_stopped = run_fn(
    ...,
    record_interval=int(record_every),
    observation_mask=obs_mask,
    n_obs_groups=n_obs,
    deme_selector=demean_sel,
)
```

### 4. Spatial 特有路径（spatial_population.py）

#### record_observation 属性

与 panmictic 类似，但 `_build_observation_mask` 通过第一个 deme 的 state shape 来编译 mask（所有 deme 的 genotype 数量和维度一致）：

```python
@record_observation.setter
def record_observation(self, obs: Optional[Observation]) -> None:
    self._observation = obs
    if obs is not None:
        ref_deme = self._demes[0]
        state = ref_deme.state
        self._observation_mask = obs.build_mask(
            n_sexes=state.individual_count.shape[0],
            n_ages=state.individual_count.shape[1] if state.individual_count.ndim == 3 else 1,
            n_genotypes=state.individual_count.shape[-1],
        )
        self._deme_selector = self._build_deme_selector()
```

#### Python Dispatch 路径

当使用 Python dispatch（hook-aware 回退路径）时，recording 在 Python 层手动打点：

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

`_record_snapshot()` 在观测模式下用广播乘法聚合：

```python
if self._observation_mask is not None:
    observed = np.sum(
        self._observation_mask[None, :, :, :, :] * ind_all[:, None, :, :, :],
        axis=-1,
    )
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
