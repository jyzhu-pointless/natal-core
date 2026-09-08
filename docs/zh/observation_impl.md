# Observation 与 History 实现解析

本文档面向 NATAL Core 的维护者和贡献者，说明 canonical Observation、类型化 History 以及空间记录路径之间的职责边界。公开用法参见 [提取种群模拟数据](2_data_output.md)。

## 实现概览

Observation 只定义“怎样从种群状态得到观测结果”，History 只定义“保存哪一种快照”。两者在构建阶段由 RecordingPlan 连接，但仍是独立概念：

```text
Configurator.with_observation(...)
  → 编译不可变的 canonical Observation

Configurator.record_history(mode=...)
  → 选择 raw 或 observation History schema

Population state
  ├─ pop.observe() → ObservationResult
  └─ sampling boundary → History
       ├─ raw mode: 保存完整状态
       └─ observation mode: 保存 canonical Observation 的投影结果
```

核心模块及职责：

| 模块 | 职责 |
|------|------|
| `output/observation.py` | 定义 `Observation`、`ObservationResult`、`ObservationFilter` 与恒等观测 |
| `output/history.py` | 定义不可变 schema、类型化数组视图、raw History 的事后投影 |
| `output/_recording.py` | 在构建阶段编译 `RecordingPlan`、行宽与空间布局 |
| `rust/src/output/history.rs` | 拥有历史环形存储、行数据、保留预算及共享历史句柄 |
| `rust/src/output/observation.rs` | 拥有数值投影及当前状态观测入口 |
| `rust/src/output/parameter_log.rs` | 拥有参数日志及日志值转换 |
| `rust/src/sessions/spatial.rs` | 批量运行空间生命周期，在 Rust 内记录边界 |
| `spatial/population.py` | 传递运行控制与 selector，包装 Rust 观测查询结果 |

## 构建阶段的公开接口

非空间 Configurator 使用：

```text
.with_observation(groups, collapse_age=False)
```

空间 Configurator 额外接受 deme 选择与处理方式：

```text
.with_observation(
    groups,
    collapse_age=False,
    demes=None,
    deme_mode="preserve",
)
```

`groups` 是从非空标签到 `IndividualSelector` 的有序映射。其插入顺序成为结果的 group 轴顺序。

空间参数的约束如下：

| 参数 | 语义 |
|------|------|
| `demes=None` | 按 Population 顺序选择全部 deme |
| `demes=[2, 0]` | 选择 deme 2 和 0，并保留这个顺序 |
| `deme_mode="preserve"` | 保留一个共享的 deme 轴 |
| `deme_mode="aggregate"` | 对选中的 deme 求和并移除 deme 轴 |

`demes` 必须是非空、无重复且位于 Population 范围内的整数序列。所有 group 共享同一个有序 deme 选择；不能为不同 group 指定不同的 deme 集合。`deme_mode` 只接受 `"preserve"` 和 `"aggregate"`。

Observation 和 History schema 在 `build()` 时冻结。运行时 Configurator 不允许更换 `with_observation()` 或 `record_history()` 规则。

## Canonical Observation

每个 Population 构建后都有一个 canonical `Observation`。如果用户没有调用 `with_observation()`，构建过程会生成恒等观测：每个活跃 ZType 对应一个 group。恒等观测不会创建二次方大小的 dense mask，而是使用 ZType 索引映射完成无损轴变换。

`Observation` 保存以下稳定信息：

- `labels`：group 轴标签。
- `collapse_age`：是否沿 age 轴求和并移除该轴。
- `population_fingerprint`：用于阻止把规则应用到不匹配的 Population 布局。
- `deme_indices`：空间观测的有序 deme 选择；非空间观测为 `None`。
- `deme_mode`：空间观测保留还是聚合 deme 轴。
- 编译后的 selector mask，或恒等观测的 ZType 索引映射。

### 数值投影

年龄结构的非空间输入形状是：

```text
(sex, age, ztype)
```

普通观测通过以下运算得到 group-first 结果：

```text
mask[group, sex, age, ztype]
  × count[sex, age, ztype]
  → 沿 ztype 求和
  → result[group, sex, age]
```

空间输入在最前面增加规则化 deme 轴：

```text
(deme, sex, age, ztype)
```

空间投影先按 `deme_indices` 切片，再对每个选中 deme 应用同一组 selector mask。`preserve` 保留切片后的 deme 轴；`aggregate` 对该轴显式求和。最后，`collapse_age=True` 会对 age 轴求和并移除它。

这套顺序保证两个关键不变量：

```text
preserve 结果 == 对原始 count 按有序 demes 直接切片后逐 deme 投影
aggregate 结果 == preserve 结果沿 deme 轴求和
```

### 公开结果轴

`pop.observe()` 返回 `ObservationResult`，其中 `values` 的轴由 `axes` 明确描述：

| 模式 | `collapse_age=False` | `collapse_age=True` |
|------|----------------------|---------------------|
| 非空间 | `(group, sex, age)` | `(group, sex)` |
| 空间 preserve | `(group, deme, sex, age)` | `(group, deme, sex)` |
| 空间 aggregate | `(group, sex, age)` | `(group, sex)` |

`labels["group"]` 与 group 轴一一对应。preserve 模式下的 deme 轴顺序严格等于 `demes` 参数的顺序。

## History 与 RecordingPlan

`record_history()` 独立选择记录模式：

```text
.record_history(mode="raw", max_rows=None)
.record_history(mode="observation", max_rows=None)
```

`compile_recording_plan()` 在构建时创建不可变的 `HistorySchema`，并据 Population 布局、Observation 轴及 History mode 计算固定 `row_size`。空间 schema 额外保存完整 deme 数量与每个 deme 的 raw payload 宽度。

### Raw mode

空间 raw History 始终保存所有 deme 的完整状态，不受 Observation 中 `demes`、`deme_mode` 或 `collapse_age` 影响：

| 数据 | 形状 |
|------|------|
| `history.individual_count` | `(record, deme, sex, age, ztype)` |
| `history.sperm_storage`（年龄结构） | `(record, deme, age, female_ztype, male_ztype)` |

因此，raw History 可在事后调用 `history.observe(observation)`，生成独立的 observation-mode History，原 History 保持不变。

### Observation mode

Observation History 只保存 canonical Observation 的数值结果：

| 模式 | `collapse_age=False` | `collapse_age=True` |
|------|----------------------|---------------------|
| 非空间 | `(record, group, sex, age)` | `(record, group, sex)` |
| 空间 preserve | `(record, group, deme, sex, age)` | `(record, group, deme, sex)` |
| 空间 aggregate | `(record, group, sex, age)` | `(record, group, sex)` |

schema 中的 `ObservationMetadata` 保存 group 标签、年龄折叠状态、有序 deme 选择与 deme mode，因此读取 `history.values` 不需要依赖当前 Population 的可变外部状态。

Observation History 已经丢弃未记录的 ZType 与未选择的 deme 信息，不能再换一套 Observation 做事后投影。

## 空间记录路径

空间会话在 Rust 内运行生命周期与迁移，在记录边界直接写入 Rust HistoryStore：raw 模式保存完整状态，observation 模式使用已编译的 selector 投影后仅保存观测值。

```text
Rust spatial session
  → lifecycle and migration
  → raw state or native observation projection
  → bounded HistoryStore and raw checkpoints
  → Python query: independent result arrays
```

当前状态观测、raw 历史的事后投影与运行时观测记录共享同一套 Rust 数值实现。Python 编译 selector、维护标签并提供只读查询出口；无 Python 回调的批量运行不逐 tick 返回 Python，也不往返传输完整状态。手动快照同样直接从会话记录。`max_rows` 在逐条写入时淘汰旧记录及其检查点，避免先积累整个运行批次。

## 已删除的 compact 空间布局

空间 Observation 不再支持按 group 定义不同的 deme 布局。旧实现中的以下概念已从空间记录路径删除：

- `CompactMeta` 与 `build_compact_metadata()`。
- `build_observation_row_spatial()`。
- per-group `mask` / `expand` / `aggregate` 布局。
- 用 `-1.0` 表示未选择 deme 的 sentinel。
- ragged group offsets 与每组不同的 row width。

现在所有 group 共用同一个规则 ndarray 轴结构。未选择的 deme 不会写入 preserve 结果，也不需要特殊数值标记；真正的零计数仍然就是 `0.0`。需要不同 deme 视角时，可以从 raw History 分别构造多套 Observation，或先记录更宽的统一选择再由调用方分离。

## 维护不变量

修改 Observation 或 History 记录路径时，至少应验证以下数值关系：

1. 空间 preserve 与按 `demes` 顺序直接切片后的逐 deme 投影逐元素相等。
2. 空间 aggregate 与 preserve 结果沿 deme 轴求和逐元素相等。
3. 空间恒等 Observation 在选中 deme 上不丢失任何坐标值。
4. `collapse_age=True` 与未折叠结果沿 age 轴求和逐元素相等。
5. raw History 保留所有 deme、ZType 及适用的 sperm storage。
6. raw History 的事后投影与同 tick 的 `Observation.apply()` 逐元素相等。
7. 引擎内批次与带外快照在确定性模拟中产生相同 ticks 和相同 payload。
8. 未选择的 deme 不依赖 sentinel 表示，真实零计数不会与选择状态混淆。

断言应比较明确的轴和逐坐标值；只比较总和或排序后的扁平数组无法发现轴交换和 deme 顺序错误。
