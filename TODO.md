# TODO

## v0.2.0

### 1. 改 `late_..._resistance` 为 `absolute_resistance`

- 增加快捷设置方式，不删原有参数
- $d+r>1$ → 报错

### 2. 确认 `expected_num_adult_females` 是否正常工作

- 是否正确从种群初始状态推断
- 如何与 `age_1_carrying_capacity` 协同工作

### 3. Spatial 并行问题

- 如果有 Hook，目前是 Python dispatch，即 migration 前的 per-deme simulation 没有真正并行
- 解决：只要 Hook 可编译，都走 njit 路径。每个 deme 只调 `run_tick`，不分步调用 panmictic `run_xxx`

### 4. 混用 CSR + njit Hook 时的 priority 跨类型比较与性能矛盾

- `has_mixed_hook_types()` 的意图不是"保守"，而是**保护 priority 语义的正确性**：
  - 内核模板中 CSR batch 先于 njit batch 执行，即使 njit hook 的 priority 更低，也总是排在所有 CSR 之后
  - HookExecutor 统一按 priority 排序所有类型 → 正确，但代价是回到 Python dispatch
  - `has_mixed_hook_types()` 检测到混用后强制回退 HookExecutor，牺牲性能换取正确性
- **panmictic 模型**：`should_use_python_dispatch()` 检查 `has_mixed_hook_types()` → 混用时回退（牺牲性能保 priority 正确）
- **spatial 模型**：`_should_use_python_dispatch()` **不**检查混合类型 → 混用时走 Numba 路径能并行，但 **priority 跨类型比较是错误的**
- 根本原因：内核模板中 CSR（`execute_csr_event_program_with_state`）和 custom（合并后 `_*_HOOK`）是两个硬编码的先后阶段，无法按 priority 交错
- 解决方向：把 CSR plan 执行嵌入 `compile_combined_hook`，在 Numba JIT 内部按 priority 生成统一 dispatch 序列，同时解决正确性和性能问题

### 5. Observation 录制逻辑在模板和 Python 路径中重复

- `RUN_FN_NAME`（4 个模板）中手工构造 `flat_state` + `observation_mask` 聚合 → Numba 内核路径
- `_run_python_dispatch`（2 个模型）中调用 `create_history_snapshot()` → Python 回退路径
- `_process_kernel_history` 中将内核 raw array 转为 History 对象 → 后处理
- 三种路径的 flatten 格式因生命周期类型不同（discrete / structured / spatial compact），但录制时机和条件判断逻辑相同
- 改善方向：将 `flatten_size` 计算和 `flat_state` 填充抽成 `@njit` 辅助函数，按生命周期类型参数化

### 6. 灵活化 embryo resistance rate 配置

- 未必是定值，可与亲本中 Cas9 copies（或表达时间）有关
- 可支持 heterozygotes / homozygotes 不同配置

### 7. Preset 系统声明式重构——增量重算

**当前问题**：每个 preset 通过黑盒回调 `fn(tensor) → tensor` 修改张量。系统不知道回调改了哪些行，
因此改任何 preset 参数都触发全量重建：
```
清除所有 modifier → 重新按顺序应用所有 preset → 重算整个 offspring_tensor
```
`reconfigure_preset(A, param=0.3)` 会白跑 B 和 C 的修饰器，并重算所有 n² 个 genotype pair。

**目标架构**：声明式规则 + 层叠式应用 + 行级所有权追踪。

**1. 行级所有权追踪**：每个 preset 注册时预计算它修改的 genotype 行：
```python
class DeclarativeModifier:
    rules: list[Rule]
    owned_rows: set[int]   # 从规则声明预计算，注册时确定

# 例如 HomingDrive(allele="X", rate=0.9) → owned_rows = {0, 2, 7}
```

**2. 层叠式应用**：每个 preset 从 Mendelian baseline 读取数据计算自己的变换，
而非依赖前一个 preset 的输出。这消除了顺序依赖，使交换律在非冲突场景下成立：
```python
def rebuild_maps(registry):
    tensor = species_baseline.copy()
    for preset in registry:
        for row in preset.owned_rows:
            tensor[row] = preset.apply_rule(row, species_baseline[row])
    return tensor
```

**3. 增量重算**：`reconfigure_preset(A)` 只重算 A 触及的行。B 和 C 的数据
直接从 baseline 保留：
```python
def reconfigure_preset(registry, changed):
    tensor = species_baseline.copy()
    for preset in registry:
        if preset == changed: continue   # 跳过旧的
        for row in preset.owned_rows:
            tensor[row] = preset.apply_rule(row, species_baseline[row])
    changed.apply_new_params(...)
    for row in changed.owned_rows:
        tensor[row] = changed.apply_rule(row, species_baseline[row])
```

**4. 增量 offspring_tensor**：只重算母亲或父亲属于 affected_rows 的 pair：
```
全量: O(n²) 个 pair
增量: O(|affected| × n) 个 pair
```
对 n=100 基因型，A 只触及 2 个：200 个 pair vs 10000 个 pair。

**5. 冲突检测**：两个 preset 声明同一行时注册期即可见，当场决定策略
（后注册覆盖 / 报错 / 合并规则）。

**6. 与现有指令式模型的对比**：

| | 指令式（当前） | 声明式（目标） |
|---|---|---|
| 修饰器 | 黑盒回调 `fn(tensor)→tensor` | 声明式规则 + owned_rows |
| 交换律 | 不满足（退化情况除外） | 不冲突时天然满足 |
| 冲突处理 | 运行时后写覆盖（不可见） | 注册期可检测 |
| reconfigure | O(全部重算) | O(受影响的行 + 部分 tensor) |
| 可并行性 | 无（顺序依赖） | 有（各 preset 独立从 baseline 读取） |

### 8. Spatial API 其他优化

- 优化初始化 deme builder 方式（`batch_setting(…)`）
- 批量设置 local hooks（使用 `deme_selector`）
- 优化 migration kernel，处理边界效应（总迁移率不应不变，而应正比于邻居数量；或可不用总迁移率设置，尝试全部设为 1；需要一个优雅的方法）

### 9. Spatial UI 问题

- 目前 square 易卡死 → 格点数太多时与 hex 一样渲染成热图
- 支持选 deme 时，显示和 panmictic 一样的 config 信息
- 支持显示所有 local hooks
- 支持 landscape 显示 genotype freq per deme 等指标

### 10. General UI 问题

- 需与 `Observation` 集成
- 支持 UI 导出集成后的 history 观测数据

### 11. Spatial History

- 保存每个 deme 的 History 数据，提供快捷解析和导出方法
- 支持 UI 导出
- 支持刷新后加载历史数据

## v0.3.0 及远期更新

- Global hooks
- Sparse（import / states）

## initialization / finish 现状

```txt
事件定义里仍有 initialization、finish（以及 first/early/late）。
base_population.py (line 51)
types.py (line 124)
kernel 加速路径目前只执行 first/early/late（CSR+chain）。
simulator.py (line 382)
finish 是 Python 层触发（run 结束或 finish_simulation()），不在 kernel 事件链里。
age_structured_population.py (line 878)
discrete_generation_population.py (line 233)
base_population.py (line 801)
initialization 目前也在 Python 事件体系里，不在 kernel 执行路径。
```
