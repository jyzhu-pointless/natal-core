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

### 7. Spatial API 其他优化

- 优化初始化 deme builder 方式（`batch_setting(…)`）
- 批量设置 local hooks（使用 `deme_selector`）
- 优化 migration kernel，处理边界效应（总迁移率不应不变，而应正比于邻居数量；或可不用总迁移率设置，尝试全部设为 1；需要一个优雅的方法）

### 8. Spatial UI 问题

- 目前 square 易卡死 → 格点数太多时与 hex 一样渲染成热图
- 支持选 deme 时，显示和 panmictic 一样的 config 信息
- 支持显示所有 local hooks
- 支持 landscape 显示 genotype freq per deme 等指标

### 9. General UI 问题

- 需与 `Observation` 集成
- 支持 UI 导出集成后的 history 观测数据

### 10. Spatial History

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
