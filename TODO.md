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

### 4. 灵活化 embryo resistance rate 配置

- 未必是定值，可与亲本中 Cas9 copies（或表达时间）有关
- 可支持 heterozygotes / homozygotes 不同配置

### 5. Spatial API 其他优化

- 优化初始化 deme builder 方式（`batch_setting(…)`）
- 批量设置 local hooks（使用 `deme_selector`）
- 优化 migration kernel，处理边界效应（总迁移率不应不变，而应正比于邻居数量；或可不用总迁移率设置，尝试全部设为 1；需要一个优雅的方法）

### 6. Spatial UI 问题

- 目前 square 易卡死 → 格点数太多时与 hex 一样渲染成热图
- 支持选 deme 时，显示和 panmictic 一样的 config 信息
- 支持显示所有 local hooks
- 支持 landscape 显示 genotype freq per deme 等指标

### 7. General UI 问题

- 需与 `Observation` 集成
- 支持 UI 导出集成后的 history 观测数据

### 8. Spatial History

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
simulation_kernels.py (line 382)
finish 是 Python 层触发（run 结束或 finish_simulation()），不在 kernel 事件链里。
age_structured_population.py (line 878)
discrete_generation_population.py (line 233)
base_population.py (line 801)
initialization 目前也在 Python 事件体系里，不在 kernel 执行路径。
```
