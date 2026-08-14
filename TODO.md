# TODO

> 最后审计：2026-08-15。已完成事项迁入本地 `TODO.legacy.md`；本文件只保留未完成或部分完成的工作。
>
> 排序逻辑：正确性 bug > 性能优化 > UX 改进 > 代码质量。同一档内，部分完成 > 未开始 > 仅设计。
>
> 状态标记：
> - ✅ DONE — 已实现
> - ⚠️ PARTIAL — 部分实现，有遗留问题
> - 📋 NOT_DONE — 未实现
> - 🎨 DESIGN_ONLY — 仅有设计方案，无实现

---

## History / Observation 重构协调与延期设计（2026-07-15）

> 本节记录此前 grill session 中已经讨论、但明确不应随当前增量重构一并实现的设计。当前重构只实现构建期 canonical Observation、单模式 History、post-hoc observation、`record_snapshot()` 和 raw checkpoint restore。

### HO-C1 📋 natal-inferencer 接口协调

natal-core 的最终接口稳定后，在 `natal-inferencer` 单独实施：

- 将 `population.record_observation` 替换为只读 `population.observation`。
- 粒子数组投影统一使用 `population.observation.apply(particle_counts)`。
- 接受 Population 自动提供 identity Observation 的默认行为。
- 删除对 `pop.create_observation()`、旧 output helpers 和兼容 alias 的依赖。
- 增加跨仓库集成测试，覆盖默认 identity 与显式 Observation。

natal-core 不为此保留 `record_observation` shim；两个项目尚未发布，可以直接协调升级。

### HO-C2 📋 Observation rule 匹配结果聚合语义

`natal-inferencer` 已提出：一条显式 observation rule 匹配到多个项时，对外应返回这些匹配项的总和，而不是把每个匹配项分别返回。

**当前暂不修改。** `natal-inferencer` 仍依赖现有的分项结果结构；单独修改 natal-core 会破坏其输入形状、标签或索引约定。该变更必须与 inferencer 迁移协调完成，不能作为 core 内部的独立修复。

后续实施时必须满足：

- 聚合边界是一条具名 observation rule；每条 rule 只产生一个对应的聚合结果。
- 聚合值在数值上等于该 rule 所有匹配项的显式求和，不能漏计或重复计数。
- 多条 rule 分别独立聚合；同一项同时匹配多条 rule 时，应分别计入各自结果。
- 默认 identity Observation 的逐 ZType 返回语义不随本项自动改变，除非另行评审。
- natal-core、`natal-inferencer` 及跨仓库集成测试应在同一次兼容性迁移中更新。

### HO-D1 🎨 Hook 条件触发与 tick 内记录

**不纳入当前重构。** 当前只增加引擎空闲时调用的 `pop.record_snapshot()`，记录完整 tick 边界。

未来如果允许 Hook 触发记录，必须先解决：

- Hook 运行在 Numba / Python 双路径中，不能直接调用 Python Population 方法。
- `first`、`early`、`late` 对应不同生命周期阶段，单独使用 tick 无法唯一标识记录。
- `early` 状态已完成繁殖但尚未完成存活和年龄推进；`late` 状态尚未完成年龄推进。这些状态不是普通 checkpoint，不能从标准 tick 入口恢复。
- 空间模型还必须明确记录发生在 per-deme 生命周期、全局迁移之前还是之后，并保证跨 deme 一致性。
- 可能需要 `(tick, phase, occurrence)` 身份、预分配 Numba buffer 和独立的 trace schema。
- “记录规则”应编译成引擎可执行的信号或条件程序，而不是让 Hook 修改 History 容器。

设计时应优先判断它是否应成为独立的 Trace / Event Record 系统，而不是继续扩张可恢复 History。

### HO-D2 🎨 可恢复的条件中止

**不纳入当前重构，也不增加 `resume()`。** 当前 `RESULT_STOP` 同时承担中止 Hook event、提前退出 `run()` 和永久 finish Population 三种语义；`stop_if_*` 触发后会设置 `is_finished=True`，无法安全继续。

不能简单清除 `_finished`：

- 在 `first` 停止虽然位于 tick 边界，但同一条件可能在下一次 `run()` 立即再次触发。
- 在 `early` / `late` 停止时状态位于 tick 中间；从 `first` 重新进入会重复生命周期步骤并破坏数值语义。
- finish Hook 已可能执行，重新开放 Population 会违反终止不变量。

后续设计应拆开：

- `finish`：永久结束，触发 finish Hook，不可继续。
- `break` / `pause`：只让当前 `run()` 在完整 tick 边界返回，Population 仍可继续。
- tick 内 abort：保留为生命周期控制，不伪装成可恢复暂停。

可能需要独立 `RunResult` / stop reason 和 tick-boundary condition compiler。当前安全替代方案是在 Python 层逐 tick `run(n_steps=1, record_every=0)`，检查 `pop.observe()` 后调用 `pop.record_snapshot()`。

### HO-D3 🎨 多 Observation 与运行时规则编译

当前只支持一个由 Configurator 在构建期确定的 canonical Observation，不公开 `pop.create_observation()` 或 `pop.observe(other_observation)`。

当前可用替代方式：

- 在 canonical Observation 中声明多个具名 group，再手动拆分结果。
- 使用 raw History 做自定义分析。

只有出现一份 Population 必须维护多套可复用 Observation 的真实需求后，才设计独立于 Population 的 rule compiler；不得通过恢复 runtime setter 解决。

### HO-D4 🎨 History 持久化存档

当前 `export_state()` 只导出当前 Population 状态，History 不随状态导入导出。`restore_checkpoint()` 只使用当前 Population 内存中的 raw History。

未来如需跨进程或长期存档，应单独设计 `History.save()` / `History.load()`：

- 文件必须保存完整 immutable schema、Population layout fingerprint、labels、axes、mode 和版本。
- raw 与 observation History 都应可往返，但只有 raw History 可以恢复 Population。
- 不应重新暴露缺少 schema 的 flat ndarray 文件格式。
- 需要明确版本迁移、压缩、分块读取和大规模空间 History 的存储策略。

---

## 本地工具与排除工作后续

### TOOL-D1 📋 放行 adversarial-review skill

当前 `.opencode/skills/adversarial-review/SKILL.md` 仅存在于本地，并被
`.gitignore` 的 `.opencode/*` 规则排除。当前机器可以执行该审查流程，但新
clone 无法从仓库恢复。后续如需让审查流程自包含，应只放行并跟踪该 skill，
其余 `.opencode` 内容继续忽略。

### TOOL-D2 📋 重新评审 cluster benchmark 工作

`benchmarks/mgdrive1/cluster/` 与
`tests/test_northstar_cluster_orchestration.py` 当前按决定排除，不属于已跟踪
benchmark 或门禁范围。后续只有在 cluster 调度实现准备纳入仓库时，才移除
对应 ignore，并连同可复现环境、测试和运行说明一起评审。

---

## Spatial Runtime Update 重构 — 延期决策（2026-07-18）

> 本节记录此前重构审查中明确**不应随当前增量一并实现**的设计决策。当前重构维持现状（离散代保留 sync、CONCAVE 模式照旧消费 `expected_*`），以下三项留待后续单独评审。

### SU-D1 🎨 离散代竞争语义收敛（FIXED-only vs 全模式）

**不纳入当前重构。** 当前重构维持现状：离散代 CONCAVE/LOGISTIC 模式消费 `expected_competition_strength` 与 `expected_survival_rate` 两个由 `compute_equilibrium_metrics` 从 K/eggs_per_female/sex_ratio/存活/交配/繁殖率推导的均衡校准常数；FIXED/NO_COMPETITION 只读 K。三个离散 demo（`discrete.py`、`discrete_ui.py`、`spatial_hex_discrete.py`）和测试辅助均用 `"concave"`，落入 Beverton-Holt 分支（`discrete_generation_simulator.py:108-114`），消费 `expected_*`。

未来若要收敛为 FIXED-only，必须先解决：

- 入口需拒绝 CONCAVE/LOGISTIC 模式（`DiscreteConfigurator.competition()` 校验 `juvenile_growth_mode ∈ {NO_COMPETITION, FIXED}`）。
- `set_param` 对离散代跳过自动 sync 的现状（`_base.py:247`）从"由 Configurator 方法层兜底"变为"永不 sync"，需删除 `DiscreteConfigurator.competition()/reproduction()` 末尾的 `self._sync_equilibrium()` 调用。
- 三个 demo + 测试辅助的 `juvenile_growth_mode="concave"` 必须迁移到 `"fixed"`——**这是模型语义改动**：调节曲线从平滑 Beverton-Holt（`r/(ratio·(r−1)+1) × expected_surv`）变为硬截断（`min(1, K/N₀)`），过渡动态和低密度增长行为都不同。需单独评审是否可接受。
- `compute_equilibrium_metrics` 的离散分支（`_base.py:1217-1228` 手工组装 survival/mating 数组）是否仍有其他消费方需审计。

设计时应优先判断"离散代是否应彻底移除 `expected_*` 字段"（连同 `age_based_relative_competition_strength`，见 SU-D3），而非仅切换模式。

### SU-D2 🎨 `competition_strength` 在 `new_adult_age=1` 下静默 no-op

**不纳入当前重构。** `parameters.jsonc:39` 的 `competition_strength` 参数写入 `age_based_relative_competition_strength[1]`（`config_path=[1]`）。当 `new_adult_age=1` 时：

- 期望侧：`compute_equilibrium_metrics` 的求和循环 `range(1, new_adult_age)` = `range(1, 1)` 为空，index 1 永不入算；
- 实际侧：`compute_actual_competition_strength` 只加权 `age < new_adult_age`（即只到 age 0），index 1 同样不入算。

结果：`pop.update().competition(competition_strength=2.0)` 在离散代（恒 `new_adult_age=1`）和任何 `new_adult_age=1` 的年龄结构配置下都是**静默 no-op**——写入成功、不报错、零效果。与 F5（hooks 静默 no-op）同类缺陷。

后续设计应：

- 离散代入口对 `competition_strength` 显式拒绝（`ValueError`，提示该参数仅多龄幼虫 `new_adult_age≥2` 有效）；
- 文档注明 `competition_strength` 实际语义是"第 1 龄（第二个年龄）幼虫的相对竞争权重"，只在 `new_adult_age≥2` 时有意义；
- 考虑是否提供 age-0 权重的合法调节入口（目前 rel[0] 由 `np.ones` 默认固定为 1.0，无用户可调路径）。

### SU-D3 🎨 离散代 `age_based_relative_competition_strength` 仅为兼容层

**不纳入当前重构。** 离散代 `new_adult_age=1`，只有 age-0 幼虫参与竞争（成体每 tick 全部替换，不进入密度调节），所以数学上只有一个竞争权重有意义——`rel[0]`。且 `rel[0]` 必须 = 1，否则均衡点偏移到 `rel[0]×K`（ratio=1 时招募数 = `produced_age_0 × expected_surv × s0 = rel[0]×K`），K 失去"承载力"含义。

实际侧引擎根本不做加权（`run_discrete_survival` 用 `total_age_0` 原始总数；WF 路径注释明说 "only age-0 juveniles compete, actual_competition_strength is just the total juvenile count"）。`(2,)` 数组仅服务于与共享 `compute_equilibrium_metrics` 代码的兼容（`config.py:240-245` 注释 "kept for spatial builder compat; inactive in discrete" 已部分覆盖此意）。

后续若做 SU-D1 的 FIXED-only 收敛，可一并移除该字段在离散代的消费；否则维持现状（默认 `np.ones`，rel[0]=1.0 自洽）。
---

## 🔴 高优先级 — 正确性 / 阻塞项

## 🟡 中优先级 — 性能 / 可维护性

### #3 📋 Observation 录制逻辑在模板和 Python 路径中重复

**此分支改动**：无。仅在本 TODO 中新增记录为遗留项，零提交触及 observation 录制逻辑。`observation_record.py` 及其辅助函数（`build_observation_row_panmictic`、`build_observation_row_spatial`）与此分支前状态一致。

**优先级理由**：🟡 三条路径（Numba 内核模板、Python dispatch 回退、后处理）的录制逻辑手工重复，一处改漏可能导致数据不一致。虽非紧急正确性 bug，但随着 v0.2.0 发布后用户增多，维护风险上升。

- `RUN_FN_NAME`（4 个模板）中手工构造 `flat_state` + `observation_mask` 聚合 → Numba 内核路径
- `_run_python_dispatch`（2 个模型）中调用 `create_history_snapshot()` → Python 回退路径
- `_process_kernel_history` 中将内核 raw array 转为 History 对象 → 后处理
- 三种路径的 flatten 格式因生命周期类型不同（discrete / structured / spatial compact），但录制时机和条件判断逻辑相同
- 改善方向：将 `flatten_size` 计算和 `flat_state` 填充抽成 `@njit` 辅助函数，按生命周期类型参数化

### #4 ⚠️ Zygote modifier 矩阵化与稀疏表示

- zygote 侧仍使用 Dict[Genotype, float] 逐 rule 迭代，未矩阵化
- `ModifierMatrix` 稀疏表示未实现（当前 dense 在 n_gtypes ≤ 250 时足够快）

### #5 ⚠️ Spatial History

**此分支改动**：无。所有 spatial history 基础设施（录制、解析、导出）均在主分支上已完成，此分支未做修改。

**优先级理由**：🟡 Per-deme 历史录制和 UI 导出已实现，但 `import_state()` 缺失 —— panmictic 模型（`DiscreteGenerationPopulation`、`AgeStructuredPopulation`）均有 `import_state()`，SpatialPopulation 没有。对于需要 checkpoint/restore 的长期空间模拟是阻塞性缺失。

- 保存每个 deme 的 History 数据，提供快捷解析和导出方法
- 支持 UI 导出
- 支持刷新后加载历史数据

### #6 📋 改 `late_..._resistance` 为 `absolute_resistance`

**此分支改动**：无。`absolute_resistance` 在该分支的 Python 源码、测试、demo、文档中均未出现。所有位置仍使用 `late_germline_resistance_formation_rate`。

**优先级理由**：🟡 纯 API 重命名，不涉及正确性或性能。但若计划在 v0.2.0 发布前完成此变更，则需尽快决定——发布后改名就是 breaking change。建议与 #7（embryo resistance 灵活化）一并设计，避免两次改动同一参数体系。

- 增加快捷设置方式，不删原有参数
- $d+r>1$ → 报错

### #7 📋 灵活化 embryo resistance rate 配置

**此分支改动**：无。`embryo_resistance_formation_rate` 仍为静态 `_SexSpecificRates`（`Tuple[float, float]`），无 Cas9 拷贝数依赖，无杂合/纯合区分。

**优先级理由**：🟡 增强功能，非 bug。对于使用 CRISPR 驱动元件（Homing Drive、Toxin-Antidote Drive）的模拟场景有意义，但取决于具体研究需求。建议与 #6 的 `absolute_resistance` 改动一同设计，统一 resistance 参数体系。

- 未必是定值，可与亲本中 Cas9 copies（或表达时间）有关
- 可支持 heterozygotes / homozygotes 不同配置

### #9 ⚠️ 重复的 modifier map 重建逻辑

**来源**：`code-quality-review-report.html` #5

**当前状态**：离散代中的冗余覆写已经移除。剩余双重实现是
`ModifierPresetMixin.refresh_modifier_maps()` 与 `Configurator._rebuild_config_maps()`；两者分别服务运行时和构建期，但必须维持相同的 Mendelian 基线与压缩轴投影语义。

**优先级理由**：🟡 维护负担——任一入口的改动都可能遗漏同步到另一入口。

- 提取公共核心为独立辅助函数，由两个入口共享

### #9.1 📋 Preset modifier 定向重编译与后缀重建

**来源**：2026-08-15 conversion refresh 修复后的架构讨论。

**当前行为**：`reconfigure_preset(preset, ...)` 能按对象身份找到被修改的
preset，但 `refresh_modifiers()` 仍会清空全部派生 modifier，按 priority 重新调用
所有 preset 的 `gamete_modifier()` / `zygote_modifier()`，随后从 Mendelian 基线重放
完整 modifier 列表并重算 `offspring_tensor` 和 preset fitness。

并非所有 modifier 都是矩阵：`GameteConversionRuleSet` 只在单个 ruleset 内编译并
组合 GType 转换矩阵；zygote conversion 仍生成分布字典，fitness 使用 patch，自定义
modifier 则是不透明 callable。不同 preset 修改同一行时，目前也尚未正式定义应当
“顺序转换”还是“后者覆盖”。在明确该组合语义前，不能安全地直接加入后缀缓存。

**优先级理由**：🟡 架构与运行时配置性能。preset 数量较多或压缩映射较大时，修改
一个参数却重新解析全部规则会产生不必要开销；但 `offspring_tensor` 的全量卷积可能
仍是主要成本，应先基准测试再决定是否引入占用大量内存的中间 checkpoint。

**建议分阶段实现**：

1. 为派生 modifier 保存明确的 preset owner 身份和 priority，不依赖名称前缀关联。
2. 先实现低风险版本：只重新编译发生变化的 preset，复用其他 preset 的已编译产物；
   map 仍从 Mendelian 基线重放全部已编译阶段，`offspring_tensor` 仍完整重算。
3. 统一内部阶段接口，明确 gamete、zygote、fitness 和 custom modifier 的输入/输出及
   跨 preset 组合语义。
4. 仅在基准证明值得时，缓存每个 preset 之前的 map checkpoint，修改第 N 个 preset
   时恢复 N-1 的结果并只重放 `[N, end)`；同时定义 registry/compression、preset
   增删、priority 变化和 manual modifier 变化时的缓存失效规则。

**验收要求**：

- 重配置结果与相同最终参数的 fresh build 逐元素一致
- 覆盖多个 preset 修改重叠行与不重叠行、相同/不同 priority、custom modifier
- 覆盖 age/discrete/spatial、compress 开关和稀疏 GType/ZType
- 记录“仅定向重编译”和“checkpoint 后缀重建”的时间、峰值内存及 break-even preset 数

### #11.5 ⚠️ Modifier 系统：genotype vs ztype 概念混用 + 冗余参数

**来源**：2026-07-10 `expand_to_ztypes` 清理后的进一步审计。

**遗留子项**：
- 📋 命名修正：`GameteModifier` Protocol docstring 中 `genotype_idx` → `ztype_idx`、`_write_zygote_mapping` docstring、`_normalize_zygote_val` docstring
- 📋 协议扩展：让 modifier 支持 slab-level 目标选择（当前 `ztype_indices_for()` 无条件全板展开）
- 📋 Conversion ruleset 新 DSL（Condition 组合条件、`add_glab_convert`、`add_slab_convert`）API 已就绪，内部委托到旧 API；矩阵编译（`to_matrix(registry)`）和完整迁移待 Stage 2

**涉及文件**：`src/natal/modifiers/module.py`、`src/natal/presets/cytoplasmic.py`、`src/natal/population/_mixins/_modifiers.py`、`src/natal/configurator/_registry_builder.py`

### #11.1 ⚠️ Hook 系统测试覆盖缺口

**来源**：2026-06-17 测试审计。`test_hook_kernel_ops.py` 是独立脚本不被 pytest 发现，`_apply_target_with_sperm` 零覆盖，多个 Op 类型无端到端生命周期测试。

**优先级理由**：🟡 `_apply_target_with_sperm` 是最复杂的执行路径（virgin/sperm 拆分、随机采样、负值检测），其 bug 会静默破坏 sperm 数据。

**遗留**：
- `test_hook_kernel_ops.py` 需转换为 pytest 格式（所有 Op 类型的运行时测试当前仅在直接执行时运行）
- `execute_csr_event_program_with_state` 无直接单元测试（已被模板间接覆盖）
- `_check_csr_condition` 无直接单元测试（已被 condition interpreter 测试覆盖）

---

## 📝 文档清理 — 过时路径引用

> 以下条目由 `refactor/hooks-naming` 的对抗式 code review workflow 发现。模块路径已重命名，但文档/注释/缓存中仍有旧引用。
> 本分支已修复 `src/` 和 `tests/` 范围内的全部 stale 引用（6 处）。`docs/` 和 `.numba_cache/` 不在此分支范围。

### #14 ⚠️ 文档中仍有过时的 `hook_executor` 字段

`natal/hooks/compiler.py` 和 `natal.hooks.executor` 等旧模块路径已经清理；目前仅剩
`docs/{zh,en}/spatial_builder.md` 与 `spatial_configurator.md` 共 8 处
`hook_executor` 字段说明，与当前运行时结构不一致。

### #15 📋 `.numba_cache/` 缓存模块包含旧 import 路径

`git clean -fdx .numba_cache/` 可解决。缓存模块是运行时生成/覆盖的，不会导致失败，但存在误导性。

### #16 📋 spatial `deme_id` 合并+过滤机制未在文档中说明

当前文档（`spatial_lifecycle_wrapper.md`、`3_advanced_hooks.md`）描述了 `_collect_effective_compiled_hooks()`（"收集所有 deme 的 hook"）和 Hook 签名接受 `deme_id` 参数，但**未解释两者的因果关系**：

- **实际机制**：所有 deme 的 hook 被打平进一份全局 `CompiledEventHooks`，编译为一组 lifecycle wrapper；在 `prange` 中每个 deme 调用同一组 wrapper，通过 `deme_id` 过滤——CSR 路径用 `njit_deme_selector_matches()` 跳过不匹配的 hook，njit 路径生成 `if deme_id == X` guard。
- **文档给人的印象**：每个 deme 独立运行自己的 hook 列表，`deme_id` 只是个"我是几号"的上下文。
- **待补充**：在 `spatial_lifecycle_wrapper.md` 的编译阶段添加一段解释合并+过滤的设计动机（编译一次 vs 编译 N 次）。

---

### #22 ⚠️ Numba JIT 缓存冲突导致空间测试排序依赖

**来源**：`feat/ztype-registry` 分支测试调试。`test_discrete_population.py` 先于 `test_spatial_builder_coverage.py` 运行时，空间测试中 30 个离散世代构建场景因 Numba JIT 缓存冲突失败：

```
RuntimeError: In 'NRT_adapt_ndarray_to_python', 'descr' is NULL
```

发生在 `run_discrete_survival()` 尝试从 `individual_count` 返回数组时——Numba 的 NRT（Numba Runtime）内部 dtype descriptor 损坏。根本原因：`test_discrete_population` 首先编译 `run_discrete_reproduction`/`run_discrete_survival` Numba 函数，使用更大的 `n_ztypes` 值（3 基因型 × 1 slab = 3）。后续 `test_spatial_builder_coverage` 的某些测试使用不同的 `n_ztypes` 值，Numba 尝试复用缓存的编译产物，但不同的数组形状导致 dtype descriptor 指针误对齐。空间测试单独运行时全部 69 个通过；先于离散测试运行时通过。

**当前措施**：`tests/conftest.py` 将空间构建器测试重新排序到 pytest 收集顺序的前面（第一个运行）。这避免了冲突，但并未解决根本的 Numba 缓存问题。

**优先级理由**：🟡 非生产 bug——仅影响测试套件排序。当前解决方法可行。若要正确修复，需要将 spatial builder 测试中使用的 `n_ztypes` 大小与其他离散世代测试对齐，或调查 Numba 的跨测试缓存失效问题。

**受影响范围**：
- `tests/test_spatial_builder_coverage.py`：69 个测试中 30 个受影响
- 触发条件：`test_discrete_population.py`（18 个测试，`n_ztypes=3`）在 `test_spatial_builder_coverage.py`（用 `n_ztypes=1` 的测试）之前运行
- 非确定性：同一个提交可能通过或失败，取决于 pytest 的收集顺序
- `main` 分支同样受影响（`143af2c`）：仅因测试文件更少而以有利的收集顺序通过

### #23 ⚠️ 统一 Numba / Python 后端选择与测试缓存隔离

**来源**：`refactor/history-observation` 分支调试。单独运行
`TestMigrationKernelBank::test_kernel_bank_build_and_run` 时，默认
`.numba_cache` 会让 Python 进程在 `_run_python_dispatch_tick()` 调用
`run_spatial_migration()` 时直接以 `SIGABRT`（exit code 134）退出；代码不变，
仅将 `NUMBA_CACHE_DIR` 指向全新临时目录后测试通过。

**根因与设计问题**：

- Numba 文件缓存不能识别被调用函数在其他文件中的变化，也会保留编译时读取的
  全局常量；NATAL 的 lifecycle、migration、NamedTuple config 和 codegen 跨多个
  文件组合，单个源文件时间戳不足以表达真实缓存 ABI。
- `@njit_switch` 在模块导入时决定返回原始函数还是 `CPUDispatcher`；
  `numba_disabled()` 只在运行时切换 `NUMBA_ENABLED`，无法把已经导入的
  `CPUDispatcher` 还原成 Python 函数。因此 `with numba_disabled():` 不保证真正
  走纯 Python fallback。
- 测试目前混用四套控制方式：`numba_disabled()` 上下文、
  `@pytest.mark.numba_off/on`、直接 `enable_numba()/disable_numba()`，以及
  `NATAL_DISABLE_NUMBA=1` 的独立 pytest 进程。其语义和状态恢复规则不同。
- `tests/conftest.py` 通过调整测试收集顺序规避缓存冲突，只是 workaround，无法
  阻止单测、并行 worker、分支切换或 dirty worktree 命中不兼容的 native 缓存。

**近期统一方案（优先实施）**：

1. 后端只允许在进程启动、导入 `natal` 之前选择，统一为
   `NATAL_BACKEND=python|numba`；Python 模式同时设置官方
   `NUMBA_DISABLE_JIT=1`，不再把运行时 context manager 当作集成测试后端开关。
2. 提供单一测试入口（例如 `scripts/test_backend.py`），分别启动 Python 与 Numba
   两个 pytest 子进程；调用者不再手工组合 marker 和环境变量。
3. marker 统一为 `python_backend`、`numba_backend`、`backend_parity`。Parity 测试
   在两个独立进程分别对同一份独立数学期望断言，不在同一进程内切换后端。
4. 测试运行使用独立缓存：
   `/tmp/natal-core-tests/<run-id>/<backend>/<worker-id>`。不得读取或写入开发环境的
   `.numba_cache`；隔离完成后删除 `tests/conftest.py` 的测试排序 workaround。
5. 禁止集成测试直接调用 `enable_numba()` / `disable_numba()` 或使用
   `numba_disabled()`；后者只保留给 `njit_switch` 自身的窄单元测试，并修正文档，
   明确它只能影响后续装饰/分派选择。

**持久缓存修复**：

- 用户运行时仍可使用持久缓存，但目录必须按 Python cache tag、Numba/NumPy
  版本、CPU 架构、显式 `NATAL_NUMBA_CACHE_ABI_VERSION` 和相关源文件内容 hash
  分 namespace。
- 必须 hash 实际文件内容，不能只使用 Git commit；否则 dirty worktree 的 ABI
  变化无法触发新缓存。
- fingerprint 变化时直接使用新目录，不尝试原地修复或探测旧 `.nbc`。native
  缓存错误可能直接 abort/segfault，Python 层没有可靠的恢复机会。

**长期后端 seam**：

建立进程启动后冻结的 `ExecutionBackend`，由 `PythonBackend` 与 `NumbaBackend`
两个 adapter 统一提供 lifecycle、steps 和 spatial migration 接口。Population 只
依赖这个 seam；`njit_switch` 与缓存管理退回后端模块内部。完成该 seam 前，不应
承诺支持导入后的运行时后端切换。

**验收标准**：

- 相同测试按任意顺序运行均不再触发 NRT 错误、SIGABRT 或 segfault。
- Python backend 进程中的 `@njit` / `@njit_switch` 调用均进入原始 Python 函数。
- Numba 与 Python parity 测试在独立进程中逐元素满足同一数值不变量。
- pytest-xdist worker 和连续两次本地测试不共享可写缓存目录。
- 删除测试排序 workaround 与集成测试中的运行时后端切换后，完整双后端门禁通过。


## 🟢 低优先级 — UX / 远期功能

### #12 📋 Spatial migration kernel 边界效应优化

**优先级理由**：🟢 功能优化，不涉及正确性。基础的 `apply_migration_adjacency`、`build_gaussian_kernel` 等已在 `spatial_topology.py` 中可用。

- 优化 migration kernel，处理边界效应（总迁移率不应不变，而应正比于邻居数量；或可不用总迁移率设置，尝试全部设为 1；需要一个优雅的方法）

### #13 📋 K 值自动推导路径测试

**来源**：`code-quality-review-report.html` #15

**此分支改动**：无。Configurator 路径的 K 值自动推导优先级链无测试。

**优先级理由**：🟢 低优先级覆盖缺口。自动推导是 fallback 逻辑，主路径已有测试。

- 添加测试验证优先级链：`carrying_capacity` > `age_1_carrying_capacity` > `initial_individual_count`
- 覆盖 Configurator 和 `pop.update()` 两个入口

### #14 🎨 PointMutation 预设 —— 多点突变 + 概率自动校正

**来源**：2026-06-05 设计讨论。用户需求：同时声明 source → [target₁, target₂, …] 的多条点突变，且各 target 的突变率互不干扰（"同时竞争"语义，而非 "先到先得"的级联语义）。

**优先级理由**：🟢 新功能。最简形式（单 source → 单 target）实现量小（~80行），可直接参考 `ToxinAntidoteDrive` 的模式。多 target + 概率校正约 30 行增量。不影响现有 preset。

**设计方案**：

1. **单 target 基础形式**（对标 `ToxinAntidoteDrive` 的简洁度）：

   ```python
   PointMutation("A2B", source_allele="A", target_allele="B", mutation_rate=1e-5)
   ```

   - `gamete_modifier`：`add_allele_convert(A→B, rate, sex_filter=sex)`，**不传 `genotype_filter`**（点突变是自发的，不依赖父本基因型）
   - `zygote_modifier`：默认返回 `None`。可选 `zygotic_mutation_rate` 参数支持胚胎期突变
   - `fitness_patch`：对 `target_allele` 调用 `_make_fitness_patch_given_allele_scaling()`

2. **多 target 扩展形式**：

   ```python
   PointMutation("MultiMut",
       source_allele="A",
       target_alleles=["B", "C", "D"],
       mutation_rates=[1e-7, 5e-6, 1e-5],
   )
   ```

3. **概率自动校正**（核心设计决策）：

   **问题**：`GameteConversionRuleSet` 内部规则是顺序级联的——Rule2 只作用于 Rule1 处理后的"剩余 source"。如果直接传用户声明的 rate，B 先抢走一部分 source，C 只能从剩余中分，有效速率会偏离用户期望。

   **为什么校正放在 PointMutation 层而非 RuleSet 层**：RuleSet 的级联语义是有意设计的——HomingDrive 的 "homing → resistance" 级联是生物学过程的忠实建模（resistance 只作用于 homing 失败的 target）。这不是 bug，不能"修正"。但点突变的多个产物是同一生物学过程的互斥结果，应该"同时竞争"——校正逻辑属于 PointMutation 的业务语义。

   **校正公式**：`r'ₖ = rₖ / (1 - Σᵢ₌₁ᵏ⁻¹ rᵢ)`

   其中 `r'ₖ` 是传给 RuleSet 的调整后速率，`rₖ` 是用户声明的期望有效速率。校正后，无论规则以什么顺序插入，每个 target 拿到的有效份额恰好等于 `rₖ`。

   **数值示例**（`r = [0.3, 0.5, 0.1]`）：

   | k | 期望 rₖ | 调整后 r'ₖ | 有效份额 |
   |---|---------|-----------|---------|
   |1| 0.3 | 0.3 | 0.3 × 1.0 = 0.3 ✓ |
   |2| 0.5 | 0.714 | 0.714 × 0.7 = 0.5 ✓ |
   |3| 0.1 | 0.5 | 0.5 × 0.2 = 0.1 ✓ |

   最终 source 剩余 = `1 - 0.3 - 0.5 - 0.1 = 0.1` ✓

4. **Σr > 1 的处理**：默认 `raise ValueError`（mutation rate 通常很小，几乎不会触发）。可选 `rate_mode="proportional"` 自动等比缩放到和为 1，方便用户用比例而非概率表达。

5. **性别维度**：校正按性别分别进行——`mutation_rates` 列表中每个元素本身可以是 `_SexSpecificRates`（`float | tuple | dict`），先 `_resolve_rates()` 展开为 `(female_rate, male_rate)`，再对每个性别独立校正。

6. **与手动叠加两个 PointMutation 的对比**：

   | 方式 | 语义 | 问题 |
   |------|------|------|
   | 两个独立 preset | 顺序级联（先到先得） | rate 大时有效份额偏离期望；顺序依赖 |
   | 多 target 单 preset + 校正 | 同时竞争（互斥） | 无 |

**实现路径**（纯加法，不改现有 API）：

| 文件 | 改动 |
|------|------|
| `genetic_presets.py` | 新增 `PointMutation` 类（~110 行），`__all__` 添加导出 |
| `test_genetic_presets.py` | 添加单 target / 多 target / 校正公式 / Σr>1 边界测试 |

**不改**：`GameteConversionRuleSet`、`GameteAlleleConversionRule`、`modifiers.py`、任何 Numba 内核。

### #17 ⚠️ `PopulationConfig._replace()` 导致 0-d ndarray 退化为 Python scalar（3 个测试失败）

**来源**：`NATAL_DISABLE_NUMBA=1 pytest` 发现（2026-06-20）。`test_spatial_population_integration.py` 中 3 个 `@pytest.mark.numba_off` 测试失败。

**根因**：

```python
# test_spatial_population_integration.py:258
cfg1 = demes[1].export_config()._replace(low_density_growth_rate=1.7)
```

`PopulationConfig` 的 `low_density_growth_rate` 字段是 **0-d ndarray**（`np.array(1.0)`），但 `namedtuple._replace(1.7)` 用 Python float 替换了它。随后 `age_structured_simulator.py:289`：

```python
config.low_density_growth_rate[()]  # 0-d indexing → float[()] → TypeError
```

**为什么 Numba 路径不出错**：Numba JIT 在编译时已将 config 字段类型固定为 0-d ndarray，运行时可能隐式包装。确切原因待验证。

**修复方向**（三选一）：
- A) 测试端：`_replace(low_density_growth_rate=np.array(1.7))`
- B) `import_config()` 内自动包装 scalar → 0-d ndarray
- C) `age_structured_simulator.py` 的 `[()]` 索引前加 `np.asarray()` 防护

**影响的测试**：
- `test_spatial_population_run_tick_supports_heterogeneous_deme_configs`
- `test_spatial_population_heterogeneous_configs_use_python_hook_dispatch`
- `test_spatial_population_heterogeneous_configs_run_uses_hook_dispatch_each_step`

---

## v0.3.0 及远期更新

> 以下四大功能详见 `v0.3.0-acceleration-and-compression-design.html` 综合设计方案。

### #19 ⚠️ Somatic Label (slab) 的转换能力补全

Somatic Label、扁平 ZType/GType 索引、slab-aware fitness/hook/observation、压缩及
`CytoplasmicPreset` 已实现。原设计中的独立 4-D state 方案已被扁平 ZType 方案取代。
仍缺少通用的 slab 转换 API：

1. **三类 Slab 转换**：
   - `T_zygotic`（glab → slab）：受精时，给定母本 glab、父本 glab、合子基因型 → 子代 slab 分布
   - `T_gametic`（slab → glab）：减数分裂时，给定个体 slab、基因型、性别 → 配子 glab 分布
   - `T_somatic`（slab → slab）：每 tick 存活阶段，个体 slab 转换（如 Cas9 表达衰退）

2. 提供 `add_slab_convert` 等公开 DSL，替代 `CytoplasmicPreset` 内部的自制循环。
3. 如需 tick 间 `T_somatic`，需明确它在生命周期中的执行阶段和 Hook 顺序。

### #20 🎨 仿真引擎性能优化审计

**来源**：2026-06-21 架构审计。对引擎热路径的 6 个优化点进行系统评估。

**优先级理由**：🟡 性能工程。4 个纳入 v0.3.0，2 个推迟。均为纯优化，不改行为。

**已完成**：offspring tensor 已由 `@njit_switch` 的
`compute_offspring_probability_tensor()` 计算，且避免构造 O(G²·HL²) 中间数组。

**仍待评估的优化**：

| ID | 优化 | 难度 | 预期收益 | 行数 |
|----|------|------|---------|------|
| #B | CSR prange 并行化 | 中 | 2-4×（G≥200 时，per-op 内 genotype 维度并行） | ~120 |
| #C | 交配矩阵缓存 | 低 | ~30% 交配计算开销 | ~80 |
| #D | 内存分配复用（TickBuffers） | 中 | 减少 40-60% 分配调用 | ~200 |

**推迟的优化**：
- #E（deme 间负载均衡）：仅在 deme 间个体数差异 >10× 时有意义，大多数均匀场景无收益。
- #F（观测录制路径统一）：与 TODO #3 重复，维护收益 > 性能收益。

### 远期功能

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

---

## Conversion Ruleset 重构（2026-07-10 grill session）

### 📋 Stage 2 待做 — 剩余迁移

1. `add_slab_convert` — gamete/zygote 端 slab 操作（当前 CytoplasmicPreset 自制循环）
2. `extract_gamete_frequencies_by_glab` 调用精简（CytoplasmicPreset 路径仍保留）

### 剩余 P2 命名清理（grill list #8-#16）

| # | 位置 | 问题 |
|---|------|------|
| 8 | engine 40+ 处 | docstring `n_genotypes` 实际是 `n_ztypes` |
| 9 | `age_structured.py:95-106` | `n_g_orig` 在 genotype/ztype 间摇摆 |
| 10 | `hooks/declarative.py:268` | `_resolve_genotypes` → `_resolve_ztypes` |
| 11 | `discrete_generation.py:161` | `n_genotypes = config.n_ztypes` |
| 12 | `migration/adjacency.py:619` | `genotype_idx` → `ztype_idx` |
| 13 | `configurator/_base.py` 10+ 处 | docstring "genotype" 应为 "ztype" |
| 14 | `configurator/_factory.py` 5+ 处 | docstring "genotype" 应为 "ztype" |
| 15 | `engine/age_structured.py` | `n_haplogenotypes`/`n_glabs` 标记 unused |
| 16 | `population/age_structured.py:95-106` | `n_g_orig` 语义歧义 |
