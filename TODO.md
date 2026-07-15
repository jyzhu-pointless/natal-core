# TODO

> 最后审计：2026-06-21。新增 v0.3.0 四大功能设计（索引压缩、Somatic Label、引擎优化、极速模式），详见 `v0.3.0-acceleration-and-compression-design.html`。
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

> 本节记录 `.codex/plans/history_observation_refactor_plan.md` grill session 中已经讨论、但明确不应随当前增量重构一并实现的设计。当前重构只实现构建期 canonical Observation、单模式 History、post-hoc observation、`record_snapshot()` 和 raw checkpoint restore。

### HO-C1 📋 natal-inferencer 接口协调

natal-core 的最终接口稳定后，在 `natal-inferencer` 单独实施：

- 将 `population.record_observation` 替换为只读 `population.observation`。
- 粒子数组投影统一使用 `population.observation.apply(particle_counts)`。
- 接受 Population 自动提供 identity Observation 的默认行为。
- 删除对 `pop.create_observation()`、旧 output helpers 和兼容 alias 的依赖。
- 增加跨仓库集成测试，覆盖默认 identity 与显式 Observation。

natal-core 不为此保留 `record_observation` shim；两个项目尚未发布，可以直接协调升级。

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

## 🔴 高优先级 — 正确性 / 阻塞项

### #2 ✅ Selector + custom hook 调用约定统一

**已完成**（`refactor/selector-hook-calling-convention` 分支）：

- **Selector hooks**：Numba 路径 wrapper 不再提取 `ind_count`/`tick`，直接转发 `(state, config, deme_id=deme_id, selector_kwargs...)`。Python 路径 `py_wrapper` 从 `(population)` 改为 `(state, config, deme_id=-1)`。`_compile_selector_njit_wrapper` 模板化。
- **Custom hooks**（无向后兼容）：`_normalize_njit_fn` 和 `_normalize_py_hook` 不做旧约定检测。`test_hook_priority_mixed.py`（2 测试）和 `test_spatial_population_run.py`（3 测试）的旧约定 hook 已迁移到 `(state, config, deme_id)`。
- **测试覆盖**：25 测试按 `@pytest.mark.numba_on` / `numba_off` 分区，新增 2 个 Numba 端到端测试 + 8 个 Python fallback 测试。
- **其余修复**：`test_lifecycle_wrappers.py` 中 2 个测试加了 `@pytest.mark.numba_on`（原无标记，Numba 禁用时错误运行）。
- **剩余**：`test_spatial_population_integration.py` 3 个测试 — `_replace(low_density_growth_rate=1.7)` 导致字段退化为 Python float（非 0-d ndarray），在 `age_structured_simulator.py:289` 的 `[()]` 索引处崩溃。见下方 #17。

---

## 🟡 中优先级 — 性能 / 可维护性

### #3 📋 Observation 录制逻辑在模板和 Python 路径中重复

**此分支改动**：无。仅在本 TODO 中新增记录为遗留项，零提交触及 observation 录制逻辑。`observation_record.py` 及其辅助函数（`build_observation_row_panmictic`、`build_observation_row_spatial`）与此分支前状态一致。

**优先级理由**：🟡 三条路径（Numba 内核模板、Python dispatch 回退、后处理）的录制逻辑手工重复，一处改漏可能导致数据不一致。虽非紧急正确性 bug，但随着 v0.2.0 发布后用户增多，维护风险上升。

- `RUN_FN_NAME`（4 个模板）中手工构造 `flat_state` + `observation_mask` 聚合 → Numba 内核路径
- `_run_python_dispatch`（2 个模型）中调用 `create_history_snapshot()` → Python 回退路径
- `_process_kernel_history` 中将内核 raw array 转为 History 对象 → 后处理
- 三种路径的 flatten 格式因生命周期类型不同（discrete / structured / spatial compact），但录制时机和条件判断逻辑相同
- 改善方向：将 `flatten_size` 计算和 `flat_state` 填充抽成 `@njit` 辅助函数，按生命周期类型参数化

### #4 🎨 修饰器矩阵化 —— Sequential Cascade → 矩阵乘法

**状态**：✅ 已实现。`GameteConversionRuleSet.to_matrix(population)` 编译 rules 为
`{(sex, ztype): (n_gtypes, n_gtypes)}` dense float64 矩阵，
`to_gamete_modifier()` 通过 `freq_vec @ M` 替代旧 `_compute_converted_gamete_freqs`
逐 rule 迭代。旧 Cascade 函数已删除（~114 行）。

**剩余工作**：
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

### #8 ✅ Selector hook 测试增强

**来源**：`code-quality-review-report.html` #12, #15

**已完成**（同 #2 一起修复）：
- `test_hook_selector_mode.py` 新增 8 个测试（共 23 个），覆盖 `desc.selectors` 值验证 + Python fallback 端到端调用
- `TestSelectorResolution`：验证单选/通配符/多选/int 选择器解析为正确的 int32 索引数组
- `TestPythonFallbackEndToEnd`：实际调用 `desc.py_wrapper(state, config, deme_id)` 验证 expand/aggregate/deme_id/multi-genotype 路径的功能正确性

### #9 ⚠️ 重复的 modifier map 重建逻辑

**来源**：`code-quality-review-report.html` #5

**此分支改动**：refresh 系统重构已移除 `discrete_generation_population` 中的冗余副本（原为父方法逐字覆写）。剩余双重实现：`BasePopulation.refresh_modifier_maps()` 与 `Configurator._rebuild_config_maps()`，两者策略不同（从头构建 vs 从 Mendelian blueprint 基线叠加），仍有维护二元性。

**优先级理由**：🟡 维护负担——三处任意一处的改动可能遗漏同步到其他两处。

- 提取公共核心为独立辅助函数，三处共享
- 删除 `discrete_generation_population` 的覆写方法（完全等同于父方法，可直接 `super()`）

### #10 📋 Configurator 对弃用 builder 模块的耦合

**来源**：`code-quality-review-report.html` #11

**此分支改动**：无。`configurator.py` 中 5 个方法依赖 `population_builder` 的 utility 函数（如 `resolve_age_param`）。

**优先级理由**：🟡 架构债务。`configurator.py`（新模块）依赖 `population_builder.py`（弃用模块）的方向是反的。

- 将共享工具函数迁移至 `population_config.py`，与 commit `1b62a3c` 的 `build_custom_array` 迁移模式一致

### #11 📋 Modifier 效果端到端测试

**来源**：`code-quality-review-report.html` #14

**此分支改动**：无。`test_gamete_and_zygote_modifier_together` 未验证实际的 `genotype_to_gametes_map` 或 `offspring_tensor` 改变。

**优先级理由**：🟡 测试只验证 modifier 不崩溃，不验证其效果正确性。

- 添加非平凡 modifier 的端到端效果测试（验证 genotype_to_gametes_map 或 offspring_tensor 改变）

### #11.2 ✅ `BatchSetting` 类型安全 —— 泛型化消除 `Any` 泄漏

**来源**：2026-07-09 代码审查。`BatchSetting` 从入口到出口一路 `Any`，类型信息完全丢失。

**已完成**（`fix/compress-once-ztype-refactor` 分支）：
- `BatchSetting(Generic[_T])`：`expand() → List[_T]`，`first_value() → Optional[_T]`
- `batch_setting()` 返回 `BatchSetting[_T]`
- 所有未参数化的 `BatchSetting` 引用改为 `BatchSetting[Any]`

**涉及文件**：`src/natal/spatial/configurator.py`、`src/natal/spatial/population.py`

### #11.3 ✅ `_compress_once` 重构 —— ztype 作为一等公民，消除 genotype 字符串中转

**来源**：2026-07-09 PR #32 Copilot review Comment 1+2+3 的共同根因。

**已完成**（`fix/compress-once-ztype-refactor` 分支）：
- `union: set[str]` → `seeds: set[int]`，返回值同步
- 模板构建前置，种子从 `initial_individual_count` 非零 z_idx 收集
- Hook refs 和 declared 字符串经 `_resolve_declared_to_ints` 转 ztype int
- 整数 `declared_zygote_types` 直接入种子
- `extra_declared` 改为 `set[int]`，合并逻辑简化

**涉及文件**：`src/natal/spatial/configurator.py`

### #11.4 ✅ 删除 `wrap_gamete_modifier` / `wrap_zygote_modifier` / `build_modifier_wrappers` 的死参数 `expand_to_ztypes`

**来源**：2026-07-09 PR #32 Copilot review Comment 3 的反向分析。

**已完成**（`fix/compress-once-ztype-refactor` 分支）：
- `src/natal/modifiers/module.py`：三个函数签名中删除参数及 docstring，转发代码移除
- `src/natal/configurator/_registry_builder.py`：调用方移除 `expand_to_ztypes=...` 传参

### #11.5 ⚠️ Modifier 系统：genotype vs ztype 概念混用 + 冗余参数

**来源**：2026-07-10 `expand_to_ztypes` 清理后的进一步审计。

**已完成的子项**（`feature/spatial-compress-unified` 分支）：
- ✅ 冗余参数清理：`_resolve_gidx`、`_apply_comp_map` 等 7 个旧函数删除，统一层 `_resolve_ztype_key`/`_resolve_gtype_key` 仅接受 registry
- ✅ `CytoplasmicPreset` 特判消除：`gamete_modifier()` 和 `zygote_modifier()` 返回真实 modifier，`isinstance(preset, CytoplasmicPreset)` 全部删除
- ✅ 手动 glab 公式清理：`module.py:555-556` 删除（统一层替代），`cytoplasmic.py` 中 glab 索引改走 registry
- ✅ `build_compression_mask` 死参数 `n_glabs`/`n_slabs` 删除，G→n_zt/HL→n_gt 重命名
- ✅ `_compress_once` ztype 一等公民化，`seeds: set[int]` 替代 `union: set[str]`
- ✅ `batch_setting` 泛型化 `Generic[_T]`
- ✅ `gamete_conversion.py` + `zygote_conversion.py` ztype/gtype 适配

**遗留子项**：
- 📋 命名修正：`GameteModifier` Protocol docstring 中 `genotype_idx` → `ztype_idx`、`_write_zygote_mapping` docstring、`_normalize_zygote_val` docstring
- 📋 协议扩展：让 modifier 支持 slab-level 目标选择（当前 `ztype_indices_for()` 无条件全板展开）
- 📋 Conversion ruleset 新 DSL（Condition 组合条件、`add_glab_convert`、`add_slab_convert`）API 已就绪，内部委托到旧 API；矩阵编译（`to_matrix(registry)`）和完整迁移待 Stage 2

**涉及文件**：`src/natal/modifiers/module.py`、`src/natal/presets/cytoplasmic.py`、`src/natal/population/_mixins/_modifiers.py`、`src/natal/configurator/_registry_builder.py`

### #11.1 ⚠️ Hook 系统测试覆盖缺口

**来源**：2026-06-17 测试审计。`test_hook_kernel_ops.py` 是独立脚本不被 pytest 发现，`_apply_target_with_sperm` 零覆盖，多个 Op 类型无端到端生命周期测试。

**优先级理由**：🟡 `_apply_target_with_sperm` 是最复杂的执行路径（virgin/sperm 拆分、随机采样、负值检测），其 bug 会静默破坏 sperm 数据。

**已完成**：`d3aab26` 补充了 25 个测试：
- `_apply_target_with_sperm` / `_apply_target_without_sperm` 14 个单元测试
- `stop_if_zero` / `stop_if_extinction` / 条件不满足 3 个 E2E 测试
- `Op.scale/sample/kill/subtract` 4 个 E2E 测试
- 边界 case（空 hook、单 hook、同 priority）4 个测试

**遗留**：
- `test_hook_kernel_ops.py` 需转换为 pytest 格式（所有 Op 类型的运行时测试当前仅在直接执行时运行）
- `execute_csr_event_program_with_state` 无直接单元测试（已被模板间接覆盖）
- `_check_csr_condition` 无直接单元测试（已被 condition interpreter 测试覆盖）

---

## 📝 文档清理 — 过时路径引用

> 以下条目由 `refactor/hooks-naming` 的对抗式 code review workflow 发现。模块路径已重命名，但文档/注释/缓存中仍有旧引用。
> 本分支已修复 `src/` 和 `tests/` 范围内的全部 stale 引用（6 处）。`docs/` 和 `.numba_cache/` 不在此分支范围。

### #14 📋 文档引用过时的模块路径

`docs/` 下 8 个 .md 文件引用已删除或已移动的路径：

| 文件 | 旧路径 | 正确路径 |
|------|--------|---------|
| `docs/en/spatial_lifecycle_wrapper.md` | `natal/hooks/compiler.py`（4 处） | `engine/lifecycle_wrappers.py` |
| `docs/zh/spatial_lifecycle_wrapper.md` | 同上（4 处） | 同上 |
| `docs/en/caching_and_codegen.md` | `natal/hooks/compiler.py`, `natal.hooks.executor` | `compile/codegen.py`, `runtime/csr_kernel.py` |
| `docs/zh/caching_and_codegen.md` | 同上 | 同上 |
| `docs/zh/spatial_builder.md` | `hook_executor` | `runtime/fallback` |
| `docs/en/spatial_builder.md` | 同上 | 同上 |
| `docs/zh/spatial_configurator.md` | `hook_executor` | `runtime/fallback` |
| `docs/en/spatial_configurator.md` | 同上 | 同上 |

**注意**：`compiler.py` 引用实为 **PR #9**（`lifecycle_wrappers` 拆分）时引入的遗留，非本分支新增。

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


## 🟢 低优先级 — UX / 远期功能

### #12 ⚠️ Spatial API：migration kernel 边界效应优化

**此分支改动**：三项子任务中，`batch_setting()`、`deme_selector` 局部 hook 在 main 上已完成；`pop.update()` / `_SpatialUpdate` / clone-on-write 是本分支实现（见已完成附录）。仅剩 migration kernel 边界效应优化未做。

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

### #18 🎨 基因型及配子索引压缩

**来源**：2026-06-21 设计讨论。用户需求：母本/父本对称性压缩（n² → n(n+1)/2）、修饰器可达性闭环分析、完全连锁配子空间压缩。

**优先级理由**：🟡 架构增强。offspring_tensor 三次方压缩（单 locus k=10: 100³ → 55³，6× 内存缩减）。对大型基因组模拟有显著空间和时间收益。但需要仔细处理兼容性（hook、observation、preset 均通过 index 访问状态）。

**设计方案**：

1. **压缩模式**：`NONE`（当前 dense）| `MATERNAL_PATERNAL`（合并 A|a ≡ a|A）| `REACHABLE`（BFS 可达性分析）| `FULL_LINKAGE`（完全连锁配子压缩）| `AUTO`（自动检测）。

2. **可达性分析（BFS）**：从 `initial_individual_count > 0` 的基因型 + `declared_genotypes`（手动声明）出发，通过 `offspring_tensor[gf, gm, go] > 0` 边 BFS 到不动点。保守分析（假设所有 (gf, gm) 对都可共存）。

3. **配置入口**：`Configurator.setup(compress_indices=True)` 或 `compression_mode="mp"`。支持 `declared_genotypes=["A|A", "A|a"]` 手动强制保留。

4. **实现路径**（~590 行，6 文件）：
   - `index_registry.py`：实现 `compact()` + `CompressionMap` 数据类 + `compute_genotype_reachability()` + `compute_gamete_reachability()`（~180 行）
   - `population_config.py`：新增 `n_genotypes_compressed`、`compress_map` 等字段 + `compress_population_config()`（~140 行）
   - `configurator.py`：build 管线集成（~30 行）
   - `engine/simulation/*.py`：确保引擎正确处理压缩维度（~40 行）
   - 测试：可达性 BFS 单元 + 端到端压缩 + 对称性验证（~200 行）

**不改**：Hook 签名、Observation API、Preset 系统（均通过名称/选择器访问，不直接依赖 index）。

### #19 🎨 Somatic Label (slab) + 基因型压缩 → 统一 EffectiveGenotypeSpace

**来源**：2026-06-21 设计讨论。用户需求：对称于 gamete label 的个体级标记系统。**关键洞察**（用户提出）：引擎对 index 完全透明——`g = int(fert_f.shape[0])`，`for go in range(g)`。压缩和 slab 在引擎看来只是 `G_total = G_comp × n_slabs` 的基数变化，两个变换正交组合。统一设计节省约 52% 代码量（975 vs 2,040 行）。

**优先级理由**：🟡 架构增强。与 #18 基因型压缩共用 GenotypeSpace 基础设施。默认 n_slabs=1 + 无压缩 → 零行为变化。

**优先级理由**：🟡 重大架构变更。牵涉 12+ 文件、~1,450 行。建议推迟到 v0.4.0。默认 `n_slabs=1` 时零破坏、零性能损失。

**设计方案**：

1. **维度扩展**：`individual_count: (2, A, G) → (2, A, G, S)`。`sperm_storage: (A, G, G) → (A, G, S, G)`（保留雌性 slab，雄性 donor 可选保留）。`viability_fitness`、`fecundity_fitness` 等 fitness 数组也扩展 slab 维度。

2. **三类 Slab 转换**：
   - `T_zygotic`（glab → slab）：受精时，给定母本 glab、父本 glab、合子基因型 → 子代 slab 分布
   - `T_gametic`（slab → glab）：减数分裂时，给定个体 slab、基因型、性别 → 配子 glab 分布
   - `T_somatic`（slab → slab）：每 tick 存活阶段，个体 slab 转换（如 Cas9 表达衰退）

3. **Slab 压缩**：与 #18 基��型压缩同模式——计算 slab 可达闭包 → 构建 compress_map → 重塑数组。

4. **实现路径**（~1,450 行，12+ 文件）：
   - Phase 1（数据结构）：Species 新增 `somatic_labels`、IndexRegistry 新增 slab 索引、PopulationConfig 新增 `n_slabs`/转换矩阵、PopulationState 重塑（~200 行）
   - Phase 2（引擎适配）：所有 `@njit_switch` 函数 + 4 个模板文件更新索引循环（~500 行）
   - Phase 3（上层 API）：Configurator `.somatic_labels()`、修饰器 slab-aware、Hook 适配、Preset 适配（~400 行）
   - Phase 4（压缩 + 测试）：`compute_slab_reachability()` + `compress_slab_dimension()` + 全覆盖（~350 行）

5. **向后兼容**：默认 `n_slabs=1`，引擎 `if n_slabs == 1: skip slab loop` 避免性能损失。所有现有测试无修改通过。

### #20 🎨 仿真引擎性能优化审计

**来源**：2026-06-21 架构审计。对引擎热路径的 6 个优化点进行系统评估。

**优先级理由**：🟡 性能工程。4 个纳入 v0.3.0，2 个推迟。均为纯优化，不改行为。

**纳入 v0.3.0 的优化**：

| ID | 优化 | 难度 | 预期收益 | 行数 |
|----|------|------|---------|------|
| #A | offspring_tensor Numba 化 | 低 | 3-5×（modifier 变更时） | ~100 |
| #B | CSR prange 并行化 | 中 | 2-4×（G≥200 时，per-op 内 genotype 维度并行） | ~120 |
| #C | 交配矩阵缓存 | 低 | ~30% 交配计算开销 | ~80 |
| #D | 内存分配复用（TickBuffers） | 中 | 减少 40-60% 分配调用 | ~200 |

**推迟的优化**：
- #E（deme 间负载均衡）：仅在 deme 间个体数差异 >10× 时有意义，大多数均匀场景无收益。
- #F（观测录制路径统一）：与 TODO #3 重复，维护收益 > 性能收益。

### #21 🎨 离散世代极速模式（Wright-Fisher）

**来源**：2026-06-21 设计讨论。用户需求：离散世代模型的极速模式——每 tick 单次多项分布抽样替代逐步模拟（mate → fertilize → survive），建模有效种群大小。

**优先级理由**：🟡 新模式。实现成本最低（~660 行），与现有完整模式完全解耦。计算量降低 10-100×。

**设计方案**：

1. **三种采样模式**：
   - `DETERMINISTIC`：无限种群极限，无随机性。适用于平衡分析、参数扫描。
   - `MULTINOMIAL`：标准 Wright-Fisher 单次多项分布抽样。适用于群体遗传学标准建模。
   - `POISSON`：独立泊松抽样。适用于极大 N（>10⁵），比多项分布快 ~2×。

2. **核心计算**：
   ```
   p[go] = Σ_{gf,gm} freq_f[gf] · freq_m[gm] · sexual_selection[gf,gm]
           · offspring_tensor[gf, gm, go] · eggs · sex_ratio
   new_count = Multinomial(N_eff, p)
   ```
   跳过：交配对抽样、受精抽样、存活抽样（均合并到 p 的权重中）。

3. **随机性差异**：极速模式轻微低估方差（缺少交配阶段的额外二项抽样），但差异 <5%。此时 N 的含义从"普查种群大小"变为"有效种群大小"——这是群体遗传学标准做法。文档需明确说明。

4. **限制**：仅支持离散世代模型。不支持精子置换、`fixed_egg_count=True`（可实现）、性染色体（可实现）。Hook 兼容性有限（per-individual kill/add 无意义）。

5. **实现路径**（~660 行，7 文件）：
   - `engine/simulation/discrete_generation.py`：新增 `compute_expected_offspring_wf()` @njit 函数（~80 行）
   - `engine/discrete_generation_simulator.py`：新增 `run_extreme_speed_tick()`（~60 行）
   - `population_config.py`：添加 `extreme_speed_mode` 和 `extreme_speed_N_eff`（~20 行）
   - `discrete_generation_population.py`：`run()` 中检测极速模式分支（~40 行）
   - `engine/templates/`：新增 `lifecycle_extreme_speed.tmpl.py`（~120 行）
   - `engine/lifecycle_wrappers.py`：编译管线扩展（~40 行）
   - 测试：中性等价性 + 选择场景 + 边界条件（~300 行）

6. **API**：
   ```python
   cfg = Configurator.for_discrete(species).setup(
       extreme_speed=True,
       extreme_speed_mode="multinomial",
   ).build()
   # 或运行时切换
   pop.enable_extreme_speed(mode="multinomial")
   ```

### 远期功能

- Global hooks
- Sparse（import / states）
- ✅ **Hook 系统分层重构** — PR #9（拆分层级）+ PR #11（命名 + 子目录 + 模板化）已完成。最终结构：`hooks/entry/`（装饰器/声明式/selector）、`hooks/compile/`（容器/codegen）、`hooks/runtime/`（CSR 内核/Python 回退）。详见已完成附录。

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

## 已完成附录

> 以下条目已实现，从活跃 TODO 中移除。按完成来源分组。

### 本分支完成

- **ZType 注册表重构（第一阶段）** — `feat/ztype-registry` 分支。将分散在 15 个文件、60+ 处的 `g·n_slabs + slab` 算术公式替换为扁平字典 ZType/GType 索引空间。新增 `_ztype_to_index`、`_gtype_to_index` 字典，`ztype_index()`、`gtype_index()` 方法，计算属性（`genotype_to_index`、`index_to_genotype` 等），扁平掩码压缩（`compress(ztype_mask, gtype_mask)` 无 `n_slabs` 参数）。删除 `compress_hg_glab`、`compress_genotype_index`、`decompress_genotype_index`、`axis_sizes`、`update_n_ztypes`。Hook `_resolve_genotypes` 修复为使用 ZType 索引展开。Oracle 验证通过。
- **ZType 全量修复（第二阶段）** — 系统性修复 `genotype_to_index` 在使用 n_slabs>1 时的 50+ 处静默错误。所有 pattern 字符串解析统一走 `ZygoteTypePattern`。删除 `genotype_to_index`、`genotype_index()`、`_ensure_genotype_registered()`、`_UnorderedGenotypeDict`。修复观察系统在 n_slabs>1 时的崩溃（mask 维度 `n_genotypes` → `n_ztypes`）。7 个 n_slabs>1 回归测试。参数重命名 `genotype_idx` → `ztype_idx`。状态：997 passed，pyright 0，ruff clean。

- **refresh 系统重构** — `rebuild_from_presets()` 拆为 `refresh_modifiers()`（public，仅 modifier 重建）+ `_reapply_preset_fitness()`（private，fitness 重置和重应用）。删除了 `refresh_modifier_maps()` public wrapper。`add_gamete_modifier` / `add_zygote_modifier` 的 `refresh` 参数现在只控制是否立即重建 maps，派生列表写入无条件发生。修复了 `rebuild_from_presets` 静默覆盖手动 fitness 的问题——`refresh_modifiers()` 不碰 fitness，只有 `apply_preset()` / `presets()` / `reconfigure_preset()` 会调用 `_reapply_preset_fitness()`。
- **#2** `expected_num_new_adult_females` — 旧机制（`base_expected_num_new_adult_females` + `get_effective_expected_adult_females()`）已全部移除。新机制通过 `Configurator.competition()` 接收参数，流经 `_compute_carrying_capacity_params()` 转换为 `external_expected_eggs`。两个专用测试验证。
- **#8 部分** `pop.update()` / `_SpatialUpdate` — 空间模型的运行时 config 修改 API，clone-on-write 语义，`test_spatial_update.py` 覆盖。
- **`parameters.jsonc` alias 修复** — `eggs_per_female` 的 alias 从冗余同名修正为 `expected_eggs_per_female`（保留向后兼容）。

### main 上已完成（本分支分歧前）

- **#3** Spatial 并行 — `prange` per-deme 并行分发（main: `7e4266a`）。非并行条件仅剩 Numba 禁用或 legacy Python hook。
- **#8 部分** `batch_setting()` + `deme_selector` 局部 hook — `spatial_builder.py` 中已有的 `BatchSetting` 类和 `set_hook(deme_selector=...)`。
- **#9** Spatial UI — `spatial_dashboard.py`（75KB，198 方法），热图渲染、deme config 信息、local hooks 显示、landscape genotype freq。
- **#10** General UI — Observation 集成 + UI 导出。本分支仅做兼容适配。

### `refactor/hooks-naming` 分支完成（PR #9 后续）

- **#1** 混用 CSR + njit Hook priority 调度 — `feat/unified-hook-priority-dispatch` 已实现（PR #8 合并），生成统一 njit 函数按 priority 交错执行。
- **Hook 系统分层重构** — `CompiledEventHooks` 拆分为纯容器 + `LifecycleWrappers`（engine 层）+ codegen 管线分离；`executor.py` 拆分为 `csr_kernel.py`（CSR 热循环）+ `fallback.py`（Python 回退）。
- **hooks 命名 + 目录重组** — `compiler.py` 拆为 `entry/decorator.py` + `compile/container.py` + `compile/codegen.py`；重命名为 `entry/` `compile/` `runtime/` 三个子包。
- **compile_combined_hook 模板化** — 从 50 行手拼字符串改为 `PLACEHOLDER_` + `str.replace` + `setattr` 模板驱动，与 `compile_unified_event_hook` 风格一致。
- **CLAUDE.md** — 新增三条项目规范（AskUserQuestion 选择题、Tasks 列表维护、优先使用专用工具）。

### `refactor/selector-hook-calling-convention` 分支完成（当前分支）

- **#2** Selector + custom hook 调用约定统一 — Numba 和 Python 路径统一为 `(state, config, deme_id=-1, selector_kwargs...)`。Custom hook 新增 `ind_count` 首参数检测，自动 wrapping 映射 `(state, config, deme_id)` → `(state.individual_count, state.n_tick, ...)`。修复了 `test_hook_priority_mixed.py`（2 测试）和 `test_spatial_population_run.py`（2 测试）的 Python fallback 崩溃。
- **#8** Selector hook 测试增强 — 25 测试按 `@pytest.mark.numba_on` / `numba_off` 分区。`TestSelectorResolution` 验证 `desc.selectors` 解析值；`TestPythonFallbackEndToEnd` 实际调用 `py_wrapper`；Numba 路径新增 `deme_id` 转发和多基因型选择器测试。
- **模板化** — `_compile_selector_njit_wrapper` 从内联字符串拼接改为 `selector_wrapper.tmpl.py` 模板驱动。
- **类型规范** — `selector.py` + `decorator.py` 消除 `Any`/`object` 滥用，改用 `PopulationState`、`PopulationConfig`、`int | NDArray[np.int32]` 等具体类型。CLAUDE.md 新增"禁止滥用 Any 和 object"规则。
- **Test marker 修复** — `test_lifecycle_wrappers.py` 中 2 个测试从无标记改为 `@pytest.mark.numba_on`。
- **发现 #17** — `NATAL_DISABLE_NUMBA=1` 全量运行发现 9 个既有失败。修复 6 个（hook 约定 + marker），剩余 3 个为 `_replace()` 0-d ndarray 退化问题。

---

## Conversion Ruleset 重构（2026-07-10 grill session）

### ✅ Stage 1 完成 — DSL 基础 + 特判消除

**已完成**（`feature/spatial-compress-unified` 分支）：

1. **Condition DSL**（`src/natal/modifiers/conditions.py`）：
   - `sex()`、`ztype_has()`、`slab()`、`is_maternal()`、`is_paternal()` + `&`/`|` 组合
2. **新 API**：
   - `GameteRuleSet.add_glab_convert(from_glab, to_glab, rate, when=...)` — 替代 `add_hg_convert(hg→hg, target_glab=...)` 的语义拐弯
   - `ZygoteRuleSet.add_glab_redirect(from_glab, to_glab, when=...)` — zygote-level glab redirect
   - 旧 API 保留，新 API 内部委托到旧 API
3. **Preset 迁移**：
   - HomingDrive + ToxinAntidote：Cas9 沉积标注从 `add_hg_convert` → `add_glab_convert`
   - CytoplasmicPreset：走 modifier 协议，`isinstance` 特判消除
4. **ztype/gtype 适配**：
   - `gamete_conversion.py`：`index_to_genotype` → `index_to_ztype`，`genotype_idx` → `ztype_idx`
   - `zygote_conversion.py`：同上 + `g = row.argmax()` 后直接用 ztype 索引

### 📋 Stage 2 待做 — 剩余迁移

1. ✅ `RuleSet.to_matrix(registry)` — 已实现（`1f06109`），CytoplasmicPreset 已在用
2. ✅ `when` 条件接线 — 已实现（`8d70791`），gamete/zygote modifier 均已支持
3. `add_slab_convert` — gamete/zygote 端 slab 操作（当前 CytoplasmicPreset 自制循环）
4. `_compute_converted_gamete_freqs` 替换为矩阵乘法
5. `extract_gamete_frequencies_by_glab` 调用精简（CytoplasmicPreset 路径仍保留）

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
