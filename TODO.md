# TODO

> 基于 `refactor/config-architecture` 分支审计结果重新排序（2026-06-01）。
>
> 排序逻辑：正确性 bug > 性能优化 > UX 改进 > 代码质量。同一档内，部分完成 > 未开始 > 仅设计。
>
> 状态标记：
> - ✅ DONE — 已实现
> - ⚠️ PARTIAL — 部分实现，有遗留问题
> - 📋 NOT_DONE — 未实现
> - 🎨 DESIGN_ONLY — 仅有设计方案，无实现

---

## 🔴 高优先级 — 正确性 / 阻塞项

### #1 ✅ 混用 CSR + njit Hook 时的 priority 跨类型比较与性能矛盾

**此分支改动**：`feat/unified-hook-priority-dispatch` 已实现根本修复。生成统一的 njit 函数 per event（`compile_unified_event_hook`），在 Numba JIT 内部按 priority 交错执行 CSR 和 njit hook。模板结构不变——过滤后的 registry 让 CSR 调用变为 no-op，统一函数嵌入 `_HOOK` 调用。`should_use_python_dispatch()` 不再因混用回退 Python 路径。

**实现要点**：
- `executor.py`: 提取 `_execute_single_csr_hook`——从 HookProgram 中按 hook_idx 执行单个 CSR hook，供统一调度使用
- `compiler.py`: 新增 `compile_unified_event_hook`（代码生成）+ `_build_filtered_hook_program`（过滤 registry）；重写 `from_compiled_hooks` 检测混用事件并生成两份统一函数（with_sperm / without_sperm）
- `base_population.py`: `should_use_python_dispatch` 移除 `has_mixed_hook_types()` 检查
- Spatial 模型自动受益——混用不再产生错误优先级
- 测试：4 个 Numba 开启下的混用优先级排序测试全部通过

### #2 ⚠️ Selector hook 调用约定不一致

**此分支改动**：Wrapper 的外部接收签名已现代化为 `(state: StateType, config, deme_id=-1)`，`_normalize_njit_fn` 和 `_normalize_py_hook` 也已更新。但 wrapper 内部**转发给用户函数的参数仍未修正**——详见下方。

**优先级理由**：🔴 正确性 bug。Numba 路径和 Python 路径传给用户函数的参数完全不同，同一用户函数无法在两个路径下工作。Python 路径（`numba_disabled()` 下触发）从未被测试覆盖，实际必定崩溃。

- **Numba 路径**（`_compile_selector_njit_wrapper`）：wrapper 自身签名是 `(state, config, deme_id=-1)` ✅，但内部提取 `ind_count`/`tick` 后传 `(ind_count, tick, ...)` 给用户函数 ❌，`config` 未转发
- **Python 路径**（`compile_selector_hook`，第 172-178 行）：`py_wrapper` 将整个 `population: BasePopulation` 对象传给用户函数 ❌，既不是旧约定也不是新约定
- 两个路径传给用户函数的参数完全不同，同一用户函数无法在两个路径下工作
- 当前测试只验证 `py_wrapper is not None`，从未真正调用 Python 路径 → bug 未被发现
- 修复：统一为用户函数接收 `(state, config, deme_id)`，selector 值和 `deme_id` 作为 kwargs 附加。注意这是 breaking change。
- 详见 memory: [[selector-hook-calling-convention-bug]]

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

**此分支改动**：新增了声明式配置适配层（`_ConfigContext`），为修饰器提供了运行时上下文绑定，是矩阵化路径的前置依赖。设计了详细的实现方案（`matrix-modifier-design.html`），但 `compile_matrix()`、`to_matrix()`、`ModifierMatrix`、`apply_transition()` 均未实现。Cascade 仍使用 `Dict[Genotype, float]` 逐 rule 迭代。

**优先级理由**：🟡 已证明的数学基础 + 明确实现路径（~130 行，3 文件），g=100、hl=8、3 modifier 场景下 offspring_tensor 加速 5000x。同时支持 reconfiguration 增量更新，对大型模拟有显著性能价值。但前提是 #1 正确性 bug 完成后才应开始。

**数学基础**（已证明）：

1. **每条 rule 是线性算子**：`_compute_converted_gamete_freqs` 对频率向量 v ∈ ℝᴰ
   只做比例分割（v'[hg] = v[hg]·(1−r), v'[converted_hg] += v[hg]·r）。
   无归一化、无阈值、无非线性步骤。

2. **Cascade ≡ 矩阵乘积**：`M_total = Rₖ · ... · R₂ · R₁`，其中每 Rᵢ 是 D×D
   稀疏矩阵（D = n_hg × n_glabs，通常 10-100）。级联合成 = 矩阵乘法。

3. **可交换条件**：Mᴬ·Mᴮ = Mᴮ·Mᴬ ⟺ affected_A ∩ affected_B = ∅ 或
   from_allele_A ≠ from_allele_B。RuleSet API 下 from_allele 静态声明，
   编译期即可检测。

4. **genotype_filter**：filter=False 的 genotype → 对应矩阵 = I（单位矩阵），
   不改动该行。ModifierMatrix = {g ∈ affected: M_total, g ∉ affected: I}。

**实现路径**（~130 行，3 文件，纯加法，不改现有 API）：

| 文件 | 改动 |
|---|---|
| gamete_allele_conversion.py | 每 rule 加 compile_matrix() → D×D 稀疏矩阵；加 to_matrix() 编译 RuleSet 为 ModifierMatrix |
| configurator.py | _rebuild_config_maps 加 if/else 分发：全部 RuleSet → 矩阵路径；否则 → 现有回调路径 |
| modifiers.py | 加 apply_transition() 逐行 @ 矩阵；_apply_comp_map 保留作为回调路径基础设施 |

**不改**：genetic_presets.py、任何 Numba 内核、population_config.py、initialize_gamete_map。

**reconfigure 增量**：只重编译被改 preset 的矩阵（纯算术），重新合成 M_total
（2 次矩阵乘法），应用到 affected_rows。不重跑其他 modifier。

**指标**：g=100, hl=8, 3 modifier, |affected|=2 → offspring_tensor 2500× 加速。
详见 `matrix-modifier-design.html`。

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

### #8 📋 Selector hook 测试增强

**来源**：`code-quality-review-report.html` #12, #15

**此分支改动**：无。12 个 selector 测试中 11 个仅断言 `py_wrapper is not None`，从未真正调用 Python 执行路径。

**优先级理由**：🟡 修复 #2 后需要验证两个路径的行为一致性。当前测试覆盖不足，无法防止回归。

- `test_hook_selector_mode.py`：验证 `desc.selectors` 解析值而非仅判非空
- 在 #12 bug 修复后，添加调用 `desc.py_wrapper(pop)` 的测试验证 Python 路径功能
- 添加多基因型索引选择器的 Python 路径测试
- 注意：修复 #2 是 #8 的前提条件

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

---

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

---

## 已完成附录

> 以下条目已实现，从活跃 TODO 中移除。按完成来源分组。

### 本分支完成

- **refresh 系统重构** — `rebuild_from_presets()` 拆为 `refresh_modifiers()`（public，仅 modifier 重建）+ `_reapply_preset_fitness()`（private，fitness 重置和重应用）。删除了 `refresh_modifier_maps()` public wrapper。`add_gamete_modifier` / `add_zygote_modifier` 的 `refresh` 参数现在只控制是否立即重建 maps，派生列表写入无条件发生。修复了 `rebuild_from_presets` 静默覆盖手动 fitness 的问题——`refresh_modifiers()` 不碰 fitness，只有 `apply_preset()` / `presets()` / `reconfigure_preset()` 会调用 `_reapply_preset_fitness()`。
- **#2** `expected_num_adult_females` — 旧机制（`base_expected_num_adult_females` + `get_effective_expected_adult_females()`）已全部移除。新机制通过 `Configurator.competition()` 接收参数，流经 `_compute_carrying_capacity_params()` 转换为 `external_expected_eggs`。两个专用测试验证。
- **#8 部分** `pop.update()` / `_SpatialUpdate` — 空间模型的运行时 config 修改 API，clone-on-write 语义，`test_spatial_update.py` 覆盖。
- **`parameters.jsonc` alias 修复** — `eggs_per_female` 的 alias 从冗余同名修正为 `expected_eggs_per_female`（保留向后兼容）。

### main 上已完成（本分支分歧前）

- **#3** Spatial 并行 — `prange` per-deme 并行分发（main: `7e4266a`）。非并行条件仅剩 Numba 禁用或 legacy Python hook。
- **#8 部分** `batch_setting()` + `deme_selector` 局部 hook — `spatial_builder.py` 中已有的 `BatchSetting` 类和 `set_hook(deme_selector=...)`。
- **#9** Spatial UI — `spatial_dashboard.py`（75KB，198 方法），热图渲染、deme config 信息、local hooks 显示、landscape genotype freq。
- **#10** General UI — Observation 集成 + UI 导出。本分支仅做兼容适配。
