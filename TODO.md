# TODO

## 本轮审查方案汇总（2026-09-13）

整合 architecture-deep-dive.html 及后续核验。**当前只记录方案，所有 CR 项均未在本轮实施；确认设计不等于代码已修复。** 下方测试结果是调查时的历史自测，不能当作新合同验收；临时复现脚本可能不再存在，实施时转为持久回归测试。源码行号为调查定位，后续可能漂移。

| 编号 | 最终范围 | 状态 |
|---|---|---|
| CR-0 | XY/ZW 解析、字符串化、精确初始化与性别约束 | ✅ 已实施（2026-09-13），待最终审查 |
| CR-1 | 四类规则、统一 filters/to、联合概率与声明顺序 | ✅ 已实施并获终审 APPROVED（2026-09-13） |
| CR-2 | 配子修饰器非法键报错，先验证再应用 | ✅ 已实施（2026-09-13） |
| CR-3 | 仅 first/early/late/finish，拒绝非法事件 | ✅ 已实施（2026-09-13） |
| CR-4 | 配置快照尺寸失配报错 | ✅ 已实施（2026-09-13） |
| CR-5 | 取模除数为正整数，解析时拒绝零 | ✅ 已实施（2026-09-13） |
| CR-6 | 注释、乱码、遗留与不可达代码、过时引用 | ✅ 已实施（2026-09-13） |
| CR-7 | 删除配子缓存，统一基线，内容快照失效 | ✅ 已实施（2026-09-13）；规则基线统一并入 CR-1 引擎 |
| CR-8 | 物种与实体缓存生命周期重构 | 明确暂缓 |
| CR-9 | 计算前校验遗传结构完整性 | ✅ 已实施（2026-09-13） |
| CR-10 | XY/ZW 同源区段交换 | 明确不在本轮实现 |
| CR-11 | 删除重复且被覆盖的 genotype 缓存键计算 | ✅ 已实施（2026-09-13） |
| CR-12 | TickMetrics 索引直查，类型名称统一 @ | ✅ 已实施并获终审 APPROVED（2026-09-13） |
| CR-13 | WF 性别分配归一化与总量守恒 | ✅ 已实施（2026-09-13） |
| CR-14 | 计数与规则统一携带年龄轴（含两处归一化收敛） | ✅ 已实施并获独立审查 APPROVED（2026-09-14） |

**实施依赖与验证边界**

- CR-0/9/13 联合验证：仅修公开性染色体标志会暴露 WF 翻倍，不将单点修改当作完整修复。
- CR-1/12 同步迁移，不兼容旧规则接口和冒号标签格式；既有无序模式 ::、空间日志冒号含义不变。
- CR-7 提供未修饰基线，CR-1 刷新验收同时检查来源、概率不重复叠加、索引对齐与对象隔离；CR-8 不随此项扩大实施。
- 实施遵循 AGENTS.md 和质量规范：基本验证、相关 stub/中英文文档/示例同步、高风险独立 evaluator 审查及最终门禁。本次仅文档整理，不运行或声称新合同代码验证。

## CR-0 ✅ DONE — 性染色体公开路径修复（2026-09-13 实施，2026-09-14 复核一致）

> **状态复核（2026-09-14）**：与上表状态一致——列出的子因均已随 `33236e7` 与 `370adf5`（D1）修复：`builder/_base.py:607` 从 `species.get_sex_chromosome_groups()` 取标志并经 blueprint 转发掩码；`genetics/structures/_construction.py` 三处改用 `get_sex_chromosome_groups()` 方法；`genetics/entities/genotype.py:399-411` 按性染色体组拼接；`model/initial_state.py:144-157` 的 Genotype 键不再转字符串；`registry/index.py:318-333` 为 O(1) 精确查找。唯一仍在的旧写法是 `patterns/elements/diploid.py:129-148` 的 `from_pair` 内部 `str(genotype)` 再解析——它是构造匹配模式的既有路径，若将来发现具体缺陷再单开条目。

**已核验的问题链**

以下路径除注明外相对 src/natal/frontend/。

- genetics/entities/genotype.py:402 按每条染色体同时取母父 haplotype，缺一侧就跳过；XY/ZW 异型对丢失性染色体部分，纯性染色体对象可得到空串。
- genetics/structures/_construction.py:216,291 读取未初始化的 sex_chromosome_groups 属性而非已有 get_sex_chromosome_groups 方法，合法 XY 字符串被按错误段数拒绝。
- model/initial_state.py:108 将 Genotype 对象转字符串再解析模式；patterns/elements/diploid.py:138 的 from_pair 也内部转字符串。registry/index.py:324 取首个匹配项，空串/宽泛模式使精确初始化落错类型；报告建议仅调用 from_pair 不足以修复。
- builder/_base.py:600 读取不存在的 has_sex_chromosomes 并回退 False；开启标志后，assembly 从配子行和推断性别 mask 也不充分，见 CR-13。
- 前期公开路径复现 XY 雄性对象初始化落到 XX 类型、XY 字符串解析失败。内部手工开启标志的测试通过不能替代公开入口覆盖。

**修复边界**

- 性别系统与性染色体分组使用一致来源；字符串化、完整解析、模式匹配按性染色体组处理，保留母父相位；覆盖 XY/ZW 以及混合常染色体的往返。
- 精确对象、精确字符串和显式 slab 初始化按 registry 精确身份定位；对象入口不先转字符串再匹配，合法模式选择与精确初始化分开，避免静默选择首个类型。
- XX/XY、ZW/ZZ 性别约束来自遗传结构，不能由配子行和推断；年龄结构、分阶段离散代与 WF 遵守同一概率定义，保留各自抽样方式。
- 按 CR-9 拒绝计算时无位点的染色体，不新增空 Y/W 字符串占位语法或自动虚拟等位基因；按 CR-10 不实现异型性染色体同源区段交换。
- 验收通过公开 builder，覆盖对象/字符串/标签输入、非 0.5 sex_ratio、XY/ZW、压缩与 slab 扩展、初始与后代性别/类型一致性、往返及总量守恒。辅助方法名称在实施时确定。

## CR-1 📋 Conversion rules 统一接口与执行语义（方案已确认，未实施）

**四类 API**

采用关键字参数。rate 必填，filters=None 表示不限制，name=None 仅用于展示。

| 类别 | 必填字段 | 可选字段 | 动作 |
|---|---|---|---|
| GameteGtypeConversionRule | to: str、rate: float | filters、name | 一次事件转换完整 gtype |
| ZygoteZtypeConversionRule | to: str、rate: float | filters、name | 一次事件转换完整 ztype |
| GameteAlleleConversionRule | from_allele: str、to_allele: str、rate: float | filters、name | 局部替换，不改变 glab |
| ZygoteAlleleConversionRule | from_allele: str、to_allele: str、rate: float | filters、name、side | 局部替换，不改变 slab |

- filters 类型为 Mapping[str, str] 或 None；name 为 str 或 None。side 为 maternal/paternal/both，默认 both，表示合子遗传副本而非亲本个体，两侧独立转换。
- 不要求 locus：Gene 名在同一 Species 内唯一（src/natal/frontend/genetics/entities/gene.py:98），由源等位基因定位位点，并验证目标等位基因属于同一位点。
- GameteGlabConversionRule 并入 Gtype 转换；ZygoteGlabRedirectRule 并入 Ztype 转换，原母源 glab 是来源条件，目标 slab 是动作。
- 新接口不兼容旧规则接口，不保留旧标签类、旧参数、旧别名或对象/callable 输入的兼容包装；同步迁移规则相关公开导出、stub、预设、文档、示例与测试，不扩大为删除全库无关历史 API。
- 现有 GameteAllele.target_glab 的成功联动模型迁移为完整 Gtype 联合转换，不能机械拆成两个独立事件。

**统一 filters**

普通字符串字典，复用既有类型模式，不新增条件 DSL；同一阶段的整体/Allele 规则支持相同的键。

| 键 | 配子规则 | 合子规则 |
|---|---|---|
| current | 当前分支 gtype 模式 | 当前分支 ztype 模式 |
| parent | 产生配子的亲本 ztype 模式 | 不支持 |
| parent_sex | female/male/both | 不支持 |
| maternal | 不支持 | 形成合子的母源配子 gtype 模式 |
| paternal | 不支持 | 形成合子的父源配子 gtype 模式 |

- 各键之间为 AND，省略表示不限制；类型模式用 @ 限定标签，裸遗传组成模式不限制标签，@default 明确限定默认标签。
- current 检查进入本条规则的分支状态；其他键检查固定来源。此前“合子 when 检查当前状态”的要求由 filters["current"] 承担，不另设 when 或 Condition 对象接口。
- 未知键、拼写错误、阶段不支持的键、非法模式均在编译时显式报错，不解释为匹配失败。

**目标、概率与顺序**

- 整体转换 to 必须为 `[genotype 或 *]@[label 或 *]`，配子侧为单倍体遗传组成。两部分显式给出，整部分 * 保留输入对应部分，具体值精确替换；不支持多候选目标或遗传组成内部的局部通配替换。
- 合子 `A|B@I` 联合替换两部分，`*@I` 只改 slab，`A|B@*` 只改 genotype；`*@*` 为恒等转换，不额外禁止。
- rate 必须有限且在 [0, 1]。整体转换成功分支同时采用目标各部分，失败分支保持原状态；独立变化用两条规则表达，条件须覆盖相应分支，不增加 independent 开关。
- Allele 转换先检查分支条件，再对指定侧带源等位基因的副本独立以 rate 转换。合法输入未匹配源正常保持原状态；未知源/目标、非法标签或跨位点替换显式报错。
- RuleSet 只按声明/追加顺序级联，无数字 priority，不按类型排序，不在首次匹配后停止。便捷方法完整暴露对应字段，不固定 rate。
- 合子编译以 `(Genotype, slab) → probability` 跟踪联合分支，不再使用所有基因型共用的 effective_slab；实施时验证概率合并、压缩轴及目标可达性。

**证据与验收**

以下 modifiers 路径均位于 src/natal/frontend/。

- modifiers/zygote_conversion.py:439,617：redirect 的 rate=0/0.25/1 均整体重定向；便捷方法固定 rate=1，已复现。
- zygote_conversion.py:627、gamete_conversion.py:776,810：redirect 整数目标 1 被当作名称 "1"，未知目标无操作；配子负索引 -1 选择末标签。新规则取消整数索引输入，不保留此行为。
- zygote_conversion.py:586,601,636：genotype 条件检查当前分支，redirect when 检查 base_gt/base_slab；A→B 后 when=B 不匹配，已复现。conditions.py:144,228,250 的合子性别条件均假、母源/父源条件均真，由新的阶段限定 filters 取代。
- gamete_conversion.py:424、zygote_conversion.py:328,484 的首次匹配获胜文档错误；实际 A→B→C 得到 C。gamete_conversion.py:927、zygote_conversion.py:756 静态发现源存在而目标缺失时可能静默跳过。
- 历史自测：`.venv/bin/python -m pytest -q tests/test_modifiers.py tests/test_conditions.py tests/test_conversion_refresh_contracts.py` → **173 passed**，未覆盖全部缺陷；`/tmp/review_conversion_rules.py` 已执行，临时文件不保证保留。不是新接口验证结果。
- 实施验收覆盖 rate=0/1/中间值、联合转换、两侧独立转换、当前与来源条件、级联、恒等目标、非法输入、标签及压缩目录、刷新不重复叠加。执行高风险独立审查与最终门禁。

## CR-2 📋 自定义配子修饰器静默吞错进入仿真（2026-09-12）

- **公开路径已复现，尚未修复：** `DiscreteGenerationPopulation.setup(...).modifiers(gamete_modifiers=[...]).build()` 接受非法源/目标键；错误没有被最终构建校验拦住。
- 在确定性模式、100 个 A|A 雌性与 100 个 A|A 雄性、每雌性 1 个卵、无竞争的对照中：无修饰器产生 100 后代；非法源键被忽略仍为 100；`{"A|A": {"NOT_A_GAMETE": 1.0}}` 将配子行清零，产生 0；`{"A|A": {"A": 0.5, "NOT_A_GAMETE": 0.5}}` 留下行和 0.5，产生 25。四种构建均成功。
- 源码：`src/natal/frontend/modifiers/module.py:264` 先清零，268 行吞目标解析异常，397/425/432 行吞源解析等异常，443 行返回结果副本。原输入数组未被直接修改，但错误输出进入配置。
- 修复方向：明确输入键必须有效，先验证再替换；无效目标报错并定位声明。不得简单把残余概率归一化，也不能统一禁止零行（合法生物学模型可能需要零配子输出）。
- **用户已确认：** 明确无效的基因型、配子、标签或越界索引应显式报错；合法条件不匹配正常跳过，合法全零分布按模型合同处理。先验证整份输出再应用；构建遇错失败，运行时更新遇错保留此前有效配置。本轮只记录决定，尚未实施。
- 主 agent 执行 `.venv/bin/python /tmp/review_modifier_public.py` 得到上述结果；脚本在临时目录。本项未修复，未运行完整门禁或独立审查。

## CR-3 📋 统一事件支持范围并拒绝未知事件（2026-09-12）

- **用户已确认：** 不支持 `initialization`；未知或拼错的事件名必须显式报错。合法事件没有 callback 时仍正常返回继续。
- 允许事件统一为 `first / early / late / finish`，注册与手动触发入口使用同一事件目录。移除 `BasePopulation.ALLOWED_EVENTS` 中的 `initialization`，注册或手动触发它均应拒绝，不映射为 `first`。
- 手动触发应先校验事件名，再初始化会话或执行其他副作用；错误信息包含传入名称及合法名称。普通种群和空间种群入口保持一致。
- 源码：`src/natal/frontend/population/base.py:156,1751`、`src/natal/frontend/hooks/types.py:186`、`src/natal/frontend/hooks/tick_context.py:726`。现有 runner 会跳过无事件 ID 的 callback。
- 现有测试明确要求未知事件无操作、initialization callback 不执行；本次是已获用户确认的合同调整，后续将这些断言更新为拒绝非法事件，并保留合法空事件、正常 callback 和空间入口覆盖，不能只删除测试。
- 自测命令：`.venv/bin/python -m pytest -q tests/test_hooks_slice4_adversarial.py -k 'trigger_event_unknown_event_is_noop or runner_skips_non_tick_event_descriptors'` → **2 passed, 30 deselected**，仅证明旧行为。尚未实施修复，未运行新合同测试、完整门禁或独立审查。

## CR-4 📋 配置快照尺寸不一致时静默回退（2026-09-12）

- `src/natal/backends/rust/rust_backend.py:59` 仅在原生张量元素数量等于 draft 数量时覆盖字段；不等时，71 行后的补齐逻辑复制旧 draft 值，形成混合快照。
- 主 agent 用真实已构建种群的 session 读代理，仅将 `viability_fitness` 的读取替换为单元素 `[0.125]`：快照仍成功返回 draft 的 `(2,2,3)` 全 1 数组。这是故障注入复现，**不是正常公开操作可触发的证据**。
- 同一真实 session 直接写入错误尺寸被原生层拒绝：`ValueError: genetics viability_fitness: expected 12 elements, got 1`。尚未发现合法公开操作导致尺寸失配；不能据此宣称压缩或恢复已损坏。
- 影响入口：普通种群 `config` 读取、空间 deme 配置投影、hook 事务候选配置物化都复用该函数。
- **用户已确认（2026-09-13）：** 在预期一致的元素数量不匹配时显式报内部一致性错误，不回退旧值；错误包含字段名和预期/实际数量。合法空张量或特殊投影单独处理。原生读取通常为扁平数组，不应直接比较 Python shape。修复方案已确认，尚未修改实现。

## CR-5 📋 Hook 条件语法说明与零除数校验（2026-09-13）

- 已复现 `tick >= -5`、`tick >= threshold` 均明确报 ValueError；非负整数字面量限制本身不属于静默计算错误。`docs/en/2_hooks.md` 和 `docs/zh/2_hooks.md` 尚未明确 N 的范围及不支持变量引用。
- **新增公开路径复现：** `parse_condition("tick % 0 == 0")` 成功编译；将 `Op.scale(factor=0.0, when="tick % 0 == 0")` 注册到 early，build 和手动触发都成功，总数保持 200。对照 `when="tick >= 0"` 同一操作使总数从 200 变为 0。
- 原因：`src/natal/frontend/hooks/entry/declarative.py:609` 的数字解析接受 0；`rust/src/hooks/interpreter.rs:488` 使用 `cond_param > 0 && tick % cond_param == 0`，将零除数表达式解释为恒假，避免崩溃但掩盖非法条件。
- **用户已确认（2026-09-13）：** 取模除数必须为正整数，零除数在解析时显式报错，不解释为恒假。保持现有有限语法，并同步中英文文档说明这一限制。修复方案已确认，尚未修改实现。
- 主 agent 执行专项 Python 片段复现上述行为；`.venv/bin/python -m pytest -q tests/test_hook_condition_interpreter.py` → **20 passed**。未运行完整门禁或独立审查。

## CR-6 📋 轻量清理方案（2026-09-13，暂不实施）

- **用户已确认清理方向，随后明确暂不执行、只记录方案。** 本轮仅调查源码，没有修改以下实现文件。
- 修正 `src/natal/frontend/hooks/entry/declarative.py:1149,1234` 的性别 mask 注释为 `[female_selected, male_selected]`，不改变实际顺序。
- 修正 `src/natal/frontend/modifiers/gamete_conversion.py:339` 默认规则名中的 `â†’` 为 `→`。
- 清理 `src/natal/frontend/patterns/selector.py` 中已标记遗留且无外部调用的选择器方法；删除前核对公开导出、stub、文档和兼容范围，不能仅凭无仓库内调用就移除公开合同。保留在用的选择器与解析功能。
- 清理 `src/natal/frontend/patterns/parser.py` 中当前调用链不触发的 species 分支、`_parse_flexible_loci`、仅供其使用的 `_is_valid_gene_char`，以及无调用者的 `_are_all_genes_single_characters`；相应移除不再需要的私有参数和导入，保持当前有效解析行为。
- 更新或删除 `src/natal/frontend/registry/index.py:430` 对已删除 `natal.frontend.population_config` 模块的注释引用。
- 实施时按实际变更风险验证：正文注释做准确性检查；代码删除与默认名称修复做针对性验证和最终完整门禁；若涉及公开 API 移除，执行独立审查及相关合同、stub、文档同步。
- 本项仅负责轻量清理；缓存、重复键、指标索引与 WF 数值修复分别由 CR-7/8/11/12/13 跟踪。

## CR-7 📋 物种基线、配子缓存与内容快照失效（方案已确认，未实施）

**最终边界**

- 删除 Genotype._gamete_cache，produce_gametes 直接计算；不删除实体去重缓存，不扩大到 CR-8。
- 规则编译从 Species 的未修饰孟德尔基线开始：基线生成 → 当前 registry 轴投影 → 顺序应用规则 → 种群运行矩阵。配子 RuleSet 不再自行重复 initialize_gamete_map；合子侧遵守同一来源边界。
- 不得以已施加规则的运行矩阵作为刷新基线，不污染共享基线；同一声明反复编译不叠加转换。
- 基线内部持有；编译和种群获取隔离的投影/副本。重建替换缓存条目，不原地修改旧矩阵；修改 Species 不暗中更新已建种群。构建期间修改 Species 的检测边界在实施时核对，不声称支持并发修改。

**内容快照失效机制**

- 唯一基线获取入口保存并比较依赖内容快照；有效且相同则复用，变化则重建。复用前执行必要校验；重建失败显式报错，不回退旧缓存；仅在新基线完整构建成功后更新缓存及快照。
- 快照保存独立值而非共享视图，精确比较，不用 allclose 忽略微小变化。不仅比较数组身份、标签数量或 setter 版本号，也不为生成缓存键枚举全部 genotype。
- 覆盖有序染色体/位点/等位基因目录及身份、位点位置、性染色体定义、unordered、glab/slab 名称与顺序、重组图位点对应关系及数值。结构变化后下游目录/身份缓存有效性须验证，不能假定清基线即修复全部结构缓存。
- map setter、批量入口和共享数组视图写入都应被检测；不引入 ndarray 子类或写入代理，不以禁止视图写入替代用户要求。
- 惰性失效保证下次获取不复用旧结果，不要求数组写入时立刻置 None。通过视图写入的非法数值也要在使用前检查有限性、范围、长度及位点映射。
- 更新要求用户手动清 _gamete_cache 的文档和旧测试。clear_all_caches 不能被宣传为可靠的计算结果失效入口。

**源码、证据与验收**

- src/natal/frontend/genetics/entities/genotype.py:278 和 genetics/structures/_mapping.py:104 两层非空即返回；genetics/compile.py:77 复用物种基线。Chromosome.set_recombination 只写图，不使两层缓存失效。
- 双杂合 `A1/B1|A2/B2`：r=0.1 时亲本配子各 0.45、重组配子各 0.05；改 r=0.5 后直接查询仍旧。清单基因型缓存后直接查询各 0.25，但新建种群仍用 0.05；再清 config_blueprint 后新建种群才用 0.25。此操作仅用于定位，不是推荐用户 API。
- 无 preset 首次构建每个基因型 produce_gametes 一次；同 Species 再构建与运行一代均零次；一个 HomingDrive preset 首次构建部分基因型调用两次，第二次来自 gamete_conversion.py:636 重建基线。
- `/tmp/review_cache_snapshot.py` 已验证独立副本比较检测 setter、切片视图、np.asarray、np.copyto、ufunc out 五种写入；视图写入后当前基线仍返回同一对象。另行将 gamete_labels 设为 ['default', 'tagged']，缓存 n_glabs 仍为 1；现有基线矩阵可写。
- 单数组 np.array_equal 微基准：10/1000/100000 个 float64 约 0.6/0.9/24 微秒；只代表当时本机数组比较，不是完整快照或端到端性能。临时脚本不保证长期保留。
- 历史自测：`.venv/bin/python -m pytest -q tests/test_genetic_entities.py -k 'rate_change_requires_manual_cache_clear or two_loci_half_recombination_equal_quarters'` → **2 passed, 67 deselected**，证明旧行为。新合同未实施，未运行完整门禁或独立审查。
- 实施验收：未改依赖时命中；五种写法修改后更新；标签/结构变化不复用旧轴；非法输入不回退旧结果；刷新不叠加规则；新旧种群隔离；CR-9 完整性校验不能被缓存绕过。

## CR-8 📋 物种与实体缓存的生命周期（2026-09-13，暂缓）

- **用户决定暂缓：** 改动范围较大，本轮不调整缓存归属、清理接口或实体身份合同。以下方案仅供后续单独评估，不视为已批准实施；保留现状和核验证据。

- 区分 CR-7 的配子计算结果缓存与实体去重缓存：后者保证同一 Species 下相同遗传实体返回相同实例，不能简单随 `_gamete_cache` 一并删除。
- 主 agent 最小复现：一个双等位基因 Species 枚举后有 6 条 GeneticEntity 缓存、3 条 Genotype 缓存、1 条 pattern 缓存；调用 `species.clear_all_caches()` 后分别为 0、3、1。释放外部引用并 `gc.collect()` 后 weakref 仍存活，Genotype 全局字典仍持有 Species 键。
- 源码：`genetics/entities/_base.py:42,173,184` 全局字典保存实体、实体保存结构；`genetics/entities/genotype.py:61` 全局字典直接以 Species 为键；`patterns/parser.py:39,96` 类级缓存以 `(id(species), pattern)` 为键；`genetics/structures/species.py:195,200` 只清通用实体及结构缓存，未覆盖后两者。
- 已证实的是缓存清理覆盖不完整及对象被保留，未复现 `id()` 复用导致错误匹配，也未量化长期内存增长。对象存活可能还有其他强引用根，不把上述路径声称为全部保留来源。
- 建议以 Species 作为物种绑定实体及解析缓存的生命周期拥有者，保留实体唯一性；没有外部引用后由 GC 回收物种内部引用环，避免进程级字典永久持有 Species。实施前盘点全局 fallback 等其他引用根。
- 必须区分计算结果失效与身份目录清理；活跃 registry 可能依赖对象身份，不能把清理实体缓存当成普通参数更新。明确 `clear_all_caches()` 是何种操作，再同步合同与测试。
- 不建议仅替换成 WeakKeyDictionary：若值持有实体、实体反向持有 Species，仍可能阻止键回收。本项已按用户要求暂缓，未修改实现或运行完整门禁。

## CR-9 📋 遗传结构完整性校验边界（2026-09-13，方案已确认）

- **用户已确认：** 用于遗传计算的结构中，每条染色体至少有一个位点，每个位点至少有一个等位基因。该要求同样适用于常染色体和 X/Y/Z/W；无位点染色体不作为支持的计算输入。
- `Species.__init__()` 及染色体、位点的逐步构造/编辑允许暂时不完整；仍检查已提供参数本身的合法性，但不因尚未添加子项而报错。查看结构不要求其完整。
- 在完整基因型枚举、完整基因型字符串解析、遗传矩阵生成和物种基线获取前统一校验，建议由单一 `Species.validate_structure()` 入口实现，实际名称实施时确定。种群 setup 已会获取基线，不能延迟到最终 build 才检查。
- 校验失败显式抛 ValueError，并定位 Species、染色体及位点；不静默跳过空染色体或空位点。缓存命中不能绕过有效性检查或既定失效机制。
- 无须增加 finalize 或永久冻结状态。修改期间可暂时不完整，下一次计算前再检查。现有 from_dict 支持仅声明位点、随后添加等位基因，应保留这种逐步构造能力，不未经兼容核对就强制其返回时完整。
- 单态染色体可由用户显式提供单一等位基因的标记位点，系统不自动添加占位位点或等位基因。性染色体修复因此无需新增无位点 haplotype 的字符串语法；此前空 Y/W 的支持设想由本决定替代。
- 尚未修改实现；实施时同步公开合同、stub（若新增校验 API）、中英文文档和测试，并按高风险流程独立审查。

## CR-10 🎨 XY/ZW 同源区段交换（2026-09-13，暂不实现）

- **用户决定：** 暂不实现 X/Y 或 Z/W 之间的同源区段交换，不纳入当前性染色体问题修复。
- 当前 `Genotype.produce_gametes()` 仅在两个 haplotype 属于同一 Chromosome 时计算重组；异型 X/Y、Z/W 按完整单倍型各 0.5 分离。XX、ZZ 的同型副本可使用既有重组逻辑。
- 将来若实现，需要独立定义跨染色体的同源位点对应、可交换区间及重组率、交换后的染色体身份和合法配子；属于新增科学模型能力。
- 此项暂缓不影响 XY/ZW 的字符串解析、初始化精确索引和性别约束修复；不能将这些修复宣称为同源区段交换支持。

## CR-11 📋 Genotype 构造时重复计算缓存键（2026-09-13，方案已确认）

- `src/natal/frontend/genetics/entities/genotype.py:80` 起先遍历染色体生成 `chrom_pairs/genotype_name`，111 行构造 cache_key；117 行起再按 unordered 规则规范化母父单倍体并生成 canon_name，139 行无条件覆盖 cache_key。两次赋值之间没有使用第一个 key，旧字符串计算未参与后续对象名称设置。
- 前一段在完整遗传对象上只做读取和字符串拼接，属于冗余计算，不是已确认的数值错误。即使最终命中缓存，仍会先执行该段；未测量性能收益，不夸大影响。
- 建议仅删除前一套被覆盖的计算及失效注释，保留后面的 canonical key、unordered 处理、缓存查找和对象唯一性语义；不与 CR-8 的缓存生命周期重构捆绑，也不删除实体去重缓存。
- **用户已确认：** 删除前一套被覆盖的计算，保留 canonical key 与对象唯一性。当前只记录方案，未修改实现；实际删除时验证有序/无序、重复构造对象身份及性染色体的既有行为，并按最终风险分类执行检查。

## CR-12 📋 TickMetrics 通过目录名称反查基因型（2026-09-13，索引直查方案已确认）

- `src/natal/frontend/hooks/tick_context.py:140` 的 allele_frequencies 先取得名称→计数字典，然后对每个位点、每个名称调用 `_genotype_for_name()`；184 行后的实现按 `@/:` 拆字符串，再线性扫描 registry 比较 genotype.name。
- 当前 Gene 名通过 `utils/helpers.py:31` 限制为字母、数字和下划线；标准 ztype 名由 `contracts/materialize.py:348` 从 registry 生成。因此报告对合法基因型名称含冒号的担忧尚未证实为可触发缺陷，不据此宣称频率计算错误。
- 已存在 state ztype 轴与 registry.index_to_ztype 的对应关系，先生成字符串再反查属于冗余耦合；每个位点重复线性查找还增加开销，未测量性能影响。
- **用户已确认：** allele_frequencies 直接沿当前 state 的 ztype 计数轴，通过 registry.index_to_ztype 取得 `(Genotype, slab)` 对象计算，移除字符串反查；保留 genotype_counts 等公开输出的名称字典形式。确认压缩、slab 扩展及空间 deme 目录对齐，失配明确报错，不能静默截断。
- 不以本项扩大到其他科学统计语义修改或实体缓存重构。本项只记录方案，未修改实现。
- 后续核对确认名称格式混用：`contracts/blueprint.py:135` 的公开 `format_type_name()` 生成 `genotype:slab` / `haplotype:glab`，materialize 与部分 hook 目录沿用；`patterns/parser.py:50` 解析 `@lab`，`output/observation.py:920` 与 `spatial/population.py:1050` 的输出也使用 `@`。用户提出统一为 `@`，建议统一具体 ztype/gtype 名称生成，目录显式保留 `@default`；裸 genotype 选择器仍保留匹配任意 slab 的既有语义。
- **用户已确认不兼容旧冒号标签格式：** 统一使用 @，不保留 genotype:slab / haplotype:glab 兼容解析。实施同步名称生成方、消费者、文档和锁定冒号格式的测试；不能简单全局替换冒号（无序模式 `::`、空间日志 `deme{i}:param` 各有独立含义）。即使格式统一，内部统计仍直接使用 registry 对象与索引，避免字符串反查。当前仅记录，未修改实现。

## CR-13 ✅ DONE — WF 融合路径性别分配已归一化

**已于 `33236e7`（2026-09-13 `fix(genetics): unify conversion rules and harden
baseline contracts`）修复，本条记录当时已过期。** 现在三条路径使用同一概率定义：

- 分阶段离散：`rust/src/kernels/discrete_generation.rs:325-347`
- WF 融合：`rust/src/kernels/discrete_generation.rs:898-913`
- 年龄结构：`rust/src/kernels/age_structured.rs:432-440`

三处都是 `f / (f + m)`（零和回退 0.5），雄性取余量，雌雄之和严格等于子代总量。

**复现验证（2026-09-14 重跑）**：`tests/test_discrete_generation_sex_chromosome_mendelian.py`
的 XY 模型，1000 雌 + 1000 雄、每雌 1 卵、无竞争、确定性，仅切换 `extreme_speed_mode`：

```
extreme_speed_mode=0 (分阶段): female=500.0 male=500.0 total=1000.0
extreme_speed_mode=3 (WF)    : female=500.0 male=500.0 total=1000.0
```

修复前 WF 路径为雌雄各 1000、总数 2000；现在两条路径一致。

## CR-14 ✅ DONE — 计数与规则统一携带年龄轴（2026-09-14，已获独立审查 APPROVED）

**已实施**（`355c551`；讨论记录见 `3dc66f7` 之前的版本）。契约与实现：

- 计数与规则**始终携带年龄轴**，缺失时归一化为长度 1 的退化年龄类。
  `apply_rule` 只接受四维规则；传二维/三维规则抛
  `rule must carry the age axis like the counts: expected 4-D (...) ...`，并给出补轴示例。
- 二维计数 `(S, Z)` 在边界升为 `(S, 1, Z)`；其规则写作 `(n_groups, S, 1, Z)`。
  `Observation.apply` 的计数输入与输出与改动前逐位一致（40 例并排对照，唯一差异是
  "掩码与计数年龄范围不匹配"的报错信息更清晰，错误类型不变）。
- 规则与计数的 sex/age/ztype 维度不匹配现在**在进入 Rust 前**报
  `rule shape does not match the counts: ...`。这顺手修掉一个旧静默错算：元素总数恰好是
  plane 整数倍、但布局不匹配的规则以前会被原生守卫放行并按错误布局重索引。
- 归一化收敛到 `observation.py` 的 `_lift_projection_counts` / `_require_age_axis_rule`，
  `apply` 与 `apply_rule` 共用；**另外三处**同样的"无年龄轴 = 1 个年龄类"推导
  （`output/_recording.py`、`population/base.py`、`hooks/tick_context.py`）已收敛到
  `frontend/data/state.py::state_axes`（内部 helper，不新增公共导出）。
- 破坏性变更已写入 `CHANGELOG.md` 的 Breaking Changes；`docs/{en,zh}/observation_impl.md`
  记录"规则与计数始终带 age 轴"；stub 无需改动（签名未变）。

**用户决定（2026-09-14）**：不新增 `age_free` 开关——"不分年龄汇总"由
`Observation(...)` + `collapse_age=True` 覆盖，`apply_rule` 的三维便利不值得一个公共参数。

**回归**：`tests/test_observation_phase2.py` 五条规则形状测试改到新契约；
独立审查新增 `tests/test_observation_age_axis_contract.py`（18 条，父提交 17 failed / 1 passed）。

> [!NOTE] 历史标注
> 与 numba 相关的 backlog 条目（`.numba_cache`、`NUMBA_ENABLED`、
> `enable_numba()/disable_numba()`、`@pytest.mark.numba_off/on` 等）在 ⑥（numba
> 全拆）完成后已全部过时，相关机制已从仓库移除。原 #15（`.numba_cache` 旧 import）、
> #22（Numba JIT 缓存导致测试排序依赖）、#23（后端选择与测试缓存隔离）三节已删除：
> 它们描述的 Numba 运行时、缓存目录与后端 seam 都不再存在，保留只会让维护者据此
> 继续实现。


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

- Hook 运行在 Rust session 内，不能直接调用 Python Population 方法。
- `first`、`early`、`late` 对应不同生命周期阶段，单独使用 tick 无法唯一标识记录。
- `early` 状态已完成繁殖但尚未完成存活和年龄推进；`late` 状态尚未完成年龄推进。这些状态不是普通 checkpoint，不能从标准 tick 入口恢复。
- 空间模型还必须明确记录发生在 per-deme 生命周期、全局迁移之前还是之后，并保证跨 deme 一致性。
- 可能需要 `(tick, phase, occurrence)` 身份、预分配的记录缓冲区和独立的 trace schema。
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

### #3 ✅ DONE — Observation 录制逻辑重复（已随 Rust-only 重构消失）

**结论**：本条描述的"三条路径"（Numba 内核模板 / Python dispatch 回退 / 后处理）与
`observation_record.py`、`RUN_FN_NAME`、`_run_python_dispatch`、`_process_kernel_history`
均已不存在；现在只有 Rust engine 一条录制路径，重复源已消失。

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

### #16 📋 spatial `deme_id` 合并+过滤机制未在文档中说明

当前文档（`spatial_lifecycle_wrapper.md`、`3_advanced_hooks.md`）描述了 `_collect_effective_compiled_hooks()`（"收集所有 deme 的 hook"）和 Hook 签名接受 `deme_id` 参数，但**未解释两者的因果关系**：

- **实际机制**：所有 deme 的 hook 被打平进一份全局 `CompiledEventHooks`，编译为一组 lifecycle wrapper；在 `prange` 中每个 deme 调用同一组 wrapper，通过 `deme_id` 过滤——CSR 路径用 `njit_deme_selector_matches()` 跳过不匹配的 hook，njit 路径生成 `if deme_id == X` guard。
- **文档给人的印象**：每个 deme 独立运行自己的 hook 列表，`deme_id` 只是个"我是几号"的上下文。
- **待补充**：在 `spatial_lifecycle_wrapper.md` 的编译阶段添加一段解释合并+过滤的设计动机（编译一次 vs 编译 N 次）。

## 🟢 低优先级 — UX / 远期功能

### #12 ✅ DONE — Spatial migration kernel 边界效应（由 D3′/D4 定案）

**结论**：本条原先设想"总迁出量正比于邻居数"。该行为正是默认拓扑邻接行和 = 度数所
导致的**凭空造质量**，已在 D3′/D4 修复中废止：builder 把邻接行归一化为相对迁出权重，
每个 deme 都送出完整的 `migration_rate` 配额，边界 deme 只是把配额分给更少的邻居、
每个邻居份额更大。需要"少迁移"时用 `migration_rate`，不要再缩小邻接行。
详见 `CHANGELOG.md` 的 Breaking Changes 与 `docs/{en,zh}/3_spatial_simulation.md`。

### #13 📋 K 值自动推导路径测试

**来源**：`code-quality-review-report.html` #15

**此分支改动**：无。Configurator 路径的 K 值自动推导优先级链无测试。

**优先级理由**：🟢 低优先级覆盖缺口。自动推导是 fallback 逻辑，主路径已有测试。

- 添加测试验证优先级链：`carrying_capacity` > `age_1_carrying_capacity` > `initial_individual_count`
- 覆盖 Configurator 和 `pop.update()` 两个入口

### #14 ✅ DONE — PointMutation 预设 —— 多点突变 + 概率自动校正

**结论（已实装）**：新增内置预设 `PointMutation`
（`src/natal/frontend/presets/point_mutation.py`），公开导出为
`natal.PointMutation` 与 `natal.frontend.presets.PointMutation`。实现与下方设计一致：

- 单 target（`target_allele` + `mutation_rate`）与多 target（`target_alleles` +
  `mutation_rates`）两种声明形式；速率支持 `float` / `(female, male)` / 按性别字典
- 级联补偿 `r'ₖ = rₖ / (1 - Σᵢ₌₁ᵏ⁻¹ rᵢ)`，按性别分别计算，公开
  `effective_rates()` 可查看补偿后的速率
- `rate_mode="strict"`（默认，Σr > 1 报 `ValueError`）与 `"proportional"`
  （等比缩放到和为 1）
- 仅生殖系通道：原设计中的可选 `zygotic_mutation_rate`（胚胎期突变）曾实现，随后按要求
  **暂时取消**（2026-09 维护者决定），`zygote_modifier()` 固定返回 `None`；该参数已不存在，
  传入会直接 `TypeError`。胚胎期通道若将来需要，应作为独立条目重新设计（可考虑按 target
  的速率形状与阶段维度，而不是再加一个孤立标量）
- 对所有 target 的 declarative fitness patch（`make_fitness_patch_given_allele_scaling`）
- 构造期即校验声明（形式混用、target 重复/等于 source、速率数量不匹配、速率非有限或
  为负、按性别键非法、Σr 越界、非法 `rate_mode`），`reconfigure_preset` 写入的原始值
  在编译期重新归一化并复检

**未改动**：`GameteConversionRuleSet`、`GameteAlleleConversionRule`、`modifiers.py`、
任何 Rust 内核。

**文档**：`docs/{en,zh}/2_genetic_presets.md`（新增小节 + 可运行示例）、
`docs/{en,zh}/1_quickstart.md`、`docs/{en,zh}/3_custom_presets.md`、
`docs/{en,zh}/allele_conversion_rules.md`、`CHANGELOG.md`。
**测试**：`tests/test_point_mutation_preset.py`（单/多 target、补偿公式、按性别、
Σr>1 边界、reconfigure、负向声明、germline-only 负向契约与导出）+
`tests/test_point_mutation_adversarial_review.py`（独立对抗式审查补充的输入契约回归：
非法性别键的异常类型、非有限/越界速率拒绝、reconfigure 状态一致性）+
`tests/test_point_mutation_dynamics.py`（端到端动力学：中性累积 `q(t)=1-(1-μ)^t`、
竞争 target 的 B:C 比例逐代不变、隐性致死突变-选择平衡 `√μ/(1+√μ)`、乘性有害平衡与
独立参考递推逐代一致、雌性特异速率折半、X 连锁双性别累积、spatial 全局递推与迁移扩散、
重叠世代的调度决定衰减率（并在关闭精子存储时与由实测年龄结构算出的更新方程主根
逐位吻合）、随机运行的复现性与二项一致性）。三个文件合计使新模块行覆盖率达 100%。

**原设计（历史记录）**：

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

**不改**：`GameteConversionRuleSet`、`GameteAlleleConversionRule`、`modifiers.py`、任何 Rust 内核。

### #17 ✅ DONE — `PopulationConfig._replace()` 的 0-d ndarray 退化（已随引擎重构消失）

**结论**：本条定位在 `age_structured_simulator.py`（已删除）的 0-d 索引路径上，触发手段
是 `NATAL_DISABLE_NUMBA=1` 与 `@pytest.mark.numba_off`（均已不存在）。当前配置走 Rust
engine 的 Rust 侧类型，Python 侧不再有 `[()]` 索引路径，原缺陷与三个测试均不存在。
遗留的通用约定：`import_config` 仍应拒绝用 Python scalar 替换 0-d ndarray 字段，若将来
出现同类输入请在此重开条目。

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

> 注：本条写于 Python/Numba 引擎时期，下表 #B–#E 的难度、收益与行数评估针对当时的
> 实现；Rust engine 接管后这些优化点需要重新评估，不要直接按表中的数字排期。

**已完成**：offspring tensor 由 Rust 侧的
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
