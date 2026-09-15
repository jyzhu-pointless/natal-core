# 一个模型从声明到结果的完整旅程

假设你让 agent 实现一个模拟：100 只杂合雌性和 100 只杂合雄性繁殖，后代按孟德尔规律继承等位基因，一半幼体存活。agent 给出的结果是下一代有 100 个成体，其中 AA、Aa、aa 的数量分别为 25、50、25。这个结果从哪里来？构建器、遗传张量和 Rust 会话在其中分别做了什么？

本章沿同一个模型走完整条路径。你不需要先阅读源码，也不需要记住所有函数名；先理解每一步改变了什么，再用文末的实现定位表与 agent 讨论具体修改。[下一章](reproduction.md)继续展开其中的繁殖计算。

## 把模型要求写完整

我们声明一个二倍体物种：一条常染色体上有一个基因座，允许 A、a、X 三种等位基因。每个个体在该位点携带两份等位基因，且本例不区分双亲来源，因此 A|a 与 a|A 表示同一种基因型。X 用于展示“物种允许的类型”与“本次模拟需要的类型”的区别；本例没有产生 X 的突变或转换。

下表是完整的教学场景，不是一段省略初始化的代码。

| 项目 | 本例声明 | 它决定什么 |
| --- | --- | --- |
| 模型 | 普通分阶段离散世代，关闭融合加速模式 | 一步内繁殖、生存，然后整代替换 |
| 数值模式 | 确定性 | 数量按期望计算，可以出现小数 |
| 初始成体 | 雌雄各 100，全部 A\|a | 两个亲本池 |
| 初始幼体 | 0 | age-0 槽位最初为空 |
| 遗传规则 | 孟德尔分离；配子与体细胞标签均只有 default | 无额外遗传转换和标签状态 |
| 雌雄成体交配率 | 均为 1 | 本例所有雌性可交配，雄性均进入对象权重 |
| 每雌性产卵数 | 2 | 中性配对的期望卵数 |
| 繁殖参与率 | 离散模型内部成体值为 1 | 已配对雌性全部参与产卵 |
| 全局性别比 | 0.5，即雌性份额 | 本例各类型后代均分雌雄 |
| 各类 fitness | 均为 1 | 无选择或额外生殖损失 |
| 幼体基础生存率 | 雌雄均为 0.5 | 幼体到成体的保留比例 |
| 密度调节 | no_competition | 不根据数量或承载量再缩减幼体 |
| 索引压缩 | 开启 | 删除本次模型不可达的类型 |
| 输出 | raw 历史，运行时每 tick 记录 | 保存初始及完成后的边界 |

这里的“繁殖参与率”是内核读取的内部值。离散构建入口使用成体交配率等专用参数，不能因为内部存在年龄数组，就给它传年龄结构模型的任意 per-age 参数。

这些要求决定了我们预期看到的现象：遗传比例是 1:2:1，但总量会从 200 变成 100。比例正确和数量正确是两个不同的判断。

## 第一个转换：物种描述成为类型目录

`Species` 表达遗传结构；`PopulationBuilder` 组织本次模拟的数量、生态条件和行为声明。构建器需要一份 `IndexRegistry`，将有意义的类型映射成数组能使用的整数位置。

三个等位基因在本例的无序二倍体表示中形成六种基因型。配子只携带一份等位基因，因此有三种配子类型。项目还允许类型携带标签：个体类型称为 ZType，身份是“基因型 + 体细胞标签”；配子类型称为 GType，身份是“单倍体基因型 + 配子标签”。本例只有 default 标签，暂时可以把它们看成基因型与配子种类。

本例实际构建出的完整目录如下：

| 完整 ZType 索引 | 类型名称 | 初始数量：雌 / 雄 |
| --- | --- | --- |
| 0 | A\|A@default | 0 / 0 |
| 1 | A\|a@default | 100 / 100 |
| 2 | A\|X@default | 0 / 0 |
| 3 | a\|a@default | 0 / 0 |
| 4 | a\|X@default | 0 / 0 |
| 5 | X\|X@default | 0 / 0 |

配子目录依次是 A@default、a@default、X@default。`@default` 是本次核验的实际名称格式。目录顺序来自物种枚举与标签展开，不应被调用者当作对任意物种都成立的固定编号。

目录的作用不只是提高查询速度。数量数组的第 3 列、遗传表的第 3 个类型和输出中的第 3 个名称必须指向同一对象，否则程序可以正常计算，却解释了错误的基因型。

## 第二个转换：用户参数成为统一草稿

`ModelDraft` 装的是已经数值化的构建材料。离散世代固定使用两个年龄槽：age 0 是本步产生的幼体，age 1 是参与繁殖的成体。它们表示生命周期角色，不表示两个任意长度的现实年龄区间。

在完整类型目录上，初始个体数量数组的 shape 是 `(2, 2, 6)`：性别、年龄、ZType。每个性别的 age-0 行全零，age-1 行是 `(0, 100, 0, 0, 0, 0)`。这些格子保存聚合数量，没有为 200 个个体分别创建身份对象。

用户输入的标量也转成内核统一读取的数组：

| 草稿字段 | 本例值 | 如何读取 |
| --- | --- | --- |
| `age_based_mating_rates` | 雌性 `(0, 1)`；雄性 `(0, 1)` | 幼体不交配，成体交配率为 1 |
| `age_based_reproduction_rates` | `(0, 1)` | 只有成体繁殖 |
| `age_based_survival_rates` | 雌雄均为 `(0.5, 0)` | survival 使用 age-0 的 0.5；旧成体在 aging 被替换 |

`build_discrete_engine_config()` 建立离散模型的固定结构和默认值，`build_config_maps()` 负责共享草稿组装。声明更新写入相应字段。这里虽然保留了 age-1 生存率的位置，但不能通过把它改成 1 来令旧成体跨代存活；离散内核的代际替换规则不由该格决定。

这就是内部统一表示与用户可配置语义之间的区别。统一数组方便不同内核读取，并不表示每个数组格在每个模型里都开放为相同的自由参数。

## 第三个转换：遗传规则成为概率表

初始数量回答“现在有什么”，遗传表回答“这些亲本可以产生什么”。编译需要同时保留这两种信息。

构建器通过 `_definition_for_compile()` 捕获 `ModelDefinition`，其中包括声明、完整注册表、草稿以及预设和修饰器等信息。`_compile_products()` 可复用同一声明的编译产物，否则调用 `compile_definition()`。因此，“build 时一定把所有规则从头重算一遍”也不是准确描述。

编译结果 `CompiledProducts` 包含草稿、注册表和修饰器列表。其主要遗传产物有两张表：

| 表 | 完整轴 shape | 回答的问题 |
| --- | --- | --- |
| `zygotes_to_gametes_map`，记作 M | `(2, 6, 3)` | 某性别、某个体类型产生各种配子的概率是多少？ |
| `gametes_to_zygotes_map`，记作 F | `(3, 3, 6)` | 一个雌性配子与一个雄性配子形成哪种后代？ |

本例 A|a 亲本的 M 行是 `(0.5, 0.5, 0)`；A 配子与 a 配子的 F 行指向 A|a。初始没有 AA 和 aa 个体，仍然必须知道这些类型及其遗传关系，因为它们第一代就能产生。

如果加入预设或修饰器，编译器从基线重新组织 fitness 和映射，按规定顺序施加声明。本例没有这些干预，所以最终就是孟德尔映射。不能把已经应用过转换的表当作新的中性基线，否则重新编译可能重复应用同一个转换。

此时 `offspring_tensor` 还是 shape 为 `(0, 0, 0)` 的占位数组。它不是“所有配对都不能生育”，而是表示后代张量尚未推导；这份完整轴产物不能直接作为最终运行模型使用。

## 第四个转换：找出本次运行真正需要的类型

压缩依据遗传可达性，而不是简单删除当前数量为零的列。`plan_projection()` 从初始非零类型和显式保留类型等种子出发，沿遗传映射找到未来可达的类型；年龄结构还需要考虑初始精子存储中的类型。

本例的可达链为：

```text
初始 A|a
  → 配子 A、a
  → 后代 A|A、A|a、a|a
  → 这些后代仍只产生 A、a 配子
  → 不再出现新类型
```

所以 AA 和 aa 虽然初始为零，也要保留；X 不可达，所有含 X 的类型可以删除。本例投影结果为：

| 完整索引 | 运行时索引 | 类型 |
| --- | --- | --- |
| 0 | 0 | A\|A@default |
| 1 | 1 | A\|a@default |
| 3 | 2 | a\|a@default |
| 2、4、5 | 删除 | 含 X 的类型 |

`publish_products()` 创建新的运行注册表，并将数量、fitness、M、F 等相关数组一起投影。M 变成 `(2, 3, 2)`，F 变成 `(2, 2, 3)`，个体数量变成 `(2, 2, 3)`。如果只压缩数量，那么运行索引 2 的 aa 数量就可能被错误地送进原先 A|X 的遗传路径。

随后在最终轴上推导 P，即 `offspring_tensor`，shape 为 `(3, 3, 3)`，依次按雌性亲本、雄性亲本、后代索引。本例有 27 个元素，而完整六类型张量会有 216 个。推迟推导避免先构造完整立方张量再切掉大部分元素。

发布前还会核对类型身份、顺序、shape 和名称。相同 shape 并不足以证明两个模型可以共用同一套索引。发布后，Hook 选择器也必须在这份最终目录上解析。

如果将来需要向模型引入 X，应在构建时让它进入保留范围，或按新的布局重建模型。不能把“目录中被删除的类型”当作“一个目前为零、随时可写入的格子”。

## 第五个转换：构建产物进入执行会话

发布完成不等于已经开始模拟。`_build_published()` 创建 Python population，并调用其会话初始化路径。`RustDiscreteLifecycleBackend` 通过 `materialize()` 把草稿分成两个跨语言合同：

| 合同 | 本例中的内容 | 用途 |
| --- | --- | --- |
| `Blueprint` | 两个性别、两个年龄、三个 ZType、名称、执行标志、初始数量等 | 固定模型布局与初始条件 |
| `Params` | 产卵数、性别比、生态数组、fitness、遗传张量等 | 提供运行时参数数据 |

`materialize()` 为数组建立新的所有权；Rust 的 `from_parts()` 再读取并验证 Blueprint、`EcologyParams` 和 `GeneticsTensors`，创建自身持有的状态、RNG、tick 和执行状态。Python 暴露的原生类名是 `DiscreteEngineSession`，对应 Rust 结构 `DiscreteGenerationSession`。

这里有两个容易误解的细节。

首先，Rust 并不是到运行时才参与。刚才推导 P 的 `recompute_offspring_tensor()` 已经调用 Rust 数值内核。更准确的分工是：Python 编排模型与接口，Rust 承担原生数值计算和运行会话；两者的边界不能直接等同于 build 与 run 的边界。

其次，草稿和合同可以带有统一结构中的精子字段，但离散会话只保存个体数量，不保留跨 tick 精子库。构建数据中出现一个字段，并不保证运行模型会持有或使用它。

Builder 在建立会话后完成记录计划的编译。记录计划依赖最终索引和形状，负责让保存的数值与名称对应起来。到此，模型可以运行了，但初始数量仍然是雌雄各 100 个 A|a 成体。

## 第六个转换：一个 tick 产生新的数量

本例的 `run_tick()` 最终进入会话批量路径中的一次迭代。普通离散 tick 的过程如下；表内三元组均按 AA、Aa、aa 排列，数量按每个性别分别列出。

| 边界 | tick | 每性别 age-0 | 每性别 age-1 | 此刻发生了什么 |
| --- | --- | --- | --- | --- |
| 构建结束 / first | 0 | `(0, 0, 0)` | `(0, 100, 0)` | 亲本就绪 |
| reproduction 后 / early | 0 | `(25, 50, 25)` | `(0, 100, 0)` | 共 200 个后代，雌雄各半；旧成体仍存在 |
| survival 后 / late | 0 | `(12.5, 25, 12.5)` | `(0, 100, 0)` | 幼体保留一半 |
| aging 后的稳定边界 | 1 | `(0, 0, 0)` | `(12.5, 25, 12.5)` | 后代成为成体，旧成体被替换 |

三个阶段之间的 Hook 位置即使没有注册回调，也决定了以后插入干预时能看到什么。early 时全数组求和会得到 400，因为 200 个亲本和 200 个后代同时存在；它不是最终下一代的数量。

本例繁殖的算术链是：100 个配对 × 每配对 2 个卵 = 200；孟德尔分配为 `(50, 100, 50)`；每个性别得到 `(25, 50, 25)`；存活一半后为 `(12.5, 25, 12.5)`。小数表示聚合的确定性期望，不能四舍五入后继续声称与该模型等价。

只有正常完成整个 tick，时钟才从 0 变成 1。停止或失败可能留下部分执行边界，不能把正常路径的最后一行套到所有退出情况。

## 最后一个转换：同一份结果有不同查询布局

执行结束后，Rust 会话是当前数量的权威来源。Python 的 `pop.state` 返回带复制数组的快照；必要时先刷新本地缓存。它的 `individual_count` shape 为 `(sex, age, ZType)`，本例是 `(2, 2, 3)`。修改返回的数组不会成为引擎更新。

默认 identity observation 为每个 ZType 建一个组。单种群本例的 `pop.observe()` 返回 tick 1，轴为 `(group, sex, age)`，shape 为 `(3, 2, 2)`。A|a 组的雌雄 age-1 值都是 25。观测只是换一种有名称的组织方式；其他组定义还可以聚合多个类型，但聚合后不能一般性地反推出每个原始类型的数量。

本例启用 raw 历史并每 tick 记录，运行一步后 ticks 为 `(0, 1)`，`history.individual_count` shape 为 `(record, sex, age, ZType)`，即 `(2, 2, 2, 3)`。历史保存边界，不会自动把上表的 early 和 late 临时状态各存一行。

因此，下面三个问题应该分别回答：当前 state 是多少，当前观测怎样分组，历史保存了哪些 tick。数组看起来都是数字，并不意味着它们的轴和信息量相同。

## 你可以据此判断哪些开发方案

| agent 的建议 | 本例提供的判断依据 |
| --- | --- |
| 只保留初始非零基因型 | 会错误删除 AA 和 aa；应检查遗传闭包 |
| 给模型追加一个数组列来引入 X | 还涉及注册表、M、F、P、fitness、选择器、会话和记录布局 |
| 修改 `pop.state` 的数组来干预种群 | 改的是查询快照，应使用受控写入通道 |
| 每次运行都重新构造会话 | 需要说明如何保留时间线、参数、历史和随机流 |
| 结果满足 1:2:1，证明繁殖正确 | 仍可能有两倍产卵数或错误生存率；还要核对总量与阶段 |
| 把 Rust 后端延迟到第一次 run 再检查 | 构建阶段已经可能调用 Rust 推导 P |

前两项不仅是索引实现问题，还影响哪些生物学状态可以表示。非重叠世代和确定性数量属于模型语义；压缩与延迟推导是实现策略；快照和受控更新是接口合同。提出修改时应说明要改变哪一类。

## 实现定位与核验依据

正文已经解释了整条链路。下表供 agent 定位实现，或供你在审阅方案时检查它有没有遗漏相邻步骤。

| 实现入口 | 本章对应职责 |
| --- | --- |
| [PopulationBuilder](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/builder/_base.py)：`_compile_products()`、`_publish_and_build()`、`_build_published()` | 编译、发布、会话及记录初始化 |
| [build_registry()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/builder/_registry_builder.py) | 完整物种目录 |
| [build_discrete_engine_config()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/model/assembly.py) | 两年龄的内部表示 |
| [compile_definition()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/model/definition_compiler.py) | 声明与候选遗传产物 |
| [plan_projection() / publish_products()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/model/publication.py) | 可达性与统一换坐标 |
| [materialize()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/contracts/materialize.py)、[RustDiscreteLifecycleBackend](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/backends/rust/rust_backend.py) | 跨语言合同与适配 |
| [DiscreteGenerationSession](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/sessions/discrete_generation.rs) | 原生状态、批量循环与记录 |
| [compile_recording_plan()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/output/_recording.py) | 输出布局与名称 |

本章的目录、投影、数组形状、early/late 数量、最终 state、默认 observation 和 raw 历史均以同一组输入核验。已有 [publication contracts](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_publication_contracts.py)、[materialize contracts](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_contracts_materialize.py) 和 [recording lifecycle](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_run_state_and_recording_lifecycle.py) 测试分别保护索引、数据隔离及执行记录边界；它们并不代替所有其他模型组合的验证。

下一步阅读[一次繁殖如何计算](reproduction.md)，把本章中“产生 200 个后代”的一格进一步拆开。
