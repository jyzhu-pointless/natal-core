# SpatialPopulationBuilder：空间种群批量构造

`SpatialPopulationBuilder` 通过「构建一次模板，克隆 N-1 次」的策略解决多 deme 初始化时的重复计算问题。

## 快速开始

```python
import numpy as np
from natal import Species, HexGrid, SpatialPopulation

species = Species.from_dict(name="spatial_population_builder_demo", structure={"chr1": {"loc": ["A", "B"]}})

# 所有 deme 使用相同的离散世代模型。
pop = (
    SpatialPopulation.builder(
        species, n_demes=100, topology=HexGrid(10, 10),
        pop_type="discrete_generation",
    )
    .setup(name="homo_demo", stochastic=False)
    .initial_state(individual_count={"female": {"A|A": 5000}, "male": {"A|A": 5000}})
    .survival(female_age0_survival=1.0, male_age0_survival=1.0)
    .reproduction(eggs_per_female=2, sex_ratio=0.5)
    .competition(carrying_capacity=10000)
    .migration(kernel=np.ones((3, 3)), migration_rate=0.1)
    .build()
)

pop.run(10)
assert pop.tick == 10
```

## 核心设计

### 两层结构

```
SpatialPopulation.builder(...)
    │
    └─► SpatialPopulationBuilder         ← 面向用户的链式 API
           │
           ├─ _template        ← 单 deme 模板 PopulationBuilder（不是已删除的 Builder 类）
           │                     始终只看到一个 deme 的标量参数
           ├─ _batch_settings  ← {参数名: BatchSetting}
           │                     拦截到的跨 deme 变化参数
           └─ _declaration_log      ← [(method_name, kwargs), ...]
                                 每次链式调用的完整记录
```

`SpatialPopulationBuilder` 不新增种群类，而是在外层包装一个单 deme 模板 `PopulationBuilder`。链式调用阶段同时做三件事：

1. **代理给 `_template`** — 模板 `PopulationBuilder` 始终收到标量值，保持正确的内部状态
2. **检测 `BatchSetting`** — 拦截并存储到 `_batch_settings`，template 只拿到 `first_value()`
3. **记录到 `_declaration_log`** — 保留原始参数（含 BatchSetting 对象），供各组声明投影

### 冻结声明

`build()` 先将声明冻结为单一的 `ModelDefinition`：模板输入（单 deme 的设置、遗传规则与 Hook）即声明字段，`.spatial` 保存 deme 数量、拓扑、展开后的逐 deme batch 值、迁移、空间观测、历史容量和压缩声明。实际构建从这个声明创建隔离的编译器，再按遗传差异分组构建 deme。

batch 函数在冻结时展开一次；已缓存的模板遗传产物在最终构建时复用。冷编译使用冻结的具体值，不重新求值 batch 函数。定义查询复制 NATAL 的数组和容器，修改查询结果不影响以后构建；用户的 preset、Hook 及其外部资源保留身份，不要求支持深复制。

### 代理机制

领域方法先让模板校验本次具体参数，成功后才提交 batch 设置和声明记录。`presets()` 和 `hooks()` 分别保留自己的位置参数，batch preset 不会替换 hook。

batch 值归属于声明它的那次调用。对于同一参数，后声明的普通值会覆盖此前的 batch 设置，后声明的 batch 设置也按调用顺序生效。例如先声明 `eggs_per_female=batch_setting([10, 20])`，再声明 `eggs_per_female=30`，两个 deme 最终均使用 30。遗传操作仍遵守其累积和优先级规则。

### 参数别名

`competition()` 是跨 `pop_type` 的统一入口，内部做了参数名规范化：

```
用户传入 carrying_capacity ─┐
                             ├─ age_structured → age_1_carrying_capacity（内部键名）
用户传入 age_1_carrying_capacity ─┘
                             └─ discrete_generation → carrying_capacity（保持原名）
```

优先级：`age_1_carrying_capacity` > `old_juvenile_carrying_capacity` > `carrying_capacity`。

这在 `_declaration_log` 中统一键名，确保声明投影时参数名与对应 `PopulationBuilder` 方法一致。

## 两条构建路径

`build()` 根据是否存在 `_batch_settings` 自动分叉：

### 同构路径（无 batch_setting）

```
_build_homogeneous_demes():
    1. template = self._template.build()     # 完整流程一次
    2. config = 模板自己的 draft               # 克隆按引用共享
    3. demes = [template]
    4. for i in 1..n_demes:
           demes.append(_clone_deme(template, config))
    5. return SpatialPopulation(demes, ...)
```

### 异构路径（有 batch_setting）

```
_build_heterogeneous_demes():
    1. expanded = {name: batch.expand(n_demes, topology) for ...}
       # 把所有 BatchSetting 展开为 per-deme 值列表

    2. _genetics_batch_names() 选出影响遗传学部分的 batch kwarg，
       deme 按该遗传学签名分组
       # 例如：batch 的 fitness 行会拆分组；仅生态参数的差异
       #（carrying capacity、初始状态等）不会拆分组

    3. 按签名分组 → {sig: [deme_index, ...]}

    4. 对每组:
       a. _projected_group(values[first]) 解析该 deme 的日志
          （代入该 deme 的批值），经唯一声明解释器投影到全新基线
          ——不重新执行任何 builder 方法
       b. _carrier_from_projection 从投影恢复未发布载体；遗传与
          模板匹配的组 0 继承模板编译缓存（配方不重跑）
       c. carrier._compile_products() → compile_definition 产出该组产物
       d. 组内其余 deme：仅生态差异的 variant 只把*不同*的声明
          投影到组 config；派生标量冻结于组计算值

    5. 按索引组装所有 deme，构造 SpatialPopulation
```

投影取代了旧的重放核心：不是在全新 builder 上重新执行日志里的
builder 方法，而是由 `project_declaration_record`
（`builder/_declarations.py`）解释日志——与链式方法委托的是同一组
纯 `apply_*` 函数。因此带 batch 的构建执行零个 builder 方法（有
spy 测试钉住），且每个 deme 的编译结果与等价单种群声明一致。
可投影参数、读取时派生的平衡态指标与不支持异构的参数见
[异构 Config 共享机制](spatial_config_replace.md)。

## `_clone_deme`：零编译开销的克隆

`_clone_deme(template, config, name)` 委托给种群实例的 `_clone(name=..., config=...)`：用 `__new__` 创建实例、完全绕过 `__init__`，因此不会重复执行 Hook 编译和配置构建。

```python
def _clone_deme(template, config, name):
    # config 按引用交给克隆；同构构建中所有克隆共享同一份导出配置
    return template._clone(name=name, config=config)
```

同构构建里，模板 deme（索引 0）保留自己的 draft，索引 1 起的克隆共享模板的 config 对象（大数组按引用共享）。

`_clone` 的共享与独立关系：

| 类别 | 内容 | 原因 |
|------|------|------|
| 共享引用 | `_species`、`_config`（同构克隆共享同一份导出配置；模板 deme 保留自己的 draft）、`_index_registry`、`_registry`、`compiled_hook_descriptors`、native `HookProgram`、`_hook_runner`、`_genotypes_list`、`_haploid_genotypes_list` | 仿真期间只读，同构 deme 完全一致 |
| 浅拷贝列表 | `_presets`、`_manual_gamete`、`_manual_zygote`、`_gamete_modifiers`、`_zygote_modifiers` | 每个 deme 可独立增删修饰器，不影响其他 deme |
| 独立副本 | `_state`（复制模板数值的个体/精子数组）、`_initial_population_snapshot`、`_name`、`_deme_id`、`_tick`、`_params_log`（新的原生日志）、`_reconfiguration_log`、`_run_program` | 每个 deme 有自己的运行状态与审计记录 |
| 重置 | Rust 会话桥接字段（`_rust_lifecycle_backend = None` 等） | 克隆从无后端状态开始，首次运行时按需建立会话 |

> `_config` 的共享是**构建期**的数据去重，不是运行期可写共享：修改运行中的 deme 请走 `deme(i).write_ecology(...)` / `write_genetics(...)` 或 `pop.params.tensor_write(...)`。

## `BatchSetting`：跨 deme 变化的参数

```python
from natal.frontend.spatial import batch_setting

# 列表：按索引一一对应
batch_setting([10000, 5000, 5000, 8000])        # kind="scalar"

# NumPy 数组
batch_setting(np.array([10000, 5000, ...]))      # kind="array"

# 空间函数：(flat_index) -> float 或 (row, col) -> float
batch_setting(lambda i: 10000 if i < 50 else 5000)  # kind="spatial"
```

三种 kind 在 `build()` 时通过 `expand(n_demes, topology)` 统一展开为 Python 列表。

接受 `BatchSetting` 的参数（按方法）：

- `initial_state`：`individual_count`、`sperm_storage`
- `survival`（离散世代标量）：`female_age0_survival`、`male_age0_survival`；`reproduction`（离散世代标量）：`female_adult_mating_rate`、`male_adult_mating_rate`
- `reproduction`：`eggs_per_female`、`sex_ratio`
- `competition`：`carrying_capacity` / `age_1_carrying_capacity`、`low_density_growth_rate`、`juvenile_growth_mode`、`expected_num_new_adult_females`、`equilibrium_distribution`
- `presets`：位置参数形式的 preset 对象
- `fitness`：`viability`、`fecundity`、`sexual_selection`、`zygote_viability`
- `migration`：`kernel`、`migration_rate`

## 构造开销

同构构建只完整执行一次模板构建，其余 deme 走 `_clone`；异构构建按 config 签名分组，每组构建一个模板再克隆。首次模板构建的耗时取决于 Hook 数量与遗传规模，克隆本身只复制状态数组。

本页不给出历史测量数字：此前表格未注明版本与测量条件，不能当作当前性能保证。需要性能结论时，请以自己的模型和工作负载实测。

## 与现有 API 的关系

`SpatialPopulationBuilder` 不修改任何现有类：

- 旧的 Builder 类（`AgeStructuredPopulationBuilder` / `DiscreteGenerationPopulationBuilder`）已移除，`SpatialPopulationBuilder` 是唯一的批量配置路径
- `SpatialPopulation.__init__` — 不变，`build()` 最终调用它，传入已构建好的 deme 列表
- 旧的逐 deme 构造写法仍然有效

## 边界与限制

1. **影响遗传学的 batch 值各成编译组** — `fitness` 的行参数与 `presets` 都支持 `batch_setting`，但每个不同的取值都会形成自己的遗传学组、经 `compile_definition` 各编译一次；组内仅生态差异的 variant 共享该组的遗传产物大数组（见[异构 Config 共享机制](spatial_config_replace.md)）
2. **spatial kind 需要 topology** — `batch_setting(lambda row, col: ...)` 形式的回调要求 builder 传入了 topology 参数，否则 expand 时报错；`(flat_idx)` 形式不依赖 topology（按参数个数自动识别）
3. **同构 deme 共享同一 `_config` 引用** — 这是构建期的数据去重，不代表运行期可以直接写 `_config`。直接改 `pop.demes[0]._config` 的数组字段既绕过会话同步，也会影响所有共享该 config 的 deme；运行期修改单个 deme 用 `deme(i).write_ecology(...)` / `write_genetics(...)`，批量修改用 `pop.params.tensor_write(...)`
