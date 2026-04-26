# SpatialBuilder：空间种群批量构造

`SpatialBuilder` 通过「构建一次模板，克隆 N-1 次」的策略解决多 deme 初始化时的重复计算问题。2601 个同构 deme 的构造时间从 ~2.6s 降至 ~16ms。

## 快速开始

```python
from natal import Species, HexGrid, SpatialPopulation
from natal.spatial_builder import batch_setting

species = Species.from_dict(name="demo", structure={"chr1": {"loc": ["A", "B"]}})

# 同构：所有 deme 参数一致
pop = SpatialPopulation.builder(species, n_demes=100, topology=HexGrid(10, 10)) \
    .setup(name="homo_demo", stochastic=False) \
    .initial_state(individual_count={"female": {"A|A": 5000}, "male": {"A|A": 5000}}) \
    .reproduction(eggs_per_female=50) \
    .competition(carrying_capacity=10000) \
    .migration(migration_rate=0.1) \
    .build()

pop.run(10)
```

## 核心设计

### 两层结构

```
SpatialPopulation.builder(...)
    │
    └─► SpatialBuilder         ← 面向用户的链式 API
           │
           ├─ _template        ← AgeStructuredPopulationBuilder（或 DiscreteGeneration…）
           │                     始终只看到一个 deme 的标量参数
           ├─ _batch_settings  ← {参数名: BatchSetting}
           │                     拦截到的跨 deme 变化参数
           └─ _replay_log      ← [(method_name, kwargs), ...]
                                 每次链式调用的完整记录
```

`SpatialBuilder` 不修改现有 builder 类，而是在外层包装。链式调用阶段同时做三件事：

1. **代理给 `_template`** — template builder 始终收到标量值，保持正确的内部状态
2. **检测 `BatchSetting`** — 拦截并存储到 `_batch_settings`，template 只拿到 `first_value()`
3. **记录到 `_replay_log`** — 保留原始参数（含 BatchSetting 对象），供异构场景回放

### 代理机制

每一个链式方法最终都经过 `_detect_and_delegate`：

```python
# 以 .competition(carrying_capacity=batch_setting([10000, 5000, 5000, 8000])) 为例

def _detect_and_delegate(self, method_name, kwargs):
    concrete = {}
    for key, value in kwargs.items():
        if isinstance(value, BatchSetting):
            self._batch_settings[key] = value        # 存储原对象
            first = value.first_value()               # 取第一个标量值
            if first is not None:
                concrete[key] = first                 # template 只看到标量
        else:
            concrete[key] = value                     # 普通参数原样传递

    self._replay_log.append((method_name, dict(kwargs)))  # 记录原始调用

    method = getattr(self._template, method_name)
    method(**{k: v for k, v in concrete.items() if v is not None})
    return self
```

`presets()` 和 `hooks()` 有位置参数，走 `_delegate_positional`，逻辑相同。

### 参数别名

`competition()` 是跨 `pop_type` 的统一入口，内部做了参数名规范化：

```
用户传入 carrying_capacity ─┐
                             ├─ age_structured → age_1_carrying_capacity（内部键名）
用户传入 age_1_carrying_capacity ─┘
                             └─ discrete_generation → carrying_capacity（保持原名）
```

优先级：`age_1_carrying_capacity` > `old_juvenile_carrying_capacity` > `carrying_capacity`。

这在 `_replay_log` 中统一键名，确保异构回放时参数名与 template builder 签名一致。

## 两条构建路径

`build()` 根据是否存在 `_batch_settings` 自动分叉：

### 同构路径（无 batch_setting）

```
_build_homogeneous():
    1. template = self._template.build()     # 完整流程一次
    2. config = template.export_config()      # 导出 PopulationConfig
    3. demes = [template]
    4. for i in 1..n_demes:
           demes.append(_clone_deme(template, config))
    5. return SpatialPopulation(demes, ...)
```

### 异构路径（有 batch_setting）

```
_build_heterogeneous():
    1. expanded = {name: batch.expand(n_demes, topology) for ...}
       # 把所有 BatchSetting 展开为 per-deme 值列表

    2. 按 (参数名, 参数值) 元组计算每个 deme 的 config 签名
       # 例如 deme 0: (("age_1_carrying_capacity", 10000.0),)

    3. 按签名分组 → {sig: [deme_index, ...]}

    4. 对每组:
       a. _build_template_for_group(sig_map)
          # 创建新 builder，重放 _replay_log，替换 batch 参数为组值
       b. 组内其余 deme = _clone_deme(group_template)

    5. 按索引组装所有 deme，构造 SpatialPopulation
```

`_build_template_for_group` 是回放的核心：

```python
def _build_template_for_group(self, sig_map):
    builder = AgeStructuredPopulationBuilder(self._species)  # 全新的 builder

    for method_name, kwargs in self._replay_log:
        resolved = {}
        for key, value in kwargs.items():
            if key in sig_map:
                resolved[key] = sig_map[key]   # 替换为该组的标量值
            elif isinstance(value, BatchSetting):
                resolved[key] = value.first_value()  # 未覆盖的 batch 取首个值
            else:
                resolved[key] = value           # 非 batch 参数原样传递

        getattr(builder, method_name)(**resolved)

    return builder.build()
```

## `_clone_deme`：零编译开销的克隆

克隆用 `__new__` 创建实例，完全绕过 `__init__`，避免重复执行 hook 编译和 config 构建。

```python
def _clone_deme(template, config, name):
    clone = AgeStructuredPopulation.__new__(AgeStructuredPopulation)

    # === 共享引用（仿真期间只读）===
    clone._species           = template._species
    clone._compiled_hooks    = template._compiled_hooks     # 已编译的 hook 函数
    clone._hook_executor     = template._hook_executor       # hook 执行引擎
    clone._config            = config                        # PopulationConfig（共享）
    clone._index_registry    = template._index_registry      # 基因型查找表
    clone._registry          = template._registry
    clone._gamete_modifiers  = template._gamete_modifiers    # 配子修饰器
    clone._zygote_modifiers  = template._zygote_modifiers
    clone._genotypes_list    = template._genotypes_list
    clone._haploid_genotypes_list = template._haploid_genotypes_list

    # === 独立复制 ===
    clone._name    = name
    clone._history = []
    clone._state   = State.create(...)                        # 新 state 数组
    clone._state_nn.individual_count[:] = template._state_nn.individual_count
    clone._state_nn.sperm_storage[:]    = template._state_nn.sperm_storage
    clone._initial_population_snapshot  = (copy of template's snapshot)

    return clone
```

| 属性 | 共享/独立 | 原因 |
|------|----------|------|
| `_compiled_hooks`, `_hook_executor` | 共享引用 | 无状态，只读 registry |
| `_config` | 共享引用 | 同构 deme 的 config 完全一致 |
| `_index_registry`, `_registry` | 共享引用 | 物种相同则基因型索引相同 |
| `_gamete_modifiers`, `_zygote_modifiers` | 浅拷贝列表 | preset 可能操作同一对象 |
| `_state` | 独立创建 | 每个 deme 有独立的个体数量 |
| `_history`, `snapshots` | 独立空列表/dict | 记录各自的仿真历史 |

## `BatchSetting`：跨 deme 变化的参数

```python
from natal.spatial_builder import batch_setting

# 列表：按索引一一对应
batch_setting([10000, 5000, 5000, 8000])        # kind="scalar"

# NumPy 数组
batch_setting(np.array([10000, 5000, ...]))      # kind="array"

# 空间函数：(topology, deme_idx) -> float
batch_setting(lambda topo, i: 10000 if i < 50 else 5000)  # kind="spatial"
```

三种 kind 在 `build()` 时通过 `expand(n_demes, topology)` 统一展开为 Python 列表。

接受 `BatchSetting` 的参数有：`carrying_capacity`、`age_1_carrying_capacity`、`eggs_per_female`、`sex_ratio`、`low_density_growth_rate`、`juvenile_growth_mode`、`expected_num_adult_females`。

## 性能数据

测试条件：2 等位基因（4 种基因型），初始每 deme 200 个体。

| 场景 | 耗时 | 说明 |
|------|------|------|
| 同构 100 demes | ~400ms | 含首次 Numba 编译 ~350ms |
| 同构 2601 demes | ~16ms | 克隆阶段，不含首次 template 构建 |
| 2 组异构 4 demes | ~6ms | 每组一个 template + 各克隆 1 次 |
| 首次 template 构建 | ~2-3ms | Numba 编译有文件缓存，命中后更快 |

首次 template 构建的时间取决于 hook 数量和 Numba 缓存状态，后续调用通常 < 5ms。

## 与现有 API 的关系

`SpatialBuilder` 不修改任何现有类：

- `AgeStructuredPopulationBuilder` / `DiscreteGenerationPopulationBuilder` — 不变，SpatialBuilder 通过组合方式包装
- `SpatialPopulation.__init__` — 不变，`build()` 最终调用它，传入已构建好的 deme 列表
- 旧的逐 deme 构造写法仍然有效

## 边界与限制

1. **`batch_setting` 不支持 fitness / presets** — fitness 和 presets 修改的是 config 内部的 NumPy 数组（in-place），不适合通过标量值表达。需要异构 fitness 时，在 build 后手动修改对应 deme 的 config 数组
2. **spatial kind 需要 topology** — `batch_setting(lambda topo, i: ...)` 要求 builder 传入了 topology 参数，否则 expand 时报错
3. **同构共享同一 `_config` 引用** — 如果后续代码直接修改 `pop.demes[0]._config` 的数组字段（非 `_replace`），会同时影响所有共享该 config 的 deme。正确做法是先用 `config._replace(...)` 创建独立副本
