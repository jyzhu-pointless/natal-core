# Builder → Configurator 迁移指南

v0.2.0 中，`PopulationBuilder` 及其子类（`DiscreteGenerationPopulationBuilder`、
`AgeStructuredPopulationBuilder`、`SpatialBuilder`）已被 `Configurator` 链式 API 取代，
且没有遗留的 `legacy_path` 逃逸舱——旧 Builder 类与 `setup(legacy_path=True)` 并**不存在**。

## 变更对照

| 之前 (v0.1.x) | 之后 (v0.2.0) |
|---|---|
| `PopulationBuilder` / `setup()` 返回 Builder | `setup()` 返回 `Configurator`，`build()` 产出种群对象 |
| `.competition(...)` 延迟到 `build()` 写入 | 参数立即写入配置数组（链式语法相同） |
| `SpatialBuilder(species, topology)` | `SpatialPopulation.builder(species, n_demes, pop_type)` 返回 `SpatialConfigurator` |
| Hook 签名 `(ind_count, tick)` | `(state, config, deme_id)`（njit 时代）→ 现为单参数 `TickContext` 回调 / 声明式 `Op` |

## 不变的部分

链式 API 语法**完全一致**——以下代码无需修改：

```python
pop = (nt.DiscreteGenerationPopulation
    .setup(species=sp, name="MyPop", stochastic=True)
    .initial_state({"male": {"WT|WT": 500}, "female": {"WT|WT": 500}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .build()
)
```

## API 差异

### 1. 导入路径

```python
# 顶层用户 API 不变：从 natal（或 import natal as nt）顶层导入
from natal import Species, HomingDrive, Op
```

具体模块路径位于 `natal.frontend.*`（遗传结构/实体/模式均在
`natal.frontend.genetics`、`natal.frontend.patterns` 等真实包内）。

### 2. `setup()` 返回 Configurator

```python
# v0.2.0 — setup() 返回 Configurator，不是 Builder
configurator = nt.DiscreteGenerationPopulation.setup(species=sp)
print(type(configurator))  # <class 'natal.frontend.configurator.Configurator'>
```

### 3. 运行时修改（新增）

Configurator 支持旧版 Builder 无法做到的运行时修改：

```python
# 构建后直接修改参数，无需重建
pop.update().competition(carrying_capacity=5000)
pop.update().reproduction(eggs_per_female=100)

# 首选参数面：pop.params（经边界校验 + 参数快照）
pop.params.sex_ratio = 0.55
```

### 4. 参数变更

- `female_age_based_survival_rates` → `female_age_based_survival`（所有 `_rates` 后缀已移除）
- `species_scale`、`base_carrying_capacity`、`base_expected_num_new_adult_females` 已删除
- `carrying_capacity` 现在是直接的 普通标量

### 5. `SpatialBuilder` → `SpatialConfigurator`

```python
# v0.2.0 — 先声明 deme 数与种群类型，再走空间构建链
from natal.frontend.spatial import SpatialPopulation

pop = (
    SpatialPopulation.builder(species=sp, n_demes=4, pop_type="age_structured")
    .setup(name="demo", stochastic=False)
    .age_structure(n_ages=4, new_adult_age=1)
    ...
    .migration(adjacency=adjacency, migration_rate=0.1)
    .build()
)
```

注意：空间种群没有 `SpatialPopulation.setup(...)` 静态方法——入口是
`SpatialPopulation.builder(...)`。

## 关键行为变更

1. **立即写入**：Configurator 链式方法立即写入 NumPy 数组，不再延迟到 `build()`。对大多数代码透明。
2. **默认 `Species.unordered=True`**：`A|a` 和 `a|A` 现在产生同一个 `Genotype` 实例。如需追踪亲本起源，设置 `unordered=False`。
3. **Hook 形态统一**：声明式（无参返回 `List[HookOp]`）、单参数回调（`TickContext`）、选择器回调（`selectors={...}`）三种；`(state, config, deme_id)` 三参数签名已不可用。
4. **默认存活率**：年龄结构模型默认所有年龄 100% 存活（原为衰减值）。
5. **运行时参数写入**：`pop.params` / `Op.set_param` / `set_param(config, name, value)`；每次变化记录到 `pop.params_log`。
