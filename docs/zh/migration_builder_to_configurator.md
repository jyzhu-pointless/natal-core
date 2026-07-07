# Builder → Configurator 迁移指南

v0.2.0 中，`PopulationBuilder` 及其子类（`DiscreteGenerationPopulationBuilder`、`AgeStructuredPopulationBuilder`、`SpatialBuilder`）已被 `Configurator` 链式 API 取代。

## 变更对照

| 之前 (v0.1.x) | 之后 (v0.2.0) |
|---|---|
| `PopulationBuilder(species).build()` | `DiscreteGenerationPopulation.setup(species).build()` |
| `.competition(carrying_capacity=...)` 返回 Builder | 返回 `Configurator`（链式语法相同） |
| 参数延迟到 `build()` 才写入 | 参数立即写入配置数组 |
| `SpatialBuilder` 类 | `SpatialConfigurator`（通过 `pop.update()`） |
| `PopulationBuilder` 类 | 通过 `setup(legacy_path=True)` 访问 |

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
# v0.1.x（已删除）
from natal.population_builder import PopulationBuilder
from natal.genetic_presets import HomingDrive

# v0.2.0
from natal import HomingDrive  # 或 nt.HomingDrive
```

### 2. `setup()` 返回 Configurator

```python
# v0.2.0 — setup() 返回 Configurator，不是 Builder
configurator = nt.DiscreteGenerationPopulation.setup(species=sp)
print(type(configurator))  # <class 'natal.configurator.discrete.DiscreteConfigurator'>
```

### 3. 旧版 Builder 路径

如果依赖旧版 Builder API，向 `setup()` 传入 `legacy_path=True`：

```python
builder = nt.DiscreteGenerationPopulation.setup(species=sp, legacy_path=True)
```

### 4. 新增运行时修改

Configurator 支持旧版 Builder 无法做到的运行时修改：

```python
# 构建后直接修改参数，无需重建
pop.update().competition(carrying_capacity=5000)
pop.update().reproduction(eggs_per_female=100)
```

### 5. 参数变更

- `female_age_based_survival_rates` → `female_age_based_survival`（所有 `_rates` 后缀已移除）
- `species_scale`、`base_carrying_capacity`、`base_expected_num_adult_females` 已删除
- `carrying_capacity` 现在是直接的 0-d ndarray

### 6. `SpatialBuilder` → `SpatialConfigurator`

```python
# v0.1.x
builder = SpatialBuilder(species, topology)
pop = builder.build()

# v0.2.0
from natal.spatial import SpatialPopulation
pop = SpatialPopulation.setup(species=sp, topology=grid).build()
```

## 关键行为变更

1. **立即写入**：Configurator 链式方法立即写入 NumPy 数组，不再延迟到 `build()`。对大多数代码透明。
2. **默认 `Species.unordered=True`**：`A|a` 和 `a|A` 现在产生同一个 `Genotype` 实例。如需追踪亲本起源，设置 `unordered=False`。
3. **Hook 签名统一**：`(state, config, deme_id=-1)`。旧版 `(ind_count, tick)` 不再可用。
4. **默认存活率**：年龄结构模型默认所有年龄 100% 存活（原为衰减值）。
5. **`set_param()` / `hook_set_param()`**：新增底层 API，支持按参数名进行运行时修改。
