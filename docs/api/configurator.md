# configurator Module

Parameter configuration — build and runtime modification of population models.

## Overview

`Configurator` is the unified API for setting and modifying simulation parameters.
It replaces the legacy `PopulationBuilder` classes with a single class that works
identically at build time and runtime.

Key features:

- **Fluent chain API** — `.competition(K=10000).reproduction(eggs=50).build()`
- **Immediate writes** — every chain method writes to NumPy arrays in-place
- **Runtime modification** — `pop.update().competition(K=5000)` without rebuilding
- **Model-specific subclasses** — `DiscreteConfigurator` / `AgeStructuredConfigurator`
  with narrowed parameter signatures
- **Preset/modifier/fitness** — applied directly to config arrays, no deferred execution
- **Equilibrium sync** — `carrying_capacity` / `eggs_per_female` / `sex_ratio` changes
  auto-trigger `sync_equilibrium_metrics`

## Quick Start

```python
import natal as nt

sp = nt.Species.from_dict(name="demo", structure={"auto": {"A": ["WT", "Var"]}})

# Build-time
pop = (
    nt.DiscreteGenerationPopulation.setup(sp)
    .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
    .reproduction(eggs_per_female=50, sex_ratio=0.5)
    .competition(carrying_capacity=10000, low_density_growth_rate=6.0)
    .custom(temperature=25.0)
    .build()
)

# Runtime
pop.update().competition(carrying_capacity=5000)
pop.update().reproduction(eggs_per_female=100, sex_ratio=0.6)
```

## DiscreteConfigurator

`DiscreteGenerationPopulation` 的专属配置器。参数只展示离散模型相关字段。

```python
# 创建
cfg = nt.Configurator.for_discrete(species)

# 或通过 setup()
cfg = nt.DiscreteGenerationPopulation.setup(species)

# 链式配置
cfg.age_structure(n_ages=2, new_adult_age=1)        # 固定 2 年龄
cfg.reproduction(
    eggs_per_female=50,             # 每雌产卵数
    sex_ratio=0.5,                  # 后代雌性比例
    female_adult_mating_rate=1.0,   # 成年雌性交配概率
    male_adult_mating_rate=1.0,     # 成年雄性交配概率
)
cfg.survival(
    female_age0_survival=0.9,       # 雌性幼体存活率
    male_age0_survival=0.9,         # 雄性幼体存活率
)
cfg.competition(
    carrying_capacity=10000,        # 均衡承载力 K
    low_density_growth_rate=6.0,    # 低密度增长率 r
    juvenile_growth_mode="concave", # 密度制约模式
)
```

## AgeStructuredConfigurator

`AgeStructuredPopulation` 的专属配置器。支持 per-age 数组参数和 Champer 均衡模型。

```python
cfg = nt.Configurator.for_age_structured(species)

cfg.age_structure(n_ages=8, new_adult_age=2)

# 所有 per-age 参数支持灵活输入:
#   标量 — 填满所有年龄
#   列表 — 逐年龄指定
#   字典 — 稀疏映射 {age: value}
#   callable — lambda age: ...
cfg.reproduction(
    eggs_per_female=100,
    sex_ratio=0.5,
    female_age_based_mating_rates=[0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.3, 0.0],
    use_sperm_storage=True,
)
cfg.survival(
    female=[1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.5, 0.0],
    male=[1.0, 0.9, 0.85, 0.8, 0.7, 0.5, 0.3, 0.0],
)
cfg.competition(
    carrying_capacity=5000,
    low_density_growth_rate=6.0,
    juvenile_growth_mode="logistic",
    competition_strength=5.0,
    # Champer 模型 — 自定义均衡分布
    equilibrium_distribution=custom_dist,
)
```

## Shared Methods

两种 Configurator 都有以下方法：

### `setup(**flags)`
```python
cfg.setup(name="MyPop", stochastic=False)
```
配置模拟标志和种群名称。

### `initial_state(distribution, sperm_storage=None)`
```python
cfg.initial_state({
    "female": {"WT|WT": [0, 200, 150, 100]},
    "male":   {"WT|WT": [0, 200, 150, 100]},
})
```
设置初始种群分布。

### `custom(**fields)`
```python
cfg.custom(temperature=25.0, debug=True)
```
注册自定义字段，存入 `config.custom`。Hook 内通过 `config.custom['name'][()]` 读写。

### `presets(*presets)`
```python
cfg.presets(homing_drive)
```
应用基因驱动预设。立即写入 config——非延迟执行。

### `modifiers(gamete_modifiers=None, zygote_modifiers=None)`
```python
cfg.modifiers(gamete_modifiers=[my_mod])
```
注册配子/合子修饰器，立即重建基因型/配子映射。

### `fitness(viability=None, fecundity=None, sexual_selection=None, zygote_viability=None, mode="replace")`
```python
cfg.fitness(
    viability={"WT|WT": 0.8, "WT|Var": 1.0},
    fecundity={"female": {"WT|Var": 1.2}},
    mode="multiply",
)
```
写入适应度数组。支持平铺 dict（两性相同）和嵌套 dict（分雌雄）。`mode="replace"` 覆盖，`mode="multiply"` 乘法缩放。

### `hooks(*hook_items)`
```python
cfg.hooks(my_hook)
```
注册事件 hooks，传给 `build()` 时的 Population 构造函数。

### `build(name=None, hooks=None)`
```python
pop = cfg.build(name="MyPop")
```
执行 equilibrium sync 并创建 Population 对象。

### `apply()`
```python
cfg.apply()
```
单独执行 equilibrium sync（不创建 Population）。通常不需要显式调用——`build()` 内部自动调用。

## Runtime Modification

### `pop.update()`
```python
# 单参数
pop.update().competition(carrying_capacity=5000)

# 链式多参数
pop.update().reproduction(eggs_per_female=100).competition(K=10000)

# 自定义字段
pop.update().custom(temperature=35.0)
```

所有修改立即写入 config 的 0-d ndarray——无需 `freeze()` 或重新构建。

### Hook 内修改
```python
@nt.hook(event="early", custom=True)
def my_hook(state, config, deme_id):
    config.carrying_capacity[()] = 5000
    config.custom['temperature'][()] = 40.0
```

### Spatial Population
```python
# 修改所有 deme
pop.update().competition(carrying_capacity=5000)

# 修改单个 deme（clone-on-write）
pop.update(deme=3).competition(carrying_capacity=8000)

# 批量 per-deme 修改
from natal.spatial_builder import batch_setting
pop.update().competition(
    carrying_capacity=batch_setting([100, 200, 300, 400])
)
```

## 底层接口

### `set_param(config, name, value)`
```python
from natal.configurator import set_param
set_param(config, "competition.carrying_capacity", 5000.0)
set_param(config, "carrying_capacity", 5000.0)  # 短名也支持
```
所有高层 API 的底层实现。从 `parameters.py` 注册表解析参数名，定位 config 字段和索引，原地写入。equilibrium-sensitive 参数（K/eggs/sr）自动触发 sync。

### `Configurator.for_config(config)`
```python
cfg = nt.Configurator.for_config(pop.config)
```
根据 config 类型返回 `DiscreteConfigurator` 或 `AgeStructuredConfigurator`。

## 类型层次

```
Configurator                  # 基类：setup, build, apply, preset, fitness, hooks...
├── DiscreteConfigurator      # + competition(离散), reproduction(离散), survival(离散)
└── AgeStructuredConfigurator # + competition(Champer), reproduction(per-age), survival(per-age)
```

## 向后兼容

旧 `PopulationBuilder` 类仍可通过 `legacy_path=True` 使用：

```python
pop = nt.DiscreteGenerationPopulation.setup(sp, legacy_path=True)
    .initial_state(...).build()
```

Configurator 已完全覆盖 Builder 的功能——`legacy_path` 默认值为 `False`，新代码不需要显式指定。
