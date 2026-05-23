# 运行时参数修改

种群构建完成后，所有参数都可以在模拟运行时动态修改——无需重建种群对象。本章覆盖三种修改方式和底层机制。

## 1. 方式一：`pop.update()` 链式 API

最简单的方式。拿当前 config 包一层 `Configurator`，链式方法即时写入：

```python
import natal as nt

sp = nt.Species.from_dict(name="demo", structure={"auto": {"A": ["WT"]}})
pop = (
    nt.DiscreteGenerationPopulation.setup(sp)
    .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000, low_density_growth_rate=6.0)
    .build()
)

# 运行时修改单个参数
pop.update().competition(carrying_capacity=5000)

# 链式修改多个参数
pop.update().reproduction(eggs_per_female=100, sex_ratio=0.6)

# 修改自定义字段
pop.update().custom(temperature=35.0)
```

每个方法调用 `set_param(config, name, value)` 直接写 0-d ndarray——立即生效，无需 `freeze()` 或重建。

### 工作原理

`pop.update()` 返回一个 `Configurator` 或子类（根据 config 类型自动选择 `DiscreteConfigurator` / `AgeStructuredConfigurator`）。返回的 Configurator 包装的是 population 当前的 config，链式方法和构建时完全一样。修改后对当前和后续 tick 立即可见。

## 2. 方式二：`set_param()`——底层字符串接口

`pop.update()` 的底层实现。适合脚本、notebook、objmode hook：

```python
from natal.configurator import set_param

set_param(pop.config, "competition.carrying_capacity", 5000.0)

# 支持短名和别名
set_param(pop.config, "carrying_capacity", 5000.0)
set_param(pop.config, "reproduction.eggs_per_female", 100.0)
set_param(pop.config, "expected_eggs_per_female", 100.0)  # 别名
```

内部流程：
1. 在 `parameters.py` 注册表中查找参数名（全名 → 短名 → 别名）
2. 定位 `PopulationConfig` 字段和数组索引
3. 原地写入：`config.carrying_capacity[()] = 5000.0`
4. Equilibrium-sensitive 参数（K / eggs / sex_ratio）自动调用 `sync_equilibrium_metrics`

## 3. 方式三：Hook 内直接修改

最快路径——Hook 签名包含 `config` 参数，Numba nopython 直接写：

```python
import natal as nt
from natal.discrete_population_config import DiscretePopulationConfig
from natal.population_state import DiscretePopulationState

@nt.hook(event="early", custom=True)
def environment_change(
    state: DiscretePopulationState,
    config: DiscretePopulationConfig,
    _deme_id: int,
) -> int:
    if state.n_tick == 10:
        # 直接写 0-d ndarray（nopython，最快）
        config.carrying_capacity[()] *= 0.5
        config.expected_eggs_per_female[()] *= 0.7
        # 读写自定义字段
        config.custom['temperature'][()] = 40.0
    return 0
```

### 三种 Hook 内写法的性能对比

| 写法 | 性能 | 适用场景 |
|---|---|---|
| `config.carrying_capacity[()] = v` | 最快（nopython） | 你知道字段名 |
| `set_config_param(config, PARAM_ID, v)` | 快（nopython，整数路由） | 需要动态参数选择 |
| `with objmode(): set_param(config, "name", v)` | 慢（Python 回退） | 需要 Python 生态 |

## 4. 自定义字段——`config.custom`

`config.custom` 是一个 0-d structured numpy array，支持任意命名的标量字段。构建时通过 `.custom()` 注册，运行时 `[()]` 读写：

```python
# 构建时注册
pop = (
    nt.DiscreteGenerationPopulation.setup(sp)
    .custom(temperature=25.0, season_idx=0, debug=False)
    .build()
)

# Hook 内读写
@nt.hook(event="early", custom=True)
def seasonal_hook(state, config, _deme_id):
    season = int(config.custom['season_idx'][()])
    temp = config.custom['temperature'][()]
    if season == 1:
        config.custom['temperature'][()] = 35.0

# 运行时通过 update() 修改
pop.update().custom(temperature=35.0, season_idx=1)
```

支持三种类型：`bool`、`float`、`int`。

## 5. 空间种群——per-deme 修改

`SpatialPopulation.update()` 支持全部 deme 或单个 deme 的修改，与构建时的 `batch_setting` API 一致：

```python
from natal.spatial_builder import batch_setting

# 修改所有 deme
pop.update().competition(carrying_capacity=5000)

# 修改单个 deme（自动 clone-on-write）
pop.update(deme=3).competition(carrying_capacity=8000)

# 批量 per-deme（None 表示跳过该 deme）
pop.update().competition(
    carrying_capacity=batch_setting([100, None, 300, None])
)
```

### Clone-on-write

当多个 deme 共享 config 的 0-d ndarray 时（同构空间种群），修改单个 deme 会先复制这些数组，创建该 deme 的私有副本，确保其他 deme 不受影响。检测通过 `config.carrying_capacity` 的数组 identity 完成。

## 6. 原理：0-d ndarray + 原地写入

所有修改方式最终都落在同一个机制上：9 个生态参数是 0-d ndarray，`field[()] = value` 是原子操作。

```
set_param(config, "carrying_capacity", 5000.0)
  → _resolve_param("carrying_capacity")  # 查 parameters.py
  → config.carrying_capacity             # 0-d ndarray
  → carrying_capacity[()] = 5000.0       # 原地写
  → sync_equilibrium_metrics(config)     # 自动重算竞争指标
```

这保证了无论从 `pop.update()`、`set_param()` 还是 hook 内直接写，行为完全一致。

## 7. 对比：新旧路径

| | 旧（Builder） | 新（Configurator） |
|---|---|---|
| 构建后修改 | 不支持原生接口 | `pop.update()` 链式修改 |
| Hook 内修改 | 只读（需声明式 Op） | 直接写 `config.field[()] = v` |
| 自定义字段 | `ConfigMutator`（已删除） | `config.custom` + `.custom()` |
| 空间 per-deme | 不支持 | `pop.update(deme=N)` + batch_setting |
| 底层接口 | 无 | `set_param(config, name, value)` |
