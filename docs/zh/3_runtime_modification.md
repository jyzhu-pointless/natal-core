# 运行时参数修改

种群构建完成后，所有参数都可以在模拟运行时动态修改——无需重建。覆盖三种场景：

- **between-tick**：Python 侧通过 `pop.update()` 或 `set_param()` 修改
- **hook 内**：Numba nopython 直接写 `config.field[()] = v`
- **spatial**：per-deme 修改 + clone-on-write

---

## 1. between-tick 修改：`pop.update()`

`pop.update()` 返回当前 config 的 `Configurator` 包装。链式方法和构建时完全一样，修改后立即生效：

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

# 单个参数
pop.update().competition(carrying_capacity=5000)

# 链式多个参数
pop.update().reproduction(eggs_per_female=100, sex_ratio=0.6)

# 自定义字段（Hook 可读写）
pop.update().custom(temperature=35.0)
```

每次调用内部走 `set_param(config, name, value)` → 原地写 0-d ndarray。

---

## 2. between-tick 修改：`set_param()` 底层接口

`pop.update()` 的底层实现。适合脚本、notebook：

```python
from natal.configurator import set_param

set_param(pop.config, "competition.carrying_capacity", 5000.0)

# 全名、短名、别名均可
set_param(pop.config, "carrying_capacity", 5000.0)
set_param(pop.config, "reproduction.eggs_per_female", 100.0)
set_param(pop.config, "eggs_per_female", 100.0)  # 别名
```

内部四步：

1. 查 `parameters.py` 注册表：全名 → 短名 → 别名
2. 定位 config 字段和数组索引
3. 原地写入：`config.carrying_capacity[()] = 5000.0`
4. K / eggs / sex_ratio 修改后自动 `sync_equilibrium_metrics`

---

## 3. Hook 内修改

Hook 签名统一为 ``(state, config) → int``。``config`` 可原地修改，修改后对当前 tick 后续 hook 和流程立即可见。Spatial 模型如需在函数体内按 deme 分支，可加可选的 ``deme_id`` 参数，但绝大多数场景不需要。

### 3.1 方式 A：直接写 `config.field[()] = v`

最快路径。Numba nopython，纯 C 级 ndarray 操作：

```python
from natal.population_config import DiscretePopulationConfig
from natal.population_state import DiscretePopulationState

@nt.hook(event="early", custom=True)
def environment_change(
    state: DiscretePopulationState,
    config: DiscretePopulationConfig,
) -> int:
    if state.n_tick == 10:
        config.carrying_capacity[()] *= 0.5
        config.eggs_per_female[()] *= 0.7
        config.custom['temperature'][()] = 40.0
    return 0
```

> 直接写不会自动 sync equilibrium。Age-structured 模型需要手动调 `sync_equilibrium_metrics(config)`。

### 3.2 方式 B：`hook_set_param(config, "name", v)`

封装了 `objmode` + `set_param`。性能与裸 `with objmode()` 完全相同（同一个 Numba→Python 边界），但语法更简洁：

```python
from natal.configurator import hook_set_param

@nt.hook(event="early", custom=True)
def recovery_hook(state, config):
    if state.n_tick == 10:
        hook_set_param(config, "carrying_capacity", 5000.0)
        hook_set_param(config, "eggs_per_female", 100.0)
    return 0
```

每次调用单独跨越一次 objmode 边界。**批量修改**多个参数时，裸 `with objmode()` 更高效——一次边界完成多次 `set_param`。

### 3.3 方式 C：裸 `with objmode()`

需要 Hook 内执行日志、文件 I/O 等任意 Python 操作，或者批量修改参数时：

```python
from numba import objmode
from natal.configurator import set_param

@nt.hook(event="early", custom=True)
def batch_hook(state, config):
    if state.n_tick == 10:
        with objmode():
            print(f"[tick={state.n_tick}] emergency recovery")  # 日志
            set_param(config, "carrying_capacity", 5000.0)
            set_param(config, "eggs_per_female", 100.0)
            set_param(config, "sex_ratio", 0.5)
    return 0
```

### 对比

| 方式 | 性能 | 推荐场景 |
|---|---|---|
| `config.field[()] = v` | 最快（nopython） | 你知道字段名 |
| `hook_set_param(config, "name", v)` | objmode 边界（单次便捷） | 需要字符串参数名 |
| `with objmode(): set_param(...)` | objmode 边界（批量高效） | 批量修改或需要 Python 生态 |

---

## 4. 自定义字段 `config.custom`

0-d structured numpy array。构建时通过 `.custom()` 注册字段和初始值，Hook 内 `[()]` 读写，运行时 `pop.update().custom()` 修改：

```python
# 构建
pop = (
    nt.DiscreteGenerationPopulation.setup(sp)
    .custom(temperature=25.0, season_idx=0)
    .build()
)

# Hook 内
@nt.hook(event="early", custom=True)
def seasonal_hook(state, config):
    temp = config.custom['temperature'][()]
    if int(config.custom['season_idx'][()]) == 1:
        config.custom['temperature'][()] = 35.0

# 运行时
pop.update().custom(temperature=35.0, season_idx=1)
```

支持 `bool`、`float`、`int`。

> **注意**：自定义字段不在参数注册表中，因此 `set_param()` 和 `hook_set_param()` 无法访问它们。应在 hook 中使用直接数组写入 (`config.custom["temperature"][()] = value`) 或 `pop.update().custom(temperature=30.0)`。

---

## 5. 空间种群 per-deme 修改

`SpatialPopulation.update()` 接口与 panmictic 一致，额外支持 per-deme 和批量修改：

```python
from natal.spatial_builder import batch_setting

# 全部 deme
pop.update().competition(carrying_capacity=5000)

# 单个 deme（自动 clone-on-write）
pop.update(deme=3).competition(carrying_capacity=8000)

# 批量（None = 跳过该 deme）
pop.update().competition(
    carrying_capacity=batch_setting([100, None, 300, None])
)
```

**Clone-on-write**：同构空间种群中多个 deme 共享相同的 0-d ndarray。修改单个 deme 时，先复制这些数组为私有副本，确保其他 deme 不受影响。

---

## 6. 底层机制

所有修改方式最终落在同一个操作上：

```
set_param / pop.update() / hook 内直接写
  → config.carrying_capacity           # 0-d ndarray
  → carrying_capacity[()] = 5000.0     # 原地写（原子操作）
  → sync_equilibrium_metrics(config)   # K/eggs/sr 自动触发
```

9 个生态参数（K、eggs、sex_ratio、sperm_displacement_rate、low_density_growth_rate、juvenile_growth_mode、generation_time、expected_competition_strength、expected_survival_rate）均为 0-d ndarray。

### `set_config()` — 整体配置替换

`pop.set_config(new_config)` 一次性替换种群的整个配置对象。适用于从头重建配置后（例如修改了 custom 字段结构）。新配置必须与原有配置类型相同（`PopulationConfig` 或 `DiscretePopulationConfig`）。

Configurator 的 `custom()` 方法在添加新字段时会触发此路径：它会重建 custom 结构化数组并调用 `set_config()` 将新配置写回种群。

---

## 7. 参数参考

参数按领域分组，与 Configurator 链式 API 方法对应。

| 领域 | 参数名 | 别名 | 适用模型 | set_param |
|---|---|---|---|---|
| setup | `stochastic` | — | both | ❌ 构建时 |
| setup | `continuous_sampling` | — | both | ❌ 构建时 |
| setup | `fixed_egg_count` | — | both | ❌ 构建时 |
| setup | `has_sex_chromosomes` | — | both | ❌ 构建时 |
| age_structure | `n_ages` | — | age-structured | ❌ 构建时 |
| age_structure | `new_adult_age` | — | age-structured | ❌ 构建时 |
| age_structure | `generation_time` | — | age-structured | ❌ 构建时 |
| survival | `female_age_based_survival` | — | age-structured | ✅ |
| survival | `male_age_based_survival` | — | age-structured | ✅ |
| reproduction | `eggs_per_female` | `expected_eggs_per_female` | both | ✅ |
| reproduction | `sex_ratio` | — | both | ✅ |
| reproduction | `sperm_displacement_rate` | — | both | ✅ |
| competition | `carrying_capacity` | — | both | ✅ |
| competition | `low_density_growth_rate` | — | both | ✅ |
| competition | `juvenile_growth_mode` | `growth_mode` | both | ✅ |
| fitness | `viability` | — | both | ❌ 张量 |
| fitness | `fecundity` | — | both | ❌ 张量 |
| fitness | `sexual_selection` | — | both | ❌ 张量 |
| fitness | `zygote_viability` | — | both | ❌ 张量 |
| migration | `migration_rate` | — | spatial | 仅空间 |

## 8. 新旧对比

| | 旧（Builder） | 新（Configurator） |
|---|---|---|
| 构建后修改 | 不支持 | `pop.update()` |
| Hook 内修改 | 声明式 Op | `config.field[()] = v` |
| 自定义字段 | ConfigMutator（已删除） | `config.custom` |
| 空间 per-deme | 不支持 | `pop.update(deme=N)` + batch_setting |
| 底层接口 | 无 | `set_param(config, name, value)` |
