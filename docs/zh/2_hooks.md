# Hook 系统

Hook 用于在模拟流程的固定时间点插入用户逻辑。

如果你希望在每个 tick 的某个阶段执行"额外操作"，例如周期投放、条件干预、阈值终止，Hook 是最直接的方式。

## Hook 的作用时机

Hook 的作用时机包括：

- `initialization`：模拟初始化完成后、进入首个 tick 之前。
- `first`：每个 tick 的早期阶段。
- `early`：繁殖步骤（`reproduction`）后、生存步骤（`survival`）前。
- `late`：生存步骤（`survival`）后、衰老步骤（`aging`）前。
- `finish`：模拟结束时。

其中，`initialization` 和 `finish` 是一次性的事件，而 `first`、`early`、`late` 可以根据需要在多个 tick 中重复执行。

选择事件时，建议先明确干预发生在哪个具体的时机，这会显著影响结果解释。

## 声明式 Hook

对于大多数用户，推荐使用 `@nt.hook` 与 `nt.Op.*`，在种群对象上链式注册：

```python
import natal as nt

@nt.hook(event="first", priority=10)
def periodic_release():
    return [
        nt.Op.add(genotypes="Drive|WT", ages=[2, 3, 4], delta=200, when="tick % 7 == 0"),
        nt.Op.scale(genotypes="WT|WT", ages="*", factor=0.98),
    ]


pop = (
    nt.AgeStructuredPopulation
    .setup(
        name="MyPop",
        stochastic=True,
        use_continuous_sampling=False
    )
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state(individual_count={
        "female": {"WT|WT": 1000, "Drive|WT": 0},
        "male": {"WT|WT": 1000, "Drive|WT": 0}
    })
    .survival(
        female_age_based_survival_rates=0.85,
        male_age_based_survival_rates=0.8
    )
    .reproduction(eggs_per_female=50.0)
    .competition(
        low_density_growth_rate=6.0,
        age_1_carrying_capacity=10000
    )
    .hooks(periodic_release)
    .build()
)

pop.run(n_steps=200, record_every=10)
```

这种方式可读性高、维护成本低，也更便于团队复核模型规则。

## `Op` 操作

常用操作包括：

- `Op.add`：增加个体数量。
- `Op.subtract`：减少个体数量。
- `Op.scale`：按比例缩放。
- `Op.set_count`：设置目标数量。
- `Op.kill`：按死亡概率处理。
- `Op.sample`：无放回抽样。
- `Op.stop_if_*`：满足条件时停止运行。包括：
  - `Op.stop_if_below`：当指定基因型的个体数量低于阈值时停止运行。
  - `Op.stop_if_above`：当指定基因型的个体数量高于阈值时停止运行。
  - `Op.stop_if_zero`：当指定基因型的个体数量为零时停止运行。
  - `Op.stop_if_extinction`：当种群个体数量为零时停止运行。

把它们理解为"对状态张量进行声明式变换"。

## 随机性处理

当 Declarative Hook 操作后导致有个体死亡（个体数量少于原有数量）时，根据配置可能进行抽样，以决定哪些个体存活。

Declarative Hook 中的 `Op` 操作会根据种群创建（链式 API 中）时 `setup` 中的 `stochastic` 配置自动选择执行方式：

| 配置 | `Op.scale` / `Op.set_count` / `Op.subtract` | `Op.kill` |
|------|--------------------------------|---------|
| `stochastic=True` | 使用二项分布随机采样 | 使用二项分布决定每个个体的存活 |
| `stochastic=False` | 确定性缩放（直接乘以系数） | 确定性缩放（乘以存活概率） |

当 `stochastic=True` 时，还可以通过 `use_continuous_sampling` 配置选择采样方式：

- `use_continuous_sampling=True`：使用连续采样（使用矩匹配的 Beta/Gamma 分布替代二项/柏松分布）
- `use_continuous_sampling=False`：使用离散采样

声明式 Hook 的优势在于：你只需要用同样的 Op 语法编写规则，系统会根据配置自动在确定性和随机性之间切换，无需修改 Hook 代码。

## 条件表达式（when）

`when` 用于控制操作在何时生效，常见写法：

- `tick == N`
- `tick % N == 0`
- `tick >= N`
- `tick > N`
- `tick <= N`
- `tick < N`

并支持 `and`、`or`、`not` 与括号组合。

示例：

```python
when="tick >= 10 and tick < 50"
when="tick % 7 == 0 and not (tick == 14)"
```

## 多个 Hook 的注册

链式 API 中的 `.hooks()` 方法支持传入多个 Hook 函数：

```python
import natal as nt

@nt.hook(event="first", priority=10)
def release_hook():
    return [nt.Op.add(genotypes="Drive|WT", ages=[2, 3, 4], delta=100, when="tick % 5 == 0")]

@nt.hook(event="late", priority=5)
def culling_hook():
    return [nt.Op.scale(genotypes="WT|WT", ages="*", factor=0.95, when="tick > 50")]

@nt.hook(event="late", priority=0)
def stop_hook():
    return [nt.Op.stop_if_above(genotypes="Drive|WT", threshold=5000)]

pop = (
    nt.AgeStructuredPopulation
    .setup(stochastic=True)
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state(individual_count={
        "female": {"WT|WT": 1000},
        "male": {"WT|WT": 1000}
    })
    .hooks(release_hook, culling_hook, stop_hook)
    .build()
)

pop.run(n_steps=100, record_every=10)
```

如果存在多个 Hook，建议通过 `priority` 明确执行顺序，避免隐式顺序导致结果难以复现。

## 执行模式

NATAL Core 的 Hook 系统支持两种执行模式，受全局 `NUMBA_ENABLED` 开关控制：

- **当 `NUMBA_ENABLED=True` 时（默认）**：
  - 声明式 Hook 会被编译为纯数据结构（CSR 格式），在 Numba 编译的内核中高效执行
  - 自定义 Hook 和 Selector-based Hook 需要遵循 Numba 语法
  - Python 层 Hook 会在注册阶段被拒绝（`initialization` 和 `finish` 事件除外）

- **当 `NUMBA_ENABLED=False` 时**：
  - 任意已注册 Hook 类型（declarative CSR、njit、Python）都会在 `run(...)` / `run_tick()` 中走统一的 Python 事件调度路径
  - 在该路径下，系统会根据 `priority` 按顺序执行所有 Hook，无需手动触发

当全局 `NUMBA_ENABLED=True` 时，如果同一事件混用了 declarative CSR、njit、Python 三类 Hook，运行时会自动切到统一 Python 事件调度，确保跨类型按 `priority` 排序执行。

在 `SpatialPopulation` 中，local Hook 的 `priority` 只在 deme 内部生效；不同 deme 之间不定义全局顺序。

## 与 `run` / `run_tick` 的关系

Hook 会在 `run(...)` 与 `run_tick()` 中按事件顺序自动执行。

因此，用户通常不需要手动触发 Hook；只需：

1. 定义 Hook。
2. 在链式 API 中用 `.hooks()` 注册。
3. 正常运行模拟。

## 最简示例

```python
import natal as nt

@nt.hook(event="first", priority=0)
def release():
    return [nt.Op.add(genotypes="Drive|WT", ages=[2, 3, 4], delta=100, when="tick % 5 == 0")]

@nt.hook(event="late", priority=5)
def stop_if_no_female():
    return [nt.Op.stop_if_zero(sex="female", threshold=10000)]

pop = (
    nt.AgeStructuredPopulation
    .setup(stochastic=True)
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state(individual_count={
        "female": {"WT|WT": 1000},
        "male": {"WT|WT": 1000}
    })
    .hooks(release, stop_if_no_female)
    .build()
)

pop.run(n_steps=200, record_every=10)
```

## 相关章节

- [高级 Hook 教程](3_advanced_hooks.md)
- [种群初始化](2_population_initialization.md)
- [Modifier 机制](3_modifiers.md)
- [模拟内核深度解析](4_simulation_kernels.md)
- [Numba 优化指南](4_numba_optimization.md)
- [快速开始](1_quickstart.md)
