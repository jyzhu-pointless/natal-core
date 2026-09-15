# Hook 系统

Hook 用于在模拟流程的固定时间点插入用户逻辑。

如果你希望在每个 tick 的某个阶段执行"额外操作"，例如周期投放、条件干预、阈值终止，Hook 是最直接的方式。

## Hook 的作用时机

Hook 的作用时机包括：

- `first`：每个 tick 的早期阶段。
- `early`：繁殖步骤（`reproduction`）后、生存步骤（`survival`）前。
- `late`：生存步骤（`survival`）后、衰老步骤（`aging`）前。
- `finish`：模拟结束时，不属于任何单个 tick。

其中 `finish` 是一次性的事件，而 `first`、`early`、`late` 可以根据需要在多个 tick 中重复执行。

Hook 注册和手动触发（`pop.trigger_event(...)`）共用这一事件目录：未知或拼错的事件名会直接抛出 `ValueError`（错误信息包含传入名称与合法名称），不会被静默忽略。`initialization` 不是受支持的事件；初始化逻辑应写在 `first` 事件的首个 tick（`when="tick == 1"`）或 `finish` 事件中。

选择事件时，建议先明确干预发生在哪个具体的时机，这会显著影响结果解释。

## 声明式 Hook

对于大多数用户，推荐直接把带有自身 `event` 和 `priority` 的 `nt.Op.*` 对象传给 `.hooks()`，在种群构建链上声明：

```python
import natal as nt

sp = nt.Species.from_dict(name="demo", structure={"auto": {"A": ["WT", "Var"]}})

pop = (
    nt.AgeStructuredPopulation
    .setup(species=sp, name="MyPop", stochastic=True, continuous_sampling=False)
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state(individual_count={
        "female": {"WT|WT": 1000, "Var|WT": 0},
        "male": {"WT|WT": 1000, "Var|WT": 0}
    })
    .survival(
        female_age_based_survival=0.85,
        male_age_based_survival=0.8
    )
    .reproduction(eggs_per_female=50.0)
    .competition(
        low_density_growth_rate=6.0,
        age_1_carrying_capacity=10000
    )
    .hooks(
        [
            nt.Op.add(genotypes="Var|WT", ages=[2, 3, 4], delta=200, when="tick % 7 == 0", event="first", priority=10),
            nt.Op.scale(genotypes="WT|WT", ages="*", factor=0.98, event="first", priority=10),
        ],
    )
    .build()
)

pop.run(n_steps=200, record_every=10)
```

`Op` 对象本身就是声明：构建时（`build()`）被编译为 CSR 计划注入种群，不再需要包裹函数。这种方式可读性高、维护成本低，也更便于团队复核模型规则。

## Hook 的编写形态

- **声明式（Declarative）**：直接把带有自身事件和优先级的 `Op` 对象（或其列表）传给 `.hooks()`，构建期编译为 CSR 计划。这是推荐写法。
- **回调（Callback）**：单参数 `def hook(pop: TickContext) -> int`，用 `@nt.hook` 装饰后传入（也可直接传裸函数，事件在 `.hooks(..., event=...)` 上指定）。
- **选择器回调（Selector）**：`@nt.hook(..., selectors={...})`，选择器值在构建期解析、调用时注入。

`@nt.hook` 装饰器在构建期间按函数签名识别后两类形态；零参数、返回 `List[HookOp]` 的函数仍是声明式 Hook 的兼容入口：构建流程可能先调用它收集引用，再调用它进行编译，返回的 Op 都进入正常管线。推荐直接声明带有 Op 本地 `event` / `priority` 的 Op。只有需要动态组装列表时才使用工厂形式；其装饰器和调用级元数据仍遵循下文的优先级规则。

旧的 `(state, config, deme_id)` 三参数签名已被显式拒绝（`TypeError` —— 该签名是 njit 时代的遗物，没有迁移通道）。回调 Hook 返回值 `0`（或 `RESULT_CONTINUE`）继续模拟，非零值（或 `RESULT_STOP`）停止模拟。

四个生命周期事件（`first`、`early`、`late` 和 `finish`）全部由 Rust native session 执行。旧的 Python CSR 执行器、采样器和低层执行导出已删除；当前只保留编译后的 `HookProgram` 数据和 Python 回调桥接。

`.hooks()` 是声明 Hook 的唯一入口，且只存在于构建链式 API 中：Hook 计划在 `build()` 时基于最终 registry 一次性编译并注入种群。构建完成后不存在任何注册通道——`pop.update()` 返回的运行时更新句柄上根本没有 `hooks()`（访问即抛出 `AttributeError`）。如需在运行期改变行为，请在构建时声明 Hook（可用 `when` 条件或回调内的 tick 判断控制触发），并在需要时手动触发同一事件。

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
- `Op.set_param`：按 tick 计划表调度一个生态参数（见下文）。
- `Op.convert`：一对一概率性基因型转换（见下文）。

把它们理解为"对状态张量进行声明式变换"。所有 `Op` 工厂方法都接受 `event` 与 `priority` 关键字参数。推荐把它们写在 Op 上。为兼容已有代码，`.hooks(..., event=...)` 可为缺少事件的 Op 提供默认值，但 Op 显式指定的事件优先；`.hooks(..., priority=...)` 可为一次声明中的 Op 赋同一优先级，并覆盖 Op 本地优先级。

### `Op.set_param`：无代码参数调度

`Op.set_param(param, value, every=1, start=0, when=None)` 按 tick 计划重写一个运行时可变生态标量：

```python
nt.Op.set_param("carrying_capacity", "K * 0.95", every=10, event="early", priority=0)
```

- `value` 是**算术表达式**（RPN 编译）：操作数是 jsonc 参数名（`K` 是 `carrying_capacity` 的注册别名）或数字字面量，运算符是 `+ - * /`，支持括号。纯数字等价于常量表达式。表达式**每次触发时对当前值求值**，因此 `"K * 0.95"` 会复利递减。
- `every` / `start` 控制触发计划：`tick >= start and (tick - start) % every == 0`；`when` 提供额外条件。
- `event` 参数默认 `early`。
- 目标是以下 **5 个生态参数**（同一组可由 `ctx.params` 直接属性写入、并由 Rust 会话作为列持有的生态标量）：

| 参数名 | 说明 |
|---|---|
| `carrying_capacity` | K |
| `eggs_per_female` | 雌性产卵数 |
| `sex_ratio` | 性比 |
| `sperm_displacement_rate` | 精子置换率 |
| `low_density_growth_rate` | 低密度增长率 |

向量/张量参数会抛 `ValueError` —— 请改用 `pop.update()` / `pop.params.tensor_write()`。

**写入语义**：

- 运行外，写入通过 `pop.params.<name> = ...` 相同的通道（路由分派、会话刷新、参数快照日志）生效。
- `run()` 运行中，写入在会话拥有的生态列内部演化（事件粒度相同、jsonc 边界校验相同；非有限值或越界值如 `"K / 0"` 会在运行中抛 `ValueError`）。每次成功变化都会在事件边界写入 native `ParameterLog`，后续读取从会话快照取得当前值。

### `Op.convert`：按个体条件转换

用 `from_` 选择参与转换的个体，用 `to` 描述改变哪些属性。两者都使用 `IndividualSelector`；Python 的 `from` 是保留字，因此参数写作 `from_`。

下面的声明假设种群包含所用标签，且至少有三个年龄：

```python
nt.Op.convert(
    from_=nt.IndividualSelector(age=2, ztype="*@uninfected"),
    to=nt.IndividualSelector(age=1, ztype="*@infected"),
    probability=0.25,
    event="late",
)
```

它把选中个体的 25% 转到 1 龄、感染标签，各自的基因型和性别保持不变。确定性模式按比例转移计数；随机模式抽样确定转换个体。

- `from_` 中省略的字段和通配符表示不限。selector 并集保留条件之间的组合关系，重叠坐标只处理一次。
- `to` 中省略的字段、未写出的染色体组和 `*` 表示保留对应来源值。`*@infected` 只改标签；`A|A@*` 替换给出的基因型部分，保留标签和省略的染色体组。
- 每个来源必须得到一个合法目标。目标不接受并集、集合、否定或多个年龄／性别值。目标不在当前类型目录中时抛出 `ValueError`，不会在 Hook 中动态创建类型。
- 同一条 Op 根据执行前的状态计算全部转移，不会再次转换本条操作刚转入的个体。不同 Op 仍按 Hook 执行顺序级联。
- 转换不会跳过后续生命周期阶段。例如在 `late` 修改年龄后，仍会正常推进年龄。

雌性转换后仍为雌性时，精子存储随个体迁移，包括成年回退到 `new_adult_age` 以下。雌转雄只丢弃实际转换雌性携带的精子；雄转雌从未交配状态开始。其他雌性已存储的精子，不因其来源雄性改变类型或性别而被改写。转移保持个体总数不变。

旧字符串写法继续可用：

```python
nt.Op.convert("A|A", "A|a", probability=0.3, event="early"),
nt.Op.convert("A|A", "a|a", probability=1.0, event="early"),
```

旧写法的来源和目标必须各自只匹配一个不同的 ZType，对全部年龄、两种性别执行。不要与 `from_`／`to` 参数混用。第二条转换处理第一条留下的剩余个体；不另设多目标概率分配接口。

### `Op.clear_sperm_storage`：清除选中雌性的精子存储

```python
nt.Op.clear_sperm_storage(
    selector=nt.IndividualSelector(sex="female", age=1),
    event="late",
)
```

清空选中雌性携带的全部精子存储，个体计数不变，后续按未交配状态处理。selector 选择携带精子的雌性，不筛选存储中的雄性类型。选中的雄性坐标会被忽略；重复执行或用于没有精子存储的模型，不会产生额外变化。

操作按执行时的状态选择个体。放在转换之后的清除操作会影响所有匹配雌性，不会自动只清除刚被转换的个体。


## 随机性处理

当 Declarative Hook 操作后导致有个体死亡（个体数量少于原有数量）时，根据配置可能进行抽样，以决定哪些个体存活。

Declarative Hook 中的 `Op` 操作会根据种群创建（链式 API 中）时 `setup` 中的 `stochastic` 配置自动选择执行方式：

| 配置 | `Op.scale` / `Op.set_count` / `Op.subtract` | `Op.kill` |
|------|--------------------------------|---------|
| `stochastic=True` | 使用二项分布随机采样 | 使用二项分布决定每个个体的存活 |
| `stochastic=False` | 确定性缩放（直接乘以系数） | 确定性缩放（乘以存活概率） |

当 `stochastic=True` 时，还可以通过 `continuous_sampling` 配置选择采样方式：

- `continuous_sampling=True`：使用连续采样（使用矩匹配的 Beta/Gamma 分布替代二项/泊松分布）
- `continuous_sampling=False`：使用离散采样

声明式 Hook 的优势在于：你只需要用同样的 Op 语法编写规则，系统会根据配置自动在确定性和随机性之间切换，无需修改 Hook 代码。

## 条件表达式（when）

`when` 用于控制操作在何时生效，常见写法：

- `tick == N`
- `tick % N == 0`（`N` 必须是正整数字面量；`N = 0` 在解析期报 `ValueError`）
- `tick >= N`
- `tick > N`
- `tick <= N`
- `tick < N`

`N` 只支持非负整数字面量，不支持变量引用或其他表达式。并支持 `and`、`or`、`not` 与括号组合。

示例：

```python
when="tick >= 10 and tick < 50"
when="tick % 7 == 0 and not (tick == 14)"
```

## 多个 Hook 的注册

链式 API 中的 `.hooks()` 方法可以调用多次，也可以一次传入多个条目（Op、Op 列表、回调函数混排）。

`priority` 是 Op 级数据：调用级 `priority` 为本次声明的 Op 赋同一个优先级；把 Op 打包成列表本身就是声明
整组共用一个优先级（不指定时，组员自带的 `priority` 必须一致，否则构建期报 `ValueError`）。装饰器函数不受调用级 `priority` 影响，保持装饰器上的值。`event` 的解析方向不同：Op 自带的
`event`（如 `Op.set_param(..., event="late")`）优先于调用级，调用级优先于装饰器上的值。

```python
import natal as nt

pop = (
    nt.AgeStructuredPopulation
    .setup(species=sp, stochastic=True)
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state(individual_count={
        "female": {"WT|WT": 1000},
        "male": {"WT|WT": 1000}
    })
    .hooks(
        nt.Op.add(genotypes="Var|WT", ages=[2, 3, 4], delta=100, when="tick % 5 == 0", event="first", priority=10),
    )
    .hooks(
        nt.Op.scale(genotypes="WT|WT", ages="*", factor=0.95, event="late", priority=5, when="tick > 50"),
    )
    .hooks(
        nt.Op.stop_if_above(genotypes="Var|WT", threshold=5000, event="late", priority=0),
    )
    .build()
)

pop.run(n_steps=100, record_every=10)
```

如果存在多个 Hook，建议通过 `priority` 明确执行顺序，避免隐式顺序导致结果难以复现。

## 执行路径

Rust 原生引擎是唯一的执行后端，Hook 只有一条执行路径：

- 声明式 `Op` 编译为 CSR 计划（连续数组 + 偏移表），在 Rust 会话内按事件顺序执行。
- 单参数回调（`TickContext`）在事件边界跨 Python↔Rust 桥进入会话，每次调用获得独立的上下文封装；回调写入进入该次调用的事件事务，成功才提交、失败则丢弃本次候选。
- 同一事件内，声明式操作与 Python 回调按 `priority` **跨类型统一排序**（数值小者先执行；同 priority 时按声明顺序）。两类 Hook 的 `priority` 互相可比：无论回调还是声明式操作，`priority` 更小者总是先执行，后面的 Hook 能看到前面 Hook 的写入。

Hook 是"Op 即 hook"的声明式编译模型：`Op` 对象本身构成 hook 程序，直接传入 `.hooks()` 即完成声明。`@hook` 装饰的零参数函数返回 Op 列表仍可作为兼容工厂入口；构建流程可能先调用它收集引用，再调用它进行编译。`initialize` 事件不存在 —— 初始化阶段的逻辑请用 `first` 事件的首个 tick（`when="tick == 1"`）或 `finish` 事件表达。

`SpatialPopulation` 中，local Hook 的 `priority` 只在 deme 内部生效；不同 deme 之间不定义全局顺序。空间模型见 [空间模拟](3_spatial_simulation.md)。

## 与 `run` / `run_tick` 的关系

Hook 会在 `run(...)` 与 `run_tick()` 中按事件顺序自动执行。

因此，用户通常不需要手动触发 Hook；只需：

1. 定义 Hook。
2. 在链式 API 中用 `.hooks()` 注册。
3. 正常运行模拟。

## 最简示例

```python
import natal as nt

sp = nt.Species.from_dict(name="demo", structure={"auto": {"A": ["WT", "Var"]}})

pop = (
    nt.AgeStructuredPopulation
    .setup(species=sp, stochastic=True)
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state(individual_count={
        "female": {"WT|WT": 1000},
        "male": {"WT|WT": 1000}
    })
    .hooks(
        nt.Op.add(genotypes="Var|WT", ages=[2, 3, 4], delta=100, when="tick % 5 == 0", event="first", priority=0),
    )
    .hooks(
        nt.Op.stop_if_zero(sex="female", event="late", priority=5),
    )
    .build()
)

pop.run(n_steps=200, record_every=10)
```

## 单参数回调 Hook（TickContext）

需要直接读写状态数组或运行时参数时，使用单参数回调形态。参数是一个 `TickContext`，公开成员如下：

| 成员 | 类型 / 语义 |
|---|---|
| `pop.tick` | 当前模拟 tick（只读）。 |
| `pop.deme_id` | 本次调用的 deme 索引（panmictic 为 `0`，空间模型为实际 deme 下标，只读）。 |
| `pop.state` | 可写状态视图（短期借用；写入立即生效）。 |
| `pop.params` | 可写参数面（与 `pop.params` 相同的写入器栈；属性写入经边界校验，同时到达 draft、Rust 会话与参数快照日志）。 |
| `pop.blueprint` | 只读维度、名称目录与引擎开关（`n_sexes`、`n_ages`、`n_ztypes`、`ztype_names` 等）。 |
| `pop.metrics` | 按需计算的指标视图（每次访问重新计算）。 |
| `pop.rng` | 确定性随机流（每调用独立；由种群槽位、tick、deme、hook 索引派生，从不触碰全局 `numpy.random`）。 |
| `pop.update()` | 返回绑定到所属种群的 `RuntimeUpdater`（与构建链同语法）。 |
| `pop.stop()` / `pop.stop_requested` | 在事件边界请求/查询终止当前 run。 |

```python
from natal.frontend.hooks.tick_context import TickContext

@nt.hook(event="early", priority=10)
def my_hook(pop: TickContext) -> int:
    pop.state.individual_count[1, :, :] += 100  # 增加 100 只雄性
    pop.params.carrying_capacity = pop.params.carrying_capacity * 0.5
    if pop.tick > 50:
        pop.stop()
    return 0
```

关于基于选择器的 Hook 以及 Hook 内运行时参数修改的细节，参见 [高级 Hook 教程](3_advanced_hooks.md)。

## Hook 内修改参数

参见 [高级 Hook 教程](3_advanced_hooks.md)。

## 相关章节

- [高级 Hook 教程](3_advanced_hooks.md)
- [运行时参数修改](3_runtime_modification.md)
- [种群初始化](2_population_initialization.md)
- [Modifier 机制](3_modifiers.md)
- [PopulationBuilder API 参考](api/population_builder.md)
- [快速开始](1_quickstart.md)
