# 高级 Hook 教程

[基础教程](2_hooks.md) 介绍了声明式 Hook（`Op.add`、`Op.scale` 等），适合大多数常规场景。
当需要直接操作 NumPy 数组进行更灵活的状态修改时（例如条件分支、循环、自定义计算、
钩子内运行时参数写入），可以使用单参数回调 Hook 或 Selector-based Hook。

## 单参数回调 Hook（TickContext）

回调 Hook 允许你直接编写代码操作模拟状态，在每次触发时被调用。参数是一个
`TickContext` 对象，公开成员见下表：

| 成员 | 类型 / 语义 |
|---|---|
| `pop.tick` | 当前模拟 tick（只读）。 |
| `pop.deme_id` | 本次调用的 deme 索引（panmictic 为 `0`，空间模型为实际 deme 下标，只读）。 |
| `pop.state` | 可写事务候选，首次访问 state 或 metrics 时物化，回调成功后提交。 |
| `pop.params` | 可写参数面（与 `pop.params` 相同的写入器栈；属性写入在候选中经过校验，回调成功后进入 Rust 会话与审计日志）。 |
| `pop.blueprint` | 只读维度、名称目录与引擎开关（`n_sexes`、`n_ages`、`n_ztypes`、`discrete`、`stochastic`、`continuous_sampling`、`extreme_speed_mode`、`ztype_names`、`gtype_names`）。 |
| `pop.metrics` | 按需计算的指标视图（每次访问重新计算）。 |
| `pop.rng` | 该 deme 持久 Rust 随机流的受控采样器；从不触碰全局 `numpy.random`。 |
| `pop.update()` | 返回绑定到所属种群的运行时 `Configurator`（与构建链同语法）。 |
| `pop.stop()` / `pop.stop_requested` | 在事件边界请求/查询终止当前 run。 |

只有回调访问参数或配置时，参数候选才会复制到 Python。仅统计调用次数、查看状态或抽取随机数的回调不会传输参数张量。经过校验的写入直接更新原生事务；回调抛出异常时，该回调的候选会被丢弃。

### 基本用法

```python
from natal.frontend.hooks import hook
from natal.frontend.hooks.tick_context import TickContext


@hook(event="late", priority=10)
def custom_release_hook(pop: TickContext) -> int:
    # pop.state.individual_count 是个体数量的 NumPy 数组
    # 形状为 (sex, age, genotype)
    # sex=0 对应雌性，sex=1 对应雄性

    # 每 10 个 tick 释放 100 个个体
    if pop.tick % 10 == 0:
        # 假设 Var|WT 的基因型索引是 1
        pop.state.individual_count[:, :, 1] += 100

    return 0  # 0 表示继续模拟
```

### 数组索引注意事项

`pop.state.individual_count` 的维度顺序为 `(sex, age, genotype)`：

- `sex=0` 对应雌性（FEMALE），`sex=1` 对应雄性（MALE）
- 直接使用整数索引；枚举值需取 `.value`（`Sex.MALE.value`）

```python
# 正确做法
male_count = pop.state.individual_count[1, :, :].sum()
female_count = pop.state.individual_count[0, :, :].sum()

# 或者使用 .value
male_count = pop.state.individual_count[Sex.MALE.value, :, :].sum()
```

### 完整示例（含运行时参数写入）

```python
from natal.frontend.hooks import hook
from natal.frontend.hooks.tick_context import TickContext


@hook(event="early", priority=5)
def custom_culling_hook(pop: TickContext) -> int:
    # 对特定基因型进行选择性剔除
    if pop.tick > 50:
        # WT|WT 的基因型索引是 0
        wt_wt_count = pop.state.individual_count[:, :, 0].sum()
        if wt_wt_count > 10000:
            pop.state.individual_count[:, :, 0] = pop.state.individual_count[:, :, 0] * 0.9

    # hook 内修改运行时参数：立即生效并记录到 params_log
    pop.params.carrying_capacity = float(pop.params.carrying_capacity) * 0.5
    return 0
```

- 返回值 `0`（或 `RESULT_CONTINUE`）继续模拟；非零值（或 `RESULT_STOP`）立即停止。
- `pop.params.<name> = v` 是 hook 内唯一的推荐参数写入通道：写入经过 jsonc 边界校验，
  同一 tick 后续阶段立即可见，并追加一条 `(tick, name, old, new)` 到 `pop.params_log`。
- `stop()` 在 **late 事件中调用时立即停止在事件边界**（本 tick 的其余过程不再执行、
  tick 不递增）；`stop_requested` 可查询该状态。

## Selector-based Hook

Selector-based Hook 是回调 Hook 的一种形式，允许你通过符号名称（如 `"Var|WT"`）
指定目标基因型。框架在注册时自动将符号解析为整数索引；单值选择器折叠为 `int`，
多值选择器以 `int32` ndarray 注入：

### 基本用法

```python
from natal.frontend.hooks import hook
from natal.frontend.hooks.tick_context import TickContext


@hook(event="late", selectors={"target_gt": "Var|WT"}, priority=10)
def cap_target(pop: TickContext, target_gt: int) -> int:
    # target_gt 是通过选择器解析得到的基因型索引（单值时是 int）
    if pop.tick % 10 == 0:
        pop.state.individual_count[:, :, target_gt] *= 0.95
    return 0
```

### 选择器解析规则

`selectors` 的值支持以下类型：

| 类型 | 示例 | 注入值 |
|------|------|---------|
| `str`（基因型标签） | `"WT\|WT"` | 单个索引（`int`） |
| `str`（通配符） | `"*"` | 所有基因型索引（`int32` 数组） |
| `int` | `3` | 直接用作索引（`int`） |
| `range` | `range(3)` | `[0, 1, 2]`（`int32` 数组） |
| `list` / `tuple` | `["WT\|Dr", 4]` | 多个索引（`int32` 数组） |
| `Genotype` 对象 | `species.genotypes[0]` | 对应索引（`int`） |

单值选择器自动拆箱为 `int`，多值选择器保留为 `np.ndarray[int32]`。
函数签名形如 `def hook(pop: TickContext, <selector 名>) -> int`——选择器名即关键字参数名。

### 多选择器示例

```python
@hook(event="early", selectors={"drive": "Var|WT", "wt": "WT|WT"})
def balance_population(pop: TickContext, drive: int, wt: int) -> int:
    drive_count = pop.state.individual_count[:, :, drive].sum()
    wt_count = pop.state.individual_count[:, :, wt].sum()

    if drive_count > wt_count * 2:
        pop.state.individual_count[:, :, drive] *= 0.8

    return 0
```

> **注意**：选择器使用精确字符串匹配，不支持 pattern 语法（`::`、`|*` 等）。
> 如需 pattern 匹配，请在注册前自行调用 `GenotypeSelector` 转换为索引数组。

## Hook 内随机采样

回调 Hook 内的随机性必须来自 `pop.rng`（受控 Rust 采样器）。任何
`np.random` 全局状态都不会被触碰：

```python
from natal.frontend.hooks import hook
from natal.frontend.hooks.tick_context import TickContext


@hook(event="late", priority=10)
def stochastic_culling_hook(pop: TickContext) -> int:
    if pop.tick > 50:
        # 随机采样推进该 deme 的持久 Rust 流。
        # 恢复检查点也会恢复随机流位置。
        survival_prob = 0.9
        n_current = pop.state.individual_count[:, :, 0]
        pop.state.individual_count[:, :, 0] = pop.rng.binomial(
            n_current.astype(int), survival_prob
        ).astype(float)
    return 0
```

采样器支持 `random`、`uniform`、`normal`、`integers` 和 `binomial`；binomial 的计数和概率可以按数组广播。同一回调中重复访问使用同一随机流；失败回调的候选采样被丢弃。使用相同 hook 和参数恢复检查点，可重放之后的采样。回调结束后，保留的 RNG、参数和 update 句柄拒绝访问。

## 执行路径

Rust 原生引擎是唯一的执行后端。声明式 Op 编译为 CSR 计划，在引擎会话内执行；
单参数 Python 回调跨桥进入会话（每个调用获得独立的上下文封装）。
带外入口——`trigger_event` 与 finish 事件——通过 Python 侧解释器执行同一份
CSR 计划。

## 混合使用不同类型的 Hook

同一事件可以混合声明式与回调形态，全部按 `priority`（值越小越先执行）排序：

```python
from natal.frontend.hooks import hook, Op


# 声明式 Hook：定期释放个体
@hook(event="first", priority=10)
def release_hook():
    return [Op.add(genotypes="Var|WT", ages=[2, 3, 4], delta=100, when="tick % 10 == 0")]


# Selector-based Hook：基于选择器的操作
@hook(event="first", priority=7, selectors={"drive": "Var|WT"})
def check_drive_threshold(pop, drive):
    drive_count = pop.state.individual_count[:, :, drive].sum()
    if drive_count > 10000:
        pass
    return 0


# 回调 Hook：轻量死亡率
@hook(event="first", priority=5)
def custom_process_hook(pop):
    for age in range(pop.state.individual_count.shape[1]):
        pop.state.individual_count[:, age, :] *= 0.99
    return 0


pop = (
    nt.AgeStructuredPopulation.setup(species=sp)
    .hooks(release_hook, check_drive_threshold, custom_process_hook)
    .build()
)
```

同一优先级的 Hook 执行顺序不确定；跨后端的优先级语义一致。

## 性能比较

| Hook 类型 | 性能 | 灵活性 | 可读性 | 适用场景 |
|----------|------|--------|--------|----------|
| 声明式 Hook | 高 | 中 | 高 | 大多数常规场景 |
| Selector-based Hook | 高（索引烘焙） | 高 | 中 | 需要基于特定目标执行逻辑的场景 |
| 回调 Hook | 中（Python 回调） | 高 | 中 | 计算密集型、需要读写参数/自定义逻辑的场景 |

声明式 Op 完全在引擎会话内执行，是性能最优路径；回调 Hook 每次触发跨一次
Python↔Rust 边界。

## 运行时修改参数

Hook 内修改参数的推荐通道是 `pop.params`：

```python
import natal as nt
from natal.frontend.hooks.tick_context import TickContext

@nt.hook(event="early")
def heatwave(pop: TickContext) -> int:
    if pop.tick == 10:
        pop.params.carrying_capacity = 2000.0
        pop.params.sex_ratio = 0.55
    return 0
```

语义（各入口统一）：

- 写入经过 jsonc 边界校验，超出 `parameters.jsonc` 中声明的 `bounds` 抛 `ValueError`；
- 同 tick 后续阶段立即生效（tick 内和显式 `trigger_event` 写入都通过所属 Rust 会话提交）；
- 每条实际变化追加到 `pop.params_log`，格式 `(tick, name, old, new)`；
- 向量/张量参数用 `pop.params.tensor_write(name, values)`。

自定义字段通过 `population.config.custom['name']` 读取，构建时用
`.custom(temperature=25.0)` 初始化，运行时可用 `pop.update().custom(...)` 修改。
自定义值保留 bool/int/float 类型以及任意维数数组的形状；空间模型按 deme 独立保存，并纳入检查点。自定义字段不在参数注册表中，`pop.params` 无法访问它们。

Hook 内如需构建链式更新，可用 `pop.update()` 返回的 Configurator（与构建链同语法）。

## 事件事务

- 每个 Python 回调统一提交状态、生态、遗传参数、自定义值和 RNG 位置。非法状态或异常会丢弃该回调的候选；此前成功回调的提交保留。
- 声明式操作按优先级执行，同一事件中后面的操作能看到前面的写入。最终参数变化按原生事件提交，同一参数多次写入合并为一条从初值到终值的审计记录。Python 回调在声明式事件之后逐个独立提交，分别记录日志。
- 普通和空间模型中的成功参数更新，对同一 tick 的后续回调和阶段立即可见。Rust 保存当前值和参数日志，Python 配置读取返回隔离快照。空间 `ctx.params.tensor_write()` 和 deme 参数写入由原生会话为变化的遗传数据建立独立变体，不影响其他 deme。
- 回调异常保留原始 Python 异常类型。失败会话必须 reset 或恢复检查点后才能再次运行。
- `stop()` 在当前事件边界停止，保留该阶段的状态和位置，tick 不递增；继续运行需要 reset 或恢复一个 Ready 检查点。
- 只有回调访问 state 或 metrics 时才生成 Python 状态数组。仅使用参数或 RNG 的回调不会创建 Python 状态数组。

## 相关章节

- [Hook 系统](2_hooks.md) - 基础 Hook 概念和声明式 Hook 使用
- [运行时参数修改](3_runtime_modification.md)
- [Modifier 机制](3_modifiers.md) - 遗传修饰器机制
- [模拟内核深度解析](4_simulation_engine.md) - 模拟内核的工作原理
