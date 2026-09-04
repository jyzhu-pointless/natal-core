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
| `pop.deme_id` | 本次调用的 deme 索引（panmictic 为 `-1`，只读）。 |
| `pop.state` | 可写状态视图（短期借用；写入立即生效）。 |
| `pop.params` | 可写参数面（与 `pop.params` 相同的写入器栈；属性写入经边界校验，同时到达 draft、Rust 会话与参数快照日志）。 |
| `pop.blueprint` | 只读维度、名称目录与引擎开关（`n_sexes`、`n_ages`、`n_ztypes`、`discrete`、`stochastic`、`continuous_sampling`、`extreme_speed_mode`、`ztype_names`、`gtype_names`）。 |
| `pop.metrics` | 按需计算的指标视图（每次访问重新计算）。 |
| `pop.rng` | 确定性随机流（由种群槽位、tick、deme、hook 索引派生；从不触碰全局 `numpy.random`）。 |
| `pop.update()` | 返回绑定到所属种群的运行时 `Configurator`（与构建链同语法）。 |
| `pop.stop()` / `pop.stop_requested` | 在事件边界请求/查询终止当前 run。 |

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

回调 Hook 内的随机性必须来自 `pop.rng`（`np.random.Generator`）。任何
`np.random` 全局状态都不会被触碰：

```python
from natal.frontend.hooks import hook
from natal.frontend.hooks.tick_context import TickContext


@hook(event="late", priority=10)
def stochastic_culling_hook(pop: TickContext) -> int:
    if pop.tick > 50:
        # pop.rng 是本次调用独立的确定性流：同一 (槽位, tick, deme, hook 索引)
        # 在参考后端与 Rust 后端给出相同的抽取序列
        survival_prob = 0.9
        n_current = pop.state.individual_count[:, :, 0]
        pop.state.individual_count[:, :, 0] = pop.rng.binomial(
            n_current.astype(int), survival_prob
        ).astype(float)
    return 0
```

随机流由 `种群槽位 ^ (tick * 1_000_003) ^ ((deme_id + 7) * 6_559) ^ ((hook_index + 1) * 31)`
派生。**可复现性承诺**：针对同一 `setup(stochastic=True, seed=...)`、同一 hook 组合，
参考后端与 Rust 后端以及跨进程的确定性模拟产生 bit-reproducible 的结果；
任何自定义全局随机（`np.random.seed(...)` 这类）都不在承诺范围内。

## 执行路径

Hook 的物理执行路径由种群选择的后端决定（详见 [后端选择与性能](4_backend_selection.md)）：

- **参考（Python）后端**：声明式 Op 编译为 CSR 计划，由 Python 解释器逐事件执行；
  回调 Hook 直接调用。
- **Rust（原生扩展）后端**：CSR 计划与启动器在 Rust 会话内执行；单参数回调跨桥进入
  会话（每个调用获得独立的上下文封装）。
- 两条路径执行相同的事件顺序与相同确定性算术；`stochastic=False` 时轨迹逐位一致。

`backend="numba"` 已移除——选择该值会抛带迁移提示的 `ValueError`，请改用
`"rust"` 或 `"python"`。

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

运行在 Rust 后端时，声明式 Op 完全在会话内执行，是性能最优路径；回调 Hook 每次
触发跨一次 Python↔Rust 边界。

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

语义（三后端统一）：

- 写入经过 jsonc 边界校验，超出 `parameters.jsonc` 中声明的 `bounds` 抛 `ValueError`；
- 同 tick 后续阶段立即生效（参考路径直接写 draft；Rust 路径直接改会话生态列）；
- 每条实际变化追加到 `pop.params_log`，格式 `(tick, name, old, new)`；
- 向量/张量参数用 `pop.params.tensor_write(name, values)`。

自定义字段通过 `pop.state` / `config.custom['name'][()]` 读写，构建时用
`.custom(temperature=25.0)` 初始化，运行时可用 `pop.update().custom(...)` 修改。
自定义字段不在参数注册表中，`pop.params` 无法访问它们。

Hook 内如需构建链式更新，可用 `pop.update()` 返回的 Configurator（与构建链同语法）。

## 三后端一致性条款（slice ④ 语义决定）

- **late 事件中的 `stop()`**：在事件边界立即停止，当前 tick 的其余阶段不再执行，
  且 tick 不递增。
- **`stop()` 之后**：继续 `run()` 前必须先 `reset()`；否则 `run()` 抛错。
- **Hook 异常**：参考后端原始抛出的异常类型原样上抛；Rust 后端把跨桥异常包装为
  `RuntimeError`（信息保留在原 `__cause__`），因此跨后端异常类型**不对称**是
  有意行为。
- **Rust 的 hook 内参数写与 `run()` 合流**（HB-2 修复后）：会话内写发生在会话
  生态列中；`run()` 返回时审计日志按各自提交 tick 追加到 `params_log`，最终值
  同步回 draft，无需脏桥回推。

## 相关章节

- [Hook 系统](2_hooks.md) - 基础 Hook 概念和声明式 Hook 使用
- [运行时参数修改](3_runtime_modification.md)
- [Modifier 机制](3_modifiers.md) - 遗传修饰器机制
- [模拟内核深度解析](4_simulation_engine.md) - 模拟内核的工作原理
- [后端选择与性能](4_backend_selection.md) - 后端选择
