# Hook DSL 系统（现行架构）

本章基于当前实现，详细说明 Hook 的定义方式、编译结果、执行路径和边界条件。

## 一图看懂

```text
用户写 Hook (@hook)
  -> 编译为 CompiledHookDescriptor
  -> 声明式 Op.* 打包为 HookProgram (CSR 数组)
  -> 运行时按事件执行:
       a. kernel 路径: HookProgram + 合并后的 njit hook
       b. Python 路径: HookExecutor (CSR -> njit_fn -> py_wrapper)
```

关键词：

- HookProgram：当前唯一的声明式 Hook 结构化数据载体
- CompiledEventHooks：kernel 使用的单一 hooks bundle
- HookExecutor：Python 侧事件触发协调器

## 事件模型

支持 6 个事件名：

| 事件 | 说明 |
|------|------|
| first | 每个 tick 开始（繁殖前） |
| reproduction | 繁殖计算后 |
| early | 繁殖后、生存前 |
| survival | 生存后 |
| late | 衰老后，tick 末尾 |
| finish | 模拟结束时触发（非每 tick） |

## 三种 Hook 写法

### 1) 声明式（推荐）

```python
from natal.hook_dsl import hook, Op

@hook(event='first', priority=10)
def release():
    return [
        Op.add(genotypes='Drive|WT', ages=[2, 3, 4], delta=200, when='tick % 7 == 0'),
        Op.scale(genotypes='WT|WT', ages='*', factor=0.98),
    ]

release.register(pop)
```

声明式 hook 会被编译为 CSR 数组并写入 HookProgram。

### 2) 选择器模式

```python
from natal.hook_dsl import hook

@hook(event='late', selectors={'target_gt': 'Drive|WT'})
def monitor(pop, target_gt):
    # target_gt 是预解析后的索引（int）
    c = pop.state.individual_count[:, :, target_gt].sum()
    if c < 50:
        pop.state.individual_count[0, 2, target_gt] += 10
```

选择器模式有两条路径：

- Python 选择器路径（默认）：生成 `py_wrapper(pop)`，在 Python 事件系统执行。
- Numba 选择器路径（`numba=True` 或函数本身是 `@njit`）：生成 `njit_fn(ind_count, tick)`，可进 kernel 主循环。

### 3) 原生 Numba

```python
from numba import njit

@njit
def mortality_boost(ind_count, tick):
    if tick >= 30:
        ind_count[:, 0, :] *= 0.9
    return 0

pop.set_hook('early', mortality_boost)
```

这类 hook 直接作为 `njit_fn` 参与合并执行。

## 执行路径细化

### 路径 A：kernel 加速路径（run）

`AgeStructuredPopulation.run()` 与 `DiscreteGenerationPopulation.run()` 会走 `simulation_kernels.run_*_with_compiled_event_hooks(...)`。

在每个事件点，执行顺序是：

1. HookProgram 中对应事件的 CSR 操作
2. 该事件合并后的 `njit_fn` hook

Python `py_wrapper` 不在 kernel 内执行。

### 路径 B：Python 事件路径（trigger_event）

`BasePopulation.trigger_event(event)` 使用 HookExecutor，顺序是：

1. CSR
2. njit_fn
3. py_wrapper

典型用途是显式事件触发与兼容逻辑（例如 `finish`）。

## Op API（按当前实现）

### 数量变换

- `Op.scale(...)`
- `Op.set_count(...)`
- `Op.add(...)`
- `Op.subtract(...)`
- `Op.sample(...)`

### 其他声明式操作类型

- `Op.kill(...)`
- `Op.stop_if_zero(...)`
- `Op.stop_if_below(...)`
- `Op.stop_if_above(...)`
- `Op.stop_if_extinction(...)`

说明：

- `Op.kill(prob=p)` 会根据配置执行：
    - `is_stochastic=False`：确定性缩放 `count = count * (1 - p)`
    - `is_stochastic=True`：按伯努利/二项抽样得到存活数
- 在年龄结构模型中，若存在 `sperm_storage`，`Op.kill` 会先对每个 sperm 类型分别抽样，再对处女雌性抽样，最后合并为雌性存活数（与 survival 语义一致）。
- 当 `use_dirichlet_sampling=True` 时，`Op.kill` 的 stochastic 分支使用连续近似（Beta/Binomial 近似）。
- `Op.scale` / `Op.subtract` / `Op.sample` / `Op.set_count` 与上述逻辑共用同一抽样内核：
    - 先换算为目标 count。
    - 对有 `sperm_storage` 的雌性，缩减时按“先各 sperm 类型、再处女”抽样；增长时等价于添加处女个体。
    - 其中 `Op.set_count`：目标大于当前值等价 `add`，小于当前值等价 `subtract`，等于当前值无变化。
- `Op.stop_if_*` 在 kernel 中会返回停止信号（`RESULT_STOP`），从而提前结束 run 循环。

## 条件表达式语法（`when=`）

当前 DSL 条件解析仅支持以下形式：

- `tick == N`
- `tick % N == 0`
- `tick >= N`
- `tick > N`
- `tick <= N`
- `tick < N`

并支持逻辑组合：

- `not EXPR`
- `EXPR and EXPR`
- `EXPR or EXPR`
- 括号分组：`(EXPR)`

示例：

- `tick >= 10 and tick < 50`
- `tick % 7 == 0 and not (tick == 14)`
- `(tick == 5 or tick == 10) and tick < 20`

仍不支持任意 Python 表达式（例如函数调用或属性访问）。

## 注册与优先级

有两种常见注册方式：

```python
# 方式 1：装饰器 + register
@hook(event='first', priority=5)
def h1():
    return [Op.add(genotypes='*', ages=0, delta=1)]

h1.register(pop)

# 方式 2：set_hook（可覆盖事件名）
pop.set_hook('first', h1)
```

注意：

- `register(pop, event_override=...)` 只接受事件覆盖，不接受优先级参数。
- 优先级请在装饰器里设置：`@hook(priority=...)`。

## selector 模式的当前边界

`compile_selector_hook` 的 Numba 包装器目前将 selector 作为编译期常量注入。

- 单值 selector：按预期注入。
- 多值 selector：当前会退化为取第一个索引（Numba 包装器限制）。

如果你需要在 selector 中保留数组语义，建议使用 Python selector 路径。

## 与历史行为相关的要点

- 声明式 `Op.add` 等操作现在通过 HookProgram 显式进入 kernel 事件循环。
- 代码层已不再依赖旧的 HookRegistry 对象。
- 为避免双轨维护，只有真实 `py_wrapper` 的 compiled hook 才会镜像进传统 `_hooks`。

## 调试建议

### 先确认走的是哪条路径

- 追求速度和稳定执行顺序：优先 kernel 路径（声明式 / njit selector / numba hook）。
- 需要 Python 断点与复杂对象访问：用 Python selector 路径。

### 快速定位问题

1. 检查事件名是否正确（`first/reproduction/early/survival/late/finish`）。
2. 检查 `when` 是否属于受支持语法。
3. 检查 selector 是否是多值且被 Numba 路径截断。
4. 检查优先级是否按预期设置在 `@hook(priority=...)`。

## 最小可用示例

```python
from natal.hook_dsl import hook, Op

@hook(event='first', priority=0)
def periodic_release():
    return [
        Op.add(genotypes='Drive|WT', ages=[2, 3, 4], delta=100, when='tick % 5 == 0')
    ]

@hook(event='late', selectors={'target_gt': 'Drive|WT'})
def cap_target(pop, target_gt):
    arr = pop.state.individual_count
    if arr[:, :, target_gt].sum() > 5000:
        arr[:, :, target_gt] *= 0.95

periodic_release.register(pop)
cap_target.register(pop)

pop.run(200, record_every=10)
```

## 相关章节

- [快速开始](01_quickstart.md)
- [Simulation Kernels 深度解析](03_simulation_kernels.md)
- [Modifier 机制](06_modifiers.md)
- [Numba 优化指南](08_numba_optimization.md)
