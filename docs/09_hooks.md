# Hook 系统

Hook 用于在模拟流程的固定时间点插入用户逻辑。

如果你希望在每个 tick 的某个阶段执行“额外操作”，例如周期投放、条件干预、阈值终止，Hook 是最直接的方式。

## 1. Hook 的使用价值

Hook 让你在不改动核心内核代码的前提下扩展模型行为，适合：

1. 周期性干预（如每隔若干步释放个体）。
2. 条件触发控制（如数量低于阈值时补充）。
3. 研究流程控制（如达到条件后提前结束）。

## 2. 事件时间点

NATAL 提供四个标准事件：

- `first`：每个 tick 的早期阶段。
- `early`：繁殖后、生存前。
- `late`：生存后、衰老前。
- `finish`：模拟结束阶段。

选择事件时，建议先明确你的干预发生在“生存前”还是“生存后”，这会显著影响结果解释。

## 3. 推荐写法：声明式 Hook

对于大多数用户，推荐使用 `@hook` 与 `Op.*`：

```python
from natal.hook_dsl import hook, Op


@hook(event="first", priority=10)
def periodic_release():
    return [
        Op.add(genotypes="Drive|WT", ages=[2, 3, 4], delta=200, when="tick % 7 == 0"),
        Op.scale(genotypes="WT|WT", ages="*", factor=0.98),
    ]


periodic_release.register(pop)
```

这种方式可读性高、维护成本低，也更便于团队复核模型规则。

## 4. Op 操作的直观理解

常用操作包括：

- `Op.add`：增加个体数量。
- `Op.subtract`：减少个体数量。
- `Op.scale`：按比例缩放。
- `Op.set_count`：设置目标数量。
- `Op.kill`：按死亡概率处理。
- `Op.stop_if_*`：满足条件时停止运行。

可以把它们理解为“对状态张量进行声明式变换”。

## 5. 条件表达式（when）

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

## 6. Selector 写法（按目标选择）

当你希望先选定某个目标（例如特定基因型）再执行逻辑时，可以使用 selector 模式：

```python
from natal.hook_dsl import hook


@hook(event="late", selectors={"target_gt": "Drive|WT"})
def cap_target(pop, target_gt):
    arr = pop.state.individual_count
    if arr[:, :, target_gt].sum() > 5000:
        arr[:, :, target_gt] *= 0.95
```

这类写法适合需要读取当前状态并做条件判断的场景。

## 7. 注册方式

### 7.1 装饰器对象注册

```python
my_hook.register(pop)
```

### 7.2 直接设置

```python
pop.set_hook("first", my_hook)
```

如果存在多个 Hook，建议通过 `priority` 明确执行顺序，避免隐式顺序导致结果难以复现。

## 8. 与 run / run_tick 的关系

Hook 会在 `run(...)` 与 `run_tick()` 中按事件顺序自动执行。

因此，用户通常不需要手动触发 Hook；只需：

1. 定义 Hook。
2. 注册 Hook。
3. 正常运行模拟。

## 9. 实践建议

1. 保持每个 Hook 职责单一，便于验证。
2. 先做短程模拟确认行为，再扩展到长程。
3. 当多个 Hook 共同作用时，记录优先级和预期顺序。
4. 重要实验建议保存配置和 Hook 定义，以支持复现实验。

## 10. 最小组合示例

```python
from natal.hook_dsl import hook, Op


@hook(event="first", priority=0)
def release():
    return [Op.add(genotypes="Drive|WT", ages=[2, 3, 4], delta=100, when="tick % 5 == 0")]


@hook(event="late", priority=5)
def cap(pop):
    # 这里可加入更细的状态检查
    return [Op.stop_if_above(genotypes="Drive|WT", ages="*", threshold=10000)]


release.register(pop)
cap.register(pop)

pop.run(n_steps=200, record_every=10)
```

## 11. 本章小结

Hook 系统为 NATAL 提供了稳定、可扩展的行为注入能力。

在用户层，推荐遵循三步：

1. 用 `@hook` 定义规则。
2. 用 `register` 或 `set_hook` 绑定到 population。
3. 用 `run(...)` 执行并结合 history 验证结果。

---

## 相关章节

- [Modifier 机制](08_modifiers.md)
- [Simulation Kernels 深度解析](06_simulation_kernels.md)
- [Numba 优化指南](07_numba_optimization.md)
- [快速开始](01_quickstart.md)
