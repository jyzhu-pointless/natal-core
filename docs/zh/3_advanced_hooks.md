# 高级 Hook 教程

[基础教程](2_hooks.md) 介绍了声明式 Hook（`Op.add`、`Op.scale` 等），适合大多数常规场景。
当需要直接操作 NumPy 数组进行更灵活的状态修改时（例如条件分支、循环、自定义计算），
可以使用自定义 Hook 或 Selector-based Hook。

## 自定义 Hook

自定义 Hook 允许你直接编写代码操作模拟状态，在 Numba 编译后执行。

### 基本用法

```python
from natal.hooks import hook


@hook(event="late", priority=10)
def custom_release_hook(ind_count, tick, deme_id=-1):
    # ind_count 是个体数量的 NumPy 数组
    # 形状为 (sex, age, genotype)
    # sex=0 对应雌性，sex=1 对应雄性

    # 每 10 个 tick 释放 100 个个体
    if tick % 10 == 0:
        # 假设 Drive|WT 的基因型索引是 1
        ind_count[:, :, 1] += 100

    return 0  # 0 表示继续模拟
```

### 函数签名

自定义 Hook 支持两种函数签名：

- `(ind_count, tick)` — 不接收 deme 信息（简化签名）
- `(ind_count, tick, deme_id=-1)` — 接收 deme 索引，空间种群中跨 deme 运行时传入实际索引

`deme_id` 的默认值为 `-1`。当在 `SpatialPopulation` 中注册时，系统会自动传入当前 deme 的索引；
在非空间种群中注册时，可以不传该参数。

### 数组索引注意事项

`ind_count` 的维度顺序为 `(sex, age, genotype)`：

- `sex=0` 对应雌性（FEMALE），`sex=1` 对应雄性（MALE）
- 在 Numba 模式下，不能使用 `Sex.MALE` 等枚举类型直接索引，必须使用整数值或 `.value`

```python
# 正确做法
male_count = ind_count[1, :, :].sum()
female_count = ind_count[0, :, :].sum()

# 或者使用 .value
male_count = ind_count[Sex.MALE.value, :, :].sum()
```

### 完整示例

```python
from natal.hooks import hook


@hook(event="early", priority=5)
def custom_culling_hook(ind_count, tick, deme_id=-1):
    # 对特定基因型进行选择性剔除
    if tick > 50:
        # WT|WT 的基因型索引是 0
        wt_wt_count = ind_count[:, :, 0].sum()
        if wt_wt_count > 10000:
            ind_count[:, :, 0] = ind_count[:, :, 0] * 0.9

    return 0
```

## Selector-based Hook

Selector-based Hook 是自定义 Hook 的一种形式，允许你先选定某个目标（例如特定基因型），然后基于这些选择执行自定义逻辑。

### 基本用法

```python
from natal.hooks import hook


@hook(event="late", selectors={"target_gt": "Drive|WT"}, priority=10)
def cap_target(ind_count, tick, target_gt):
    # target_gt 是通过选择器解析得到的基因型索引（整数）
    if tick % 10 == 0:
        ind_count[:, :, target_gt] *= 0.95
```

### 特点

- 选择器在注册时解析，得到的索引值会烘焙到生成的代码中
- 无需手动管理基因型到索引的映射
- 适合需要基于特定目标执行逻辑的场景

### 完整示例

```python
from natal.hooks import hook


@hook(event="early", selectors={"drive_gt": "Drive|WT", "wt_gt": "WT|WT"}, priority=5)
def balance_population(ind_count, tick, drive_gt, wt_gt):
    # 平衡驱动基因型和野生型的比例
    drive_count = ind_count[:, :, drive_gt].sum()
    wt_count = ind_count[:, :, wt_gt].sum()

    if drive_count > wt_count * 2:
        ind_count[:, :, drive_gt] *= 0.8
    elif wt_count > drive_count * 2:
        ind_count[:, :, wt_gt] *= 0.8

    return 0
```

## Numba 兼容的随机采样

在自定义 Hook 中进行随机采样时，推荐使用 `natal.numba_compat` 模块提供的函数。这些函数经过优化，可以在 Numba 模式和纯 Python 模式下都保持高效：

```python
from natal.numba_compat import (
    binomial,
    binomial_2d,
    continuous_binomial,
    continuous_multinomial,
    set_numba_seed,
)
from natal.hooks import hook


@hook(event="late", priority=10)
def stochastic_culling_hook(ind_count, tick, deme_id=-1):
    if tick > 50:
        # 使用二项分布随机剔除
        # 假设对基因型 0 进行 10% 的剔除概率
        n_current = ind_count[:, :, 0]
        survival_prob = 0.9

        # continuous_binomial 对大数量更高效
        ind_count[:, :, 0] = continuous_binomial(n_current, survival_prob)

    return 0
```

### 主要 API

| 函数 | 说明 |
|------|------|
| `binomial(n, p)` | 二项分布采样，返回 n 次试验中成功的次数 |
| `binomial_2d(n, p, n_rows, n_cols)` | 对 2D 数组进行逐元素二项分布采样 |
| `continuous_binomial(n, p)` | 连续化二项分布，返回浮点数（对大数量更高效） |
| `continuous_multinomial(n, p_array, out_counts)` | 连续化多项分布 |
| `multinomial(n, pvals)` | 多项分布采样 |
| `set_numba_seed(seed)` | 设置随机数种子（确保可复现性） |
| `clamp01(x)` | 将值限制在 [0, 1] 范围内 |

### 使用场景

- **确定性操作后添加随机性**：先确定性缩放，再用随机采样添加噪声
- **条件性随机剔除**：根据当前状态动态决定剔除概率
- **批量采样操作**：使用 `binomial_2d` 对整个数组进行批量采样

```python
@hook(event="late", priority=10)
def age_specific_mortality(ind_count, tick, deme_id=-1):
    if tick % 10 == 0:
        # 对每个年龄组应用不同的存活概率
        survival_rates = np.array([0.8, 0.9, 0.95, 0.98, 0.99])

        # 使用 binomial_2d 进行批量采样
        n_ages = ind_count.shape[1]
        for age in range(n_ages):
            n_survivors = binomial_2d(
                ind_count[:, age, :],
                np.array([survival_rates[age]]),
                2,  # sex
                ind_count.shape[2]  # n_genotypes
            )
            ind_count[:, age, :] = n_survivors

    return 0
```

## 执行模式与兼容性

NATAL Core 的 Hook 系统根据全局 `NUMBA_ENABLED` 开关自动选择执行路径：

| `NUMBA_ENABLED` | 自定义 Hook 行为 |
|----------------|-----------------|
| `True`（默认） | Hook 代码必须遵循 Numba 语法，系统自动进行 Numba 编译 |
| `False` | Hook 可以使用纯 Python 语法，通过 `HookExecutor` 统一调度执行 |

### 为什么强调 Numba 语法？

框架默认开启 Numba 优化，这意味着：

1. 自定义 Hook 默认会通过 `njit_switch` 装饰器进行 Numba 编译
2. 如果代码包含不支持的 Python 特性，会在注册或首次执行时出错
3. 性能优势在大规模模拟中尤为明显

### 如果需要在 Numba 关闭时使用 Hook？

当 `NUMBA_ENABLED=False` 时：

- 所有 Hook（declarative、selector、custom）都会通过 `HookExecutor` 统一调度
- 系统会根据 `priority` 按顺序执行所有 Hook
- 无需修改 Hook 定义代码，即可切换执行路径

```python
from natal.numba_utils import numba_disabled

with numba_disabled():
    # 在这个上下文中，NUMBA_ENABLED 为 False
    pop = builder.hooks(my_custom_hook).build()
    pop.run(n_steps=100)
```

## 混合使用不同类型的 Hook

NATAL Core 允许在同一事件中混合使用不同类型的 Hook：

```python
from natal.hooks import hook, Op


# 声明式 Hook：定期释放个体
@hook(event="first", priority=10)
def release_hook():
    return [Op.add(genotypes="Drive|WT", ages=[2, 3, 4], delta=100, when="tick % 10 == 0")]


# Selector-based Hook：基于选择器的操作
@hook(event="first", priority=7, selectors={"drive_gt": "Drive|WT"})
def check_drive_threshold(ind_count, tick, drive_gt):
    drive_count = ind_count[:, :, drive_gt].sum()
    if drive_count > 10000:
        # 可以在这里添加日志或状态记录
        pass
    return 0


# 自定义 Hook：高效计算和修改状态（deme_id=-1 表示 non-spatial 默认值）
@hook(event="first", priority=5)
def custom_process_hook(ind_count, tick, deme_id=-1):
    # 执行密集计算
    for age in range(ind_count.shape[1]):
        ind_count[:, age, :] *= 0.99  # 轻微死亡率
    return 0


pop = builder.hooks(release_hook, check_drive_threshold, custom_process_hook).build()
```

### 执行顺序

当混合使用不同类型的 Hook 时：

- 系统会根据 `priority` 值排序（值越小，优先级越高）
- 同一优先级的 Hook 执行顺序不确定
- `NUMBA_ENABLED=True` 时，selector 和 custom Hook 会被合并为单个 Numba 函数执行

## 性能比较

| Hook 类型 | 性能 | 灵活性 | 可读性 | 适用场景 |
|----------|------|--------|--------|----------|
| 声明式 Hook | 高 | 中 | 高 | 大多数常规场景 |
| Selector-based Hook | 高 | 高 | 中 | 需要基于特定目标执行逻辑的场景 |
| 自定义 Hook | 最高 | 高 | 中 | 计算密集型操作 |

## 相关章节

- [Hook 系统](2_hooks.md) - 基础 Hook 概念和声明式 Hook 使用
- [Modifier 机制](3_modifiers.md) - 遗传修饰器机制
- [模拟内核深度解析](4_simulation_kernels.md) - 模拟内核的工作原理
- [Numba 优化指南](4_numba_optimization.md) - Numba 优化技巧
