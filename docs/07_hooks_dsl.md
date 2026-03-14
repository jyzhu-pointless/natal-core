# Hook DSL 系统

本章讲解 NATAL 的高阶钩子系统（Hook DSL），允许在模拟的特定阶段注入自定义逻辑。

## 核心概念

Hook 是在模拟的 **特定事件** 触发时执行的回调函数。NATAL 支持四个关键事件：

| 事件 | 触发时机 | 用途 |
|------|---------|------|
| `first` | 每个 tick 开始（在繁殖前） | 初始化、释放个体 |
| `reproduction` | 繁殖完成后 | 修改新生儿、检查性别比 |
| `early` | 生存前的早期阶段 | 未定义的中间操作 |
| `survival` | 生存完成后 | 监控死亡、应用额外筛选 |
| `late` | tick 末尾（衰老后） | 监控、记录数据、停止条件 |

## Hook 的三种写法

### 模式 1️⃣：声明式 Hook（推荐）

使用 `@hook` 装饰器和 `Op.*` 操作，最简洁明了。

#### 基本语法

```python
from natal.hook_dsl import hook, Op

@hook(event='late')
def monitor_population():
    """在每个 tick 末尾检查种群"""
    return [
        # 返回操作列表
        Op.scale(genotypes='AA', ages=[0, 1], factor=0.9),
        Op.add(genotypes='*', ages=0, delta=100, when='tick % 10 == 0'),
    ]

# 注册到种群
monitor_population.register(pop)
```

#### 支持的操作 Op.*

```python
# 修改个体数量
Op.add(genotypes='...', ages='...', delta=X, when='...')       # 增加
Op.subtract(genotypes='...', ages='...', delta=X, when='...')  # 减少
Op.scale(genotypes='...', ages='...', factor=X, when='...')    # 乘以因子
Op.set(genotypes='...', ages='...', value=X, when='...')       # 设置为

# 采样操作
Op.sample(genotypes='...', ages='...', size=X, when='...')     # 采样保留

# 停止条件（返回 STOP 信号）
Op.stop_if_zero(genotypes='...', ages='...')                   # 如果为0则停止
Op.stop_if_below(genotypes='...', ages='...', threshold=X)     # 如果低于阈值
Op.stop_if_above(genotypes='...', ages='...', threshold=X)     # 如果高于阈值
Op.stop_if_extinction()                                         # 如果种群灭绝
```

#### 操作参数详解

##### 基因型选择器

```python
Op.add(genotypes='A1|A1', ...)         # 精确匹配单个基因型
Op.add(genotypes=['A1|A1', 'A1|A2'], ...) # 列表选择
Op.add(genotypes='A1|*', ...)          # 通配符（所有 A1 为第一个等位基因）
Op.add(genotypes='*', ...)             # 所有基因型
```

##### 年龄选择器

```python
Op.add(ages=2, ...)                    # 单个年龄
Op.add(ages=[2, 3, 4], ...)            # 列表
Op.add(ages=range(2, 6), ...)          # 范围
Op.add(ages='*', ...)                  # 所有年龄
```

##### 性别选择器

```python
Op.add(genotypes='A1|A1', sex='female', ...)     # 仅雌性
Op.add(genotypes='A1|A1', sex='male', ...)       # 仅雄性
Op.add(genotypes='A1|A1', sex='both', ...)       # 两性（默认）
```

##### 条件选择器

```python
# 条件何时生效
Op.add(..., when='tick == 10')             # 恰好第 10 个 tick
Op.add(..., when='tick % 10 == 0')         # 每 10 个 tick
Op.add(..., when='tick >= 50 and tick < 100')  # 范围
Op.add(..., when='population.get_total_count() < 100')  # 种群大小
```

#### 完整示例

```python
from natal.hook_dsl import hook, Op

# 例 1：监控和报告
@hook(event='late')
def monitor():
    return [
        # 每 10 个 tick 检查一次种群大小
        Op.add(genotypes='*', ages='*', delta=0, when='tick % 10 == 0'),
        # （实际上这里 delta=0 不做任何事，但可以用于触发 hook）
    ]

monitor.register(pop)

# 例 2：定期释放不育雄性（SIT）
@hook(event='first')
def release_sterile_males():
    return [
        Op.add(
            genotypes='S|S',  # 不育标记
            ages=[2, 3, 4],
            delta=500,  # 每个 tick 增加 500 只
            when='tick >= 10 and tick < 50'  # 第 10-49 tick
        )
    ]

release_sterile_males.register(pop)

# 例 3：灭绝条件
@hook(event='late')
def check_extinction():
    return [
        Op.stop_if_extinction(),  # 种群灭绝时停止
    ]

check_extinction.register(pop)
```

### 模式 2️⃣：选择器模式（中级）

在声明式的基础上，预先计算索引。

#### 基本语法

```python
from natal.hook_dsl import hook

@hook(event='late', selectors={'target_gt': 'A1|A2'})
def modify_target(pop, target_gt):
    """
    Hook 函数接收 pop 和预解析的选择器参数
    
    selectors 会在初始化时被解析为整数索引
    """
    # target_gt 是整数索引
    count = pop.state.individual_count[1, 3, target_gt]
    
    if count < 10:
        # 如果该基因型太少，增加一些
        pop.state.individual_count[1, 3, target_gt] = 50
    
    return 0  # 0 = 继续，1 = 停止

modify_target.register(pop)
```

#### 选择器类型

```python
@hook(event='first', selectors={
    'my_gt': 'A1|A2',           # 基因型字符串
    'age_range': [2, 3, 4],     # 年龄列表
    'sex': 'female',            # 性别字符串
    'label': 'Cas9_deposited',  # 配子标签
})
def complex_hook(pop, my_gt, age_range, sex, label):
    # 所有选择器都被预解析
    # my_gt 是整数索引
    # age_range 是年龄列表
    # sex 是性别整数（0=male, 1=female）
    # label 是标签索引
    pass
```

### 模式 3️⃣：原生 Numba Hook（高级）

完全由用户控制，最大灵活性但需要 Numba 知识。

#### 基本语法

```python
from numba import njit

@njit
def my_numba_hook(ind_count, tick):
    """
    Args:
        ind_count: (n_sexes, n_ages, n_genotypes) 个体计数数组
        tick: 当前时间步
        
    Returns:
        int: 0 (继续) 或 1 (停止)
    """
    if tick == 50:
        # 第 50 个 tick 时，将年龄 2-4 的第一个基因型个体清零
        ind_count[1, 2:5, 0] = 0
    
    return 0  # 继续

# 注册（但这要求你已经知道索引值）
pop.set_hook("late", my_numba_hook)
```

#### 例：基于种群大小的动态调整

```python
from numba import njit

@njit
def dynamic_release(ind_count, tick):
    """根据种群大小动态释放个体"""
    
    total = ind_count.sum()
    
    if total > 5000 and tick % 5 == 0:
        # 种群大于 5000 时，每 5 个 tick 增加一些
        ind_count[1, 3, 0] += 100  # 增加雌性年龄3基因型0
    
    # 超过 10000 时停止
    if total > 10000:
        return 1  # 停止
    
    return 0  # 继续

pop.set_hook("first", dynamic_release)
```

## 比较三种模式

| 特性 | 声明式 | 选择器 | Numba |
|------|--------|--------|-------|
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 灵活性 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 性能 | 最佳 | 好 | 最佳 |
| 易调试 | ✅ | ✅ | ❌ |
| Numba 兼容 | ✅ | 部分 | ✅ |

**建议**：
- 简单操作 → 声明式
- 中等复杂度 → 选择器
- 复杂逻辑 + 最高性能 → Numba

## Hook 的执行时机

### tick 内的执行顺序

```
Tick t:
  ├─ first hook       (在繁殖前)
  ├─ [繁殖阶段]
  ├─ reproduction hook (繁殖后)
  ├─ early hook       (繁殖和生存之间)
  ├─ [生存阶段]
  ├─ survival hook    (生存后)
  ├─ [衰老阶段]
  ├─ late hook        (衰老后，tick 末尾)
  └─ Tick t+1 开始
```

### 使用场景

```python
# 释放个体 → 应该用 first hook
@hook(event='first')
def release():
    return [Op.add(genotypes='Drive|*', ages=[2,3,4], delta=100)]

# 监控死亡 → 应该用 survival hook
@hook(event='survival')
def monitor_deaths():
    return [...]

# 统计和停止条件 → 应该用 late hook
@hook(event='late')
def finalize():
    return [Op.stop_if_extinction()]
```

## Hook 与 Modifier 的交互

Hook 和 Modifier 可以一起使用，创建复杂的模拟逻辑：

```python
from natal.hook_dsl import hook, Op

# Modifier：基因驱动（改变遗传学）
def gene_drive_modifier(pop):
    return {
        "Drive|WT": {
            ("Drive", "Cas9"): 0.95,
            ("WT", "Cas9"): 0.05,
        }
    }

# Hook 1：释放驱动型（改变种群动态）
@hook(event='first')
def release_drive():
    return [
        Op.add(genotypes='Drive|WT', ages=[2, 3, 4, 5], 
               delta=100, when='tick == 50')
    ]

# Hook 2：监控（收集数据）
@hook(event='late')
def monitor():
    return [
        Op.stop_if_extinction(),
    ]

# 注册
pop.set_gamete_modifier(gene_drive_modifier, hook_name="drive")
release_drive.register(pop)
monitor.register(pop)

# 现在：Hook 1 注入个体 → Modifier 改变遗传学 → Hook 2 监控结果
pop.run(200, record_every=10)
```

## 高级用法

### Hook 的动态条件

```python
@hook(event='late')
def adaptive_hook():
    """
    条件可以访问 population 对象的属性
    """
    return [
        # 如果种群大小低于初始的 50%，停止
        Op.stop_if_below(
            genotypes='*', 
            ages='*',
            threshold=1000,  # 假设初始是 2000
            when='population.get_total_count() < 1000'
        ),
    ]
```

### 多个 Hook 的优先级

```python
# Hook 按注册顺序执行
@hook(event='late')
def hook_1():
    return [Op.add(genotypes='A', ages=0, delta=50)]

@hook(event='late')
def hook_2():
    return [Op.stop_if_above(genotypes='*', ages='*', threshold=10000)]

# 注册顺序：hook_1 先执行，然后 hook_2
hook_1.register(pop)
hook_2.register(pop)

# 可以通过优先级控制顺序
hook_1.register(pop, priority=2)  # 后执行
hook_2.register(pop, priority=1)  # 先执行
```

### Hook 中访问详细信息

在选择器模式中可以访问种群的完整状态：

```python
@hook(event='late', selectors={'target_gt': 'A1|A2'})
def detailed_hook(pop, target_gt):
    """在 Hook 中访问详细的种群信息"""
    
    # 访问状态
    total = pop.get_total_count()
    females = pop.get_female_count()
    
    # 访问时间步
    tick = pop._tick
    
    # 访问特定基因型的计数
    from natal.type_def import Sex
    target_count = pop.state.individual_count[Sex.FEMALE, :, target_gt].sum()
    
    # 根据条件决定操作
    if target_count < 10 and tick > 100:
        # 需要增加这个基因型
        pop.state.individual_count[Sex.FEMALE, 3, target_gt] += 50
    
    return 0  # 继续

detailed_hook.register(pop)
```

## 常见模式库

### 模式 1：定期释放

```python
@hook(event='first')
def periodic_release():
    return [
        Op.add(genotypes='Release|*', ages=[2, 3, 4], 
               delta=500, when='tick % 7 == 0 and tick >= 14')
        # 从 tick 14 开始，每 7 个 tick 释放 500 只
    ]

periodic_release.register(pop)
```

### 模式 2：灭绝检测

```python
@hook(event='late')
def extinction_check():
    return [
        Op.stop_if_zero(genotypes='WT|WT', ages='*'),
        # 或
        Op.stop_if_extinction(),  # 整个种群灭绝时停止
    ]

extinction_check.register(pop)
```

### 模式 3：人工选择

```python
@hook(event='survival')
def artificial_selection():
    return [
        Op.scale(genotypes='Desired|*', ages='*', factor=1.1),  # 增加偏好基因型
        Op.scale(genotypes='Undesired|*', ages='*', factor=0.9),  # 减少非偏好基因型
    ]

artificial_selection.register(pop)
```

### 模式 4：年龄结构维护

```python
@hook(event='early')
def maintain_age_structure():
    return [
        # 如果幼体过多，减少新生儿
        Op.scale(genotypes='*', ages=0, factor=0.8, 
                 when='population.get_total_count() > 10000'),
    ]

maintain_age_structure.register(pop)
```

## 性能考量

### Numba Hook vs 声明式 Hook

```python
# Numba Hook：完全编译，最快
@njit
def fast_hook(ind_count, tick):
    if tick % 100 == 0:
        ind_count[1, 2, 0] += 100
    return 0

# 声明式 Hook：有 Python 开销，但仍很快
@hook(event='first')
def slow_hook():
    return [
        Op.add(genotypes='A1|A1', ages=2, delta=100, when='tick % 100 == 0')
    ]

# 性能差异：通常不明显（Hook 不是性能瓶颈）
```

### Hook 放在哪个事件最高效

```python
# ✅ 高效：操作最少数据
@hook(event='first')
def quick_release():
    return [Op.add(genotypes='A1|A2', ages=[2, 3], delta=100)]

# ❌ 低效：操作所有数据
@hook(event='first')
def slow_scale():
    return [Op.scale(genotypes='*', ages='*', factor=0.99)]
```

## 调试 Hook

### 添加日志

```python
@hook(event='late')
def debug_hook():
    """调试 Hook 可以打印信息"""
    # 注意：在 Numba 编译时会丢失日志
    return [
        Op.add(genotypes='*', ages='*', delta=0),  # 无操作
        # 但可以触发后续的 Python 层调试
    ]
```

### Hook 中的断点

```python
@hook(event='late', selectors={'target_gt': 'A1|A2'})
def debug_selector_hook(pop, target_gt):
    """选择器 Hook 中可以使用 Python 调试工具"""
    import pdb
    
    count = pop.state.individual_count[:, :, target_gt].sum()
    if count < 10:
        pdb.set_trace()  # 设置断点
    
    return 0

debug_selector_hook.register(pop)
```

---

## 🎯 本章总结

| 模式 | 最佳场景 | 代码复杂度 | 性能 |
|------|---------|---------|------|
| **声明式** | 简单的操作（增加、删除、停止） | 低 | 最佳 |
| **选择器** | 需要条件判断的中等复杂度 | 中 | 好 |
| **Numba** | 复杂的自定义逻辑 | 高 | 最佳 |

**设计要点**：
1. Hook 在特定时机执行，改变种群动态
2. 三种写法各有优缺点，根据需求选择
3. Hook 和 Modifier 可以组合使用
4. 性能通常不是 Hook 的瓶颈

---

## 📚 相关章节

- [快速开始](01_quickstart.md) - Hook 的基本使用
- [Modifier 机制](06_modifiers.md) - 与 Modifier 的协同
- [Simulation Kernels 深度解析](03_simulation_kernels.md) - Hook 在计算中的角色
- [Numba 优化指南](08_numba_optimization.md) - Numba Hook 的优化

---

**准备优化性能了吗？** [前往下一章：Numba 优化指南 →](08_numba_optimization.md)
