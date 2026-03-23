# Numba 优化指南

本章讲解 NATAL 的 Numba JIT 编译机制，以及如何理解和优化性能。

## 核心概念

Numba 是一个 Python JIT 编译器，可以将 Python 代码编译为机器码，获得 C/C++ 级别的性能。

### 编译过程

```
Python 代码 (simulation_kernels.py)
    ↓ [@njit 装饰器]
首次调用时：编译 (1-5 秒)
    ↓
编译缓存 (.numba_cache/)
    ↓
后续调用：直接执行 (快 100-1000 倍)
```

### 速度提升示例

```
未优化的 Python：      1000 ms/tick
Numba JIT 编译后：     5-10 ms/tick
性能提升：            100-200 倍！
```

## @njit 装饰器与 @jitclass

### @njit（函数级编译）

```python
from numba import njit

@njit
def run_reproduction(ind_count, sperm_store, config):
    """这个函数会被编译为机器码"""
    # 纯 NumPy 操作
    for i in range(len(ind_count)):
        ind_count[i] *= 0.5
    return ind_count
```

**限制**：
- 只支持数值计算（NumPy 数组、标量）
- 不支持 Python 对象（list、dict、自定义类）
- 不支持字符串操作
- 不支持 print()（调试时需要禁用 Numba）

### @jitclass（类级编译）

```python
from numba.experimental import jitclass
from numba import types as nb_types

_spec = [
    ('n_tick', nb_types.int32),
    ('individual_count', nb_types.float64[:, :, :]),
]

@jitclass(_spec)
class PopulationState:
    """这个类及其方法都会被编译"""
    
    def __init__(self, n_genotypes, n_sexes, n_ages):
        self.n_tick = 0
        self.individual_count = np.zeros((n_sexes, n_ages, n_genotypes))
```

NATAL 的 `PopulationState` 和 `PopulationConfig` 都使用 @jitclass。

## 开关机制

Numba 默认**启用**以获得最佳性能。可以动态禁用用于调试。

### 全局开关

在 `src/natal/numba_utils.py` 中（默认开启）：

```python
# src/natal/numba_utils.py

# Numba 默认启用（最佳性能）
NUMBA_ENABLED = True  # 设为 False 用于调试
```

### 运行时控制（推荐）

**临时禁用 Numba 进行调试**（最简便）：

```python
from natal.numba_utils import numba_disabled

# 默认状态：Numba 启用
with numba_disabled():
    pop.run(n_steps=10)  # 这里用纯 Python 运行
    # 更好的错误信息用于调试

# 离开 with 块后，Numba 自动恢复
```

**全局启用/禁用**：

```python
from natal.numba_utils import enable_numba, disable_numba

# 禁用 Numba（用于调试）
disable_numba()
pop.run(n_steps=10)

# 重新启用（恢复到默认）
enable_numba()
pop.run(n_steps=100)  # 快速运行
```

**函数装饰器**：

```python
from natal.numba_utils import with_numba_disabled, with_numba_enabled

@with_numba_disabled
def debug_version():
    pop.run(n_steps=10)  # 纯 Python

@with_numba_enabled
def fast_version():
    pop.run(n_steps=1000)  # 保证 JIT 启用

debug_version()  # 禁用 Numba
fast_version()   # 启用 Numba
```

## 性能测试

### 启用 Numba 的性能（默认状态）

```python
from natal.genetic_structures import Species
from natal.nonWF_population import AgeStructuredPopulation
import time

# 创建种群
sp = Species.from_dict(...)
pop = AgeStructuredPopulation(...)

# 测试 1：Numba 启用（默认，快速运行）
from natal.numba_utils import numba_disabled

start = time.perf_counter()
pop.run(n_steps=100, record_every=1)  # 默认：Numba 启用
elapsed_numba = time.perf_counter() - start

print(f"With Numba (default): {elapsed_numba:.2f} s")

# 测试 2：禁用 Numba 以对比性能
with numba_disabled():
    pop.reset()
    start = time.perf_counter()
    pop.run(n_steps=100, record_every=1)
    elapsed_python = time.perf_counter() - start
    print(f"Without Numba: {elapsed_python:.2f} s")
    print(f"Speedup: {elapsed_python / elapsed_numba:.1f}x")
```

**典型结果**：
- n_genotypes = 10：10-20 倍加速
- n_genotypes = 50：50-100 倍加速
- n_genotypes = 100：100-200 倍加速

### 编译耗时

```
初始化（创建 PopulationConfig）：   50-500 ms
首个 tick（编译）：                  1-5 秒（包括初始化）
后续 tick：                         5-50 ms

总结：初始化和首个 tick 较慢，但只发生一次
```

## 缓存策略

### 编译缓存位置

```
.numba_cache/
├── __pycache__/
│   └── natal/
│       ├── simulation_kernels.*.nbc  # 编译后的 run_reproduction
│       ├── simulation_kernels.*.nbc  # 编译后的 run_survival
│       └── ...
└── ...
```

### 清除缓存

如果修改了代码或遇到奇怪的编译问题：

```bash
# 删除所有编译缓存
rm -rf .numba_cache/

# 或
import shutil
shutil.rmtree('.numba_cache/', ignore_errors=True)
```

### 预热编译

为了避免首次运行的延迟，可以在初始化后预编译：

```python
pop = AgeStructuredPopulation(...)

# 预热：运行一个 tick 以触发编译
print("Warming up Numba cache...")
pop.run(n_steps=1)
print("Ready!")

# 之后的运行会很快
pop.reset()
pop.run(n_steps=100)
```

## 调试技巧

### 禁用 Numba 进行调试

当遇到 Numba 编译错误或需要调试时，有两种方式临时禁用 Numba：

**方式 1：上下文管理器（推荐，仅在块内禁用）**

```python
from natal.numba_utils import numba_disabled

# 默认：Numba 启用（快速）
with numba_disabled():
    # 这里用纯 Python 运行，便于调试和查看错误信息
    pop = AgeStructuredPopulation(...)
    pop.run(n_steps=10)
    # 获得更详细的 Python 错误堆栈跟踪

# 离开 with 块后，Numba 自动恢复启用
pop.run(n_steps=100)  # 这里又是快速 JIT 运行
```

**方式 2：全局禁用（持久）**

```python
from natal.numba_utils import disable_numba, enable_numba

# 禁用 Numba（所有后续调用都用纯 Python）
disable_numba()
pop.run(n_steps=10)

# 调试完后重新启用
enable_numba()
pop.run(n_steps=100)
```

### 常见 Numba 错误

#### 错误 1：使用了不支持的 Python 特性

```python
# ❌ 错误：使用 print()
@njit
def bad_function(x):
    print(x)  # Numba 不支持 print
    return x

# ✅ 正确：删除 print，在 Python 层处理
@njit
def good_function(x):
    return x

# 如果需要调试输出，禁用 Numba
```

#### 错误 2：类型不匹配

```python
# ❌ 错误：混合类型
@njit
def bad_function(x):
    if x > 5:
        return "big"  # 字符串不支持
    else:
        return 0  # int

# ✅ 正确：保持类型一致
@njit
def good_function(x):
    if x > 5:
        return 1  # 都是 int
    else:
        return 0
```

#### 错误 3：对象方法不支持

```python
# ❌ 错误：调用 Python 对象方法
@njit
def bad_function(pop):
    return pop.get_total_count()  # pop 是 Python 对象，不支持

# ✅ 正确：只操作 NumPy 数组
@njit
def good_function(ind_count):
    return ind_count.sum()  # NumPy 方法支持
```

## 性能优化建议

### 1️⃣ 减少基因型数量

基因型数量是最主要的性能因素：

```
n_genotypes = 10    → 10 ms/tick
n_genotypes = 30    → 50 ms/tick  (配子矩阵 O(n^3))
n_genotypes = 100   → 200+ ms/tick
```

**优化**：
- 合并不相关的位点
- 使用单位点模型而非多位点
- 限制等位基因数量

### 2️⃣ 减少 hook 复杂度

```python
# ❌ 低效：Hook 中进行复杂计算
@hook(event='late')
def slow_hook():
    return [
        Op.scale(genotypes='*', ages='*', factor=0.99),  # 影响所有个体
    ]

# ✅ 高效：Hook 中只操作特定基因型
@hook(event='late')
def fast_hook():
    return [
        Op.scale(genotypes='A1|A1', ages='*', factor=0.99),  # 只影响一个基因型
    ]
```

### 3️⃣ 批量 Monte Carlo 模拟

对于大规模随机模拟，使用 `batch_ticks()` 而非多个 `pop.run()`：

```python
from natal.simulation_kernels import export_state, batch_ticks
import numpy as np

pop = AgeStructuredPopulation(...)
state, config, _ = export_state(pop)

# ❌ 低效：创建多个 Population 对象
results = []
for i in range(1000):
    pop_i = AgeStructuredPopulation(...)  # 重复初始化！
    pop_i.run(100)
    results.append(pop_i.get_total_count())

# ✅ 高效：共享配置，批量运行
particles = batch_ticks(
    state, config,
    n_particles=1000,
    n_steps_per_particle=100,
    rng=np.random.default_rng(seed=42),
    record_history=False
)

results = [p.individual_count.sum() for p in particles]
```

**性能对比**：
- 低效方式：10-50 秒
- 高效方式：1-5 秒（10-50 倍加速！）

### 4️⃣ 减少历史记录

```python
# ❌ 记录每一步（内存多，速度慢）
pop.run(n_steps=1000, record_every=1)

# ✅ 只记录关键步骤
pop.run(n_steps=1000, record_every=100)  # 10 条记录

# 或完全不记录
pop.run(n_steps=1000, record_every=0)  # 无记录
```

**内存使用**：
- 每条记录：约 1-10 MB（取决于 n_genotypes）
- 1000 steps with record_every=1：1-10 GB
- 1000 steps with record_every=100：10-100 MB

## 性能分析

### 使用 cProfile 进行性能分析

```python
import cProfile
import pstats

# 进行性能分析
pr = cProfile.Profile()
pr.enable()

pop = AgeStructuredPopulation(...)
pop.run(n_steps=100, record_every=10)

pr.disable()

# 输出结果
ps = pstats.Stats(pr)
ps.sort_stats('cumulative')
ps.print_stats(20)  # 前 20 个最耗时的函数
```

**预期输出**：
```
run_tick      80%   (主要耗时)
run_reproduction  50%
run_survival  20%
run_aging     10%
```

### 识别瓶颈

1. **如果 run_tick 耗时 > 90%**：问题在核心算法（可能需要优化 Numba）
2. **如果初始化耗时 > 30%**：生成映射矩阵很复杂（增加 n_ages 或减少 n_genotypes）
3. **如果 Hook 耗时 > 10%**：Hook 逻辑过复杂

## 并行计算（高级）

虽然 Numba 支持 @njit(parallel=True)，但 NATAL 的核心未使用并行。可以在用户层面利用：

```python
from multiprocessing import Pool
from natal.simulation_kernels import export_state, batch_ticks
import numpy as np

pop = AgeStructuredPopulation(...)
state, config, _ = export_state(pop)

# 分成多个批次并行运行
def run_batch(batch_idx):
    particles = batch_ticks(
        state, config,
        n_particles=100,
        n_steps_per_particle=100,
        rng=np.random.default_rng(seed=batch_idx),
    )
    return [p.individual_count.sum() for p in particles]

with Pool(4) as p:  # 4 个进程
    all_results = p.map(run_batch, range(10))  # 总共 1000 次运行

# 合并结果
results = [r for batch in all_results for r in batch]
```

## 性能监控

### 运行时统计

```python
import time

pop = AgeStructuredPopulation(...)

# 测量初始化
start = time.perf_counter()
pop = AgeStructuredPopulation(...)
init_time = time.perf_counter() - start

# 测量运行
start = time.perf_counter()
pop.run(n_steps=100, record_every=10)
run_time = time.perf_counter() - start

# 输出
print(f"Initialization: {init_time:.2f} s")
print(f"Simulation: {run_time:.2f} s ({run_time/100:.1f} ms/tick)")

# 预测大规模运行时间
large_run_time = (run_time / 100) * 10000  # 10000 steps
print(f"Estimated time for 10000 steps: {large_run_time:.1f} s")
```

---

## 🎯 性能检查清单

- [ ] 启用 Numba：`NJIT_ENABLED_GLOBAL = True`
- [ ] 基因型数量合理（≤ 100）
- [ ] 大规模 MC 模拟使用 `batch_ticks()`
- [ ] Hook 逻辑简单高效
- [ ] 历史记录 `record_every` 设置合理
- [ ] 编译缓存清理过
- [ ] 性能预估与实际吻合

---

## 📊 性能参考表

| 配置 | 首 tick | 平均 | 1000 steps |
|------|---------|------|-----------|
| n_gen=10 | 1.2 s | 8 ms | 8.2 s |
| n_gen=30 | 1.5 s | 25 ms | 25.5 s |
| n_gen=100 | 2.0 s | 100 ms | 102 s |
| n_ages=8 | +0.1 s | +1 ms | +1 s |

（在 2020 年 MacBook 上测得）

---

## 📚 相关章节

- [Simulation Kernels 深度解析](03_simulation_kernels.md) - @njit 函数的详细说明
- [PopulationState & PopulationConfig](04_population_state_config.md) - @jitclass 的使用
- [Hook 系统](07_hooks.md) - Hook 性能优化
- [快速开始](01_quickstart.md) - 实际使用中的性能调优

---

**准备查阅完整 API 了吗？** [前往最后一章：API 完整参考 →](09_api_reference.md)
