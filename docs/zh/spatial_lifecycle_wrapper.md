# Spatial Model 生命周期包装器重构

## 背景

原有的 spatial simulation kernel（`run_spatial_tick_with_migration`）是一个"半成品"：

1. **不支持 hooks** — 只能在每个 deme 内部运行生命周期阶段（reproduction → survival → aging），无法执行用户注册的 hook 事件（first/early/late）
2. **不支持异构 config** — 只能接受一个共享的 `PopulationConfig`，所有 deme 必须使用完全相同的参数
3. **Python dispatch 回退过宽** — 只要有 hooks 或异构 config，整个 spatial run 就退化为逐 deme 的 Python 循环，完全无法利用 Numba 加速

此前 panmictic（单种群）路径已经通过 `compile_lifecycle_wrapper` 解决了 hooks + njit 共存的问题。本次修改将同样的思路扩展到 spatial 路径。

## 重构目标

消除 spatial 生命周期序列与 panmictic 的重复：让 spatial 的 prange 体内**直接调用 panmictic 的 lifecycle tick 函数**，而不是重写阶段调用序列。

## 修改内容

### 1. compiler.py — Panmictic tick 增加 `deme_id` 参数

在 `_gen_lifecycle_source` 中，tick 函数签名增加 `deme_id=-1`：

```python
# 重构前
def _lifecycle_tick_<hash>(state, config, registry):
    ...
    result = _FIRST_HOOK(ind_count, tick)

# 重构后
def _lifecycle_tick_<hash>(state, config, registry, deme_id=-1):
    ...
    result = _FIRST_HOOK(ind_count, tick, deme_id)
```

对应的 tick body 中对 `_FIRST_HOOK`/`_EARLY_HOOK`/`_LATE_HOOK` 的调用和 `execute_csr_event_program_with_state` 的调用都传递了 `deme_id`。

这样 spatial 路径可以传入真实的 deme 索引（`d`），使 hooks 能感知 deme 上下文；而 panmictic 路径不传此参数（默认 `-1`），行为不变。

### 2. compiler.py — Spatial lifecycle wrapper 委托给 panmictic tick

spatial 的 prange 体不再自行排列生命周期阶段，改为**导入 panmictic lifecycle tick 函数并在 prange 内调用**：

```
# 重构前
for d in prange(n_demes):
    cfg = config_bank[deme_config_ids[d]]
    ind = ind_all[d].copy()
    execute_csr(FIRST, ...)    ← 重复 stage
    _FIRST_HOOK(...)
    run_reproduction(...)       ← 重复 stage
    execute_csr(EARLY, ...)
    _EARLY_HOOK(...)
    run_survival(...)
    execute_csr(LATE, ...)
    _LATE_HOOK(...)
    run_aging(...)

# 重构后
for d in prange(n_demes):
    cfg = config_bank[deme_config_ids[d]]
    ind = ind_all[d].copy()
    sperm = sperm_all[d].copy()
    state = PopulationState(tick, ind, sperm)    ← 构造 State 对象
    (ind, sperm, _), result = _run_deme_tick(state, cfg, registry, d)
```

spatial 模块不再包含自己的 `_FIRST_HOOK`/`_EARLY_HOOK`/`_LATE_HOOK` 全局变量——hook globals 只在 panmictic 模块上设置，spatial 模块导入的 panmictic tick 函数通过其自身模块的全局变量解析 hooks。

### 3. compiler.py — Spatial source 生成简化

`_gen_spatial_lifecycle_source` 现在接受 `panmictic_stem` 和 `panmictic_tick_fn_name` 参数。生成的模块源码大幅简化：

```python
# 重构前：6 个 import、3 个 hook 全局变量
import numpy as np
from natal.kernels.simulation_kernels import (run_reproduction, ...)
from natal.kernels.spatial_migration_kernels import run_spatial_migration
from natal.hooks.executor import execute_csr_event_program_with_state
from natal.hooks.types import EVENT_FIRST, EVENT_EARLY, EVENT_LATE, ...
from natal.numba_utils import njit_switch, prange

_FIRST_HOOK = None
_EARLY_HOOK = None
_LATE_HOOK = None

# 重构后：3 个 import + 1 个 panmictic tick import
import numpy as np
from natal.kernels.spatial_migration_kernels import run_spatial_migration
from natal.hooks.types import RESULT_CONTINUE, RESULT_STOP
from natal.numba_utils import njit_switch, prange
from natal.population_state import PopulationState
from natal._hook_codegen_lifecycle_structured_<key> import _lifecycle_tick_<key> as _run_deme_tick
```

### 4. compiler.py — CompiledEventHooks 扩展

在 `CompiledEventHooks` 中新增了 4 个槽位：

- `spatial_tick_fn` / `spatial_run_fn` — age-structured 的 spatial 生命周期包装器
- `spatial_discrete_tick_fn` / `spatial_discrete_run_fn` — discrete-generation 的 spatial 生命周期包装器

在 `from_compiled_hooks()` 中，当 Numba 启用时，除了原有的 panmictic wrapper，还会**预编译** spatial wrapper：

```python
if NUMBA_ENABLED:
    # Panmictic wrappers
    result.run_tick_fn, result.run_fn = compile_lifecycle_wrapper(...)
    result.run_discrete_tick_fn, result.run_discrete_fn = compile_lifecycle_wrapper(...)
    # Spatial wrappers（委托给上述 panmictic wrappers）
    result.spatial_tick_fn, result.spatial_run_fn = compile_spatial_lifecycle_wrapper(...)
    result.spatial_discrete_tick_fn, result.spatial_discrete_run_fn = compile_spatial_lifecycle_wrapper(...)
```

### 5. spatial_population.py — 运行时适配

#### `_should_use_python_dispatch()` 收缩

原有的条件：
```python
if not is_numba_enabled(): return True
if has_python_hooks() or has_compiled_hooks(): return True  # ← 过宽
return has_heterogeneous_configs()                           # ← 过宽
```

新的条件：
```python
if not is_numba_enabled(): return True
if has_python_hooks(): return True     # 只有纯 Python callback 才回退
return False                           # 其余全部走 njit
```

这意味着：
- **CSR registry hooks**（声明式 Op、njit selector hooks）→ 可通过 `execute_csr_event_program_with_state` 在 njit 内执行
- **用户 njit hooks** → 通过模块级全局变量在 njit 内执行
- **异构 configs** → 通过 `config_bank` 在 njit 内按 deme 索引查找
- 纯 Python callable hooks → 仍需回退 Python dispatch

#### `_is_discrete_demes()` 辅助方法

通过检查第一个 deme 的 state 是否包含 `sperm_storage` 属性来判断 deme 类型，从而选择 structured 或 discrete 的 spatial wrapper。

#### `_run_codegen_wrapper_tick()` 替换

从调用 `run_spatial_tick_with_migration(single_config)` 改为调用 `spatial_tick_fn(config_bank, deme_config_ids, registry, ...)`：

1. 调用 `_stack_deme_state_arrays()` 堆叠所有 deme 的状态
2. 调用 `_heterogeneous_config_bank_and_ids()` 构建 config bank
3. 根据 `_is_discrete_demes()` 选择 structured 或 discrete 的 tick 函数
4. 传入 registry（CSR hook 数据）和 migration 参数
5. 写回状态

#### `_run_codegen_wrapper_steps()` 替换

同上，但使用 `spatial_run_fn` 一次性执行多个 tick，支持 `record_interval` 历史记录。

## Spatial Model 完整工作流程

### 构建阶段

```
Species + Drive + Demes → IndexRegistry / PopulationConfig / PopulationState
                          ↓
SpatialPopulation.__init__()
                          ↓
_compile_spatial_hooks_from_demes()
    → _collect_effective_compiled_hooks()     ← 收集所有 deme 的 hook
    → _build_hook_program()                   ← 编译 CSR HookProgram
    → CompiledEventHooks.from_compiled_hooks()
        → compile_combined_hook()             ← 合并同事件 njit hooks
        → compile_lifecycle_wrapper()         ← 预编译 panmictic wrapper
        → compile_spatial_lifecycle_wrapper() ← 预编译 spatial wrapper
```

### 运行阶段 — `run_tick()`

```
spatial.run_tick()
  │
  ├─ _should_use_python_dispatch()?
  │    ├─ True  → _run_python_dispatch_tick()
  │    │            for deme in demes: deme.run_tick()
  │    │            run_spatial_migration(stacked_state)
  │    │
  │    └─ False → _run_codegen_wrapper_tick()
  │                  _stack_deme_state_arrays()
  │                  _heterogeneous_config_bank_and_ids()
  │                  spatial_tick_fn(config_bank, registry, ...)
```

### njit prange 内部流程（一次 tick）

```
_spatial_tick_<hash>(ind_all, sperm_all, config_bank, deme_config_ids, registry, tick, ...)
  │
  ├─ n_demes = ind_all.shape[0]
  ├─ stopped = zeros(n_demes, bool)
  │
  ├─ for d in prange(n_demes):              ← 并行执行每个 deme
  │    │
  │    ├─ cfg = config_bank[deme_config_ids[d]]  ← 异构 config 查找
  │    │
  │    ├─ 构造 PopulationState(tick, ind, sperm)
  │    │
  │    ├─ (ind, sperm, _), result = _run_deme_tick(state, cfg, registry, d)
  │    │    │                                    ← 委托给 panmictic tick
  │    │    ├─ [FIRST 事件]                        （带 deme_id=d）
  │    │    │    execute_csr_event_program(registry, FIRST, ind, sperm, tick, d)
  │    │    │    _FIRST_HOOK(ind, tick, d)
  │    │    │
  │    │    ├─ Reproduction（繁殖）
  │    │    │
  │    │    ├─ [EARLY 事件]
  │    │    │    execute_csr_event_program(registry, EARLY, ind, sperm, tick, d)
  │    │    │    _EARLY_HOOK(ind, tick, d)
  │    │    │
  │    │    ├─ Survival / Competition（生存/竞争）
  │    │    │
  │    │    ├─ [LATE 事件]
  │    │    │    execute_csr_event_program(registry, LATE, ind, sperm, tick, d)
  │    │    │    _LATE_HOOK(ind, tick, d)
  │    │    │
  │    │    └─ Aging（老化）→ 返回 (ind, sperm, tick+1), result
  │    │
  │    ├─ if result != CONTINUE: stopped[d] = True
  │    ├─ ind_all[d] = ind
  │    └─ sperm_all[d] = sperm
  │
  ├─ run_spatial_migration(                 ← prange 完成后统一迁移
  │      ind_all, sperm_all, ...,
  │      config_bank[0], ...)
  │
  └─ 检查 stopped[] → 返回 was_stopped
```

### 关键设计决策

1. **Panmictic tick 作为唯一事实源**：spatial 的 prange 体不再重复生命周期阶段序列，而是**委托给 panmictic lifecycle tick**。生命周期顺序（FIRST → reproduction → EARLY → survival → LATE → aging）**只在一个地方定义**，新增/调整阶段不会漏掉 spatial 路径

2. **deme_id 传递**：panmictic tick 的 `deme_id=-1` 默认参数让两种调用路径都能正常工作：
   - Panmictic 调用：不传 deme_id → 默认 -1 → 行为不变
   - Spatial 调用：传 `d`（deme 索引）→ hooks 能感知 deme 上下文

3. **Config bank 始终使用**：即使所有 deme 共用同一个 config，也通过 config bank 传递，保持生成模块的签名统一

4. **Migration 使用 config_bank[0]**：migration kernel 只需要读取 `is_stochastic` 和 `use_continuous_sampling` 两个参数，这些在 spatial population 构建时已验证为所有 deme 一致

5. **Stop 信号收集**：prange 内无法直接 break 回主线程，使用 `stopped[n_demes]` 布尔数组在每个 deme 的生命周期中标记。prange 结束后串行扫描 stopped 数组

6. **Hook globals 集中在 panmictic 模块**：spatial 模块不再设置 `_FIRST_HOOK` 等全局变量。这些只在 panmictic 模块上设置，spatial 导入的 panmictic tick 通过其自身模块的全局变量解析 hooks。每个唯一 hook 组合对应一个唯一的源码 hash，确保 Numba `cache=True` 跨进程工作

## 用户 API：简化后的 `@hook`

`@hook()` 装饰器的 `deme_selector` 参数已重命名为 `deme`，语义更直观：

```python
@hook(event="early", custom=True, deme="*")    # 所有 deme（默认）
@hook(event="early", custom=True, deme=3)       # 仅 deme 3
@hook(event="early", custom=True, deme=[0,2,4]) # 指定列表
```

不需要手动加 `@njit` — 装饰器自动处理：

```python
@hook(event="early", custom=True)
def my_hook(ind_count, tick, deme_id=-1):
    """Numba 启用时自动 njit 编译，禁用时用 Python 回退。"""
    if deme_id % 2 == 0:
        ind_count[0, 0, 0] *= 0.5
```

配合 `PopulationBuilder` 使用：

```python
pop = (
    nt.DiscreteGenerationPopulation
    .setup(species=sp, name="demo")
    .initial_state(...)
    .reproduction(...)
    .competition(...)
    .presets(drive)
    .hooks(my_hook)       # @hook 装饰过的函数直接传入
    .build()
)
```
