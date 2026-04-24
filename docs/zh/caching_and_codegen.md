# Numba 缓存与代码生成机制

本文档解释 `natal` 中 Numba 编译缓存的工作原理，以及为什么需要代码生成（codegen）来保证缓存跨进程有效。

## 背景：Numba 的缓存机制

Numba 的 `@njit(cache=True)` 装饰器可以将编译后的函数缓存到磁盘（`__pycache__`）。其缓存键由以下要素构成：

- 函数的限定名（`__qualname__`）
- 函数的字节码哈希（bytecode hash），基于编译后的字节码而非源码文本，因此仅改变注释不影响缓存，但改变变量名会
- 函数引用的全局变量的类型签名

当同一个 Python 函数在后续运行时被重新导入，Numba 会重新计算缓存键，若与磁盘上的缓存匹配，则直接加载编译后的机器码，跳过编译过程。

## 核心问题：函数参数与跨进程缓存失效

### 函数参数方式（不可缓存）

在最初的无 codegen 方案中，hook 函数通过参数传递给生命周期函数：

```python
@njit_switch(cache=True)
def run_discrete_tick_with_hooks(
    state, config, registry,
    first_hook: Callable,   # ← 函数作为参数
    early_hook: Callable,
    late_hook: Callable,
):
    ...
    result = first_hook(ind_count, tick)
```

当 Numba 编译这个函数时，`first_hook` 参数的类型是具体某个 `Dispatcher` 实例。Numba 会为每组不同的 Dispatcher 参数创建一个**特化版本**（specialization）。特化版本的缓存键中包含该 Dispatcher 的**类型标识信息**（including overload fingerprints 等与实例绑定的信息）。

问题在于：跨进程重启后，即使同一个 Python 源代码，Numba 也会创建全新的 `Dispatcher` 实例。新的实例携带不同的的类型标识（因为 overload 分辨率等内部状态是新生成的），导致之前编译的特化版本的缓存键不匹配，Numba 被迫**重新编译**。

### 全局变量方式（可缓存）

代码生成方案改为将 hook 函数设为模块级全局变量：

```python
# 通过代码生成创建的独立模块
_FIRST_HOOK = None   # 稍后注入
_EARLY_HOOK = None
_LATE_HOOK = None

@njit_switch(cache=True)
def _lifecycle_tick_527c055(...):
    ...
    result = _FIRST_HOOK(ind_count, tick)  # ← 全局变量，不是参数
```

Numba 编译 `_lifecycle_tick_527c055` 时，`_FIRST_HOOK` 是模块级全局变量。Numba 的缓存键只依赖函数自身的源码文本和函数名，以及全局变量的**类型签名**（`(ind_count, tick) -> int`）。它**不依赖**该全局变量指向的具体 Dispatcher 对象的身份标识。

因此，只要生成的函数名和源码不变（由 hash_key 保证），缓存键跨进程稳定。

## 代码生成概述

当前有两类代码生成：

### 1. Combined Hook 生成

将多个同一事件的 @njit hook 合并为一个函数，避免逐个调用的开销。

**入口**：`compile_combined_hook(njit_fns, name)`

**生成逻辑**（`natal/hooks/compiler.py`：

```python
lines = ["from natal.hook_dsl import njit_switch"]
lines.extend([f"{placeholder} = None" for placeholder in placeholder_names])
lines.append(f"def {fn_name}(ind_count, tick, deme_id=0):")
for placeholder in placeholder_names:
    lines.append(f"    _result = {placeholder}(ind_count, tick, deme_id)")
    lines.append("    if _result != 0:")
    lines.append("        return _result")
lines.append("    return 0")
```

生成的模块文件：

```python
_FN_0 = None
_FN_1 = None

@njit_switch(cache=True)
def _combined_hook_19a81f6c(...):
    _result = _FN_0(ind_count, tick, deme_id)
    if _result != 0: return _result
    _result = _FN_1(ind_count, tick, deme_id)
    if _result != 0: return _result
    return 0
```

### 2. Lifecycle Wrapper 生成

将完整的生命周期循环（reproduction → survival → aging，以及三个事件的 CSR 操作和 hook 调用）生成为一个独立的 @njit 函数。

**入口**：`compile_lifecycle_wrapper(is_discrete, first_hook, early_hook, late_hook)`

生成两个函数：
- `_lifecycle_tick_{hash}`：单次 tick，包含 CSR 事件执行和 hook 调用
- `_lifecycle_run_{hash}`：多次 tick 循环，含历史记录

根据 `is_discrete` 参数决定使用离散世代（无 sperm storage）还是有年龄结构（有 sperm storage）的版本。

生成的模块文件示例（`hook_codegen/lifecycle_discrete_527c055.py`）：

```python
import numpy as np
from natal.kernels.simulation_kernels import (
    run_discrete_reproduction, run_discrete_survival, run_discrete_aging,
)
from natal.hooks.executor import execute_csr_event_program_with_state
from natal.hooks.types import EVENT_FIRST, EVENT_EARLY, EVENT_LATE, ...

_FIRST_HOOK = None
_EARLY_HOOK = None
_LATE_HOOK = None

@njit_switch(cache=True)
def _lifecycle_tick_527c055(state, config, registry):
    ind_count = state.individual_count.copy()
    tick = state.n_tick
    # 执行 FIRST 事件（CSR + hook）
    result = execute_csr_event_program_with_state(registry, EVENT_FIRST, ...)
    result = _FIRST_HOOK(ind_count, tick)
    # reproduction
    ind_count = run_discrete_reproduction(ind_count, config)
    # 执行 EARLY 事件
    ...
    ind_count = run_discrete_survival(ind_count, config)
    # 执行 LATE 事件
    ...
    ind_count = run_discrete_aging(ind_count)
    return (ind_count, tick + 1), RESULT_CONTINUE
```

## 关键函数与数据流

### 1. `stable_callable_identity(fn)` → `module:qualname`

生成一个跨进程稳定的标识字符串，用于标识一个 callable。对 @njit 函数，取 `fn.py_func.__module__` 和 `__qualname__`；对普通函数同理：

```python
def _stable_callable_identity(fn):
    py_fn = getattr(fn, "py_func", fn)
    module_name = getattr(py_fn, "__module__", "<unknown>")
    qualname = getattr(py_fn, "__qualname__", getattr(py_fn, "__name__", "<unknown>"))
    return f"{module_name}:{qualname}"
```

### 2. `hash_key(parts)` → 16 字符十六进制

用 SHA-256 计算一段标识的摘要，截取前 16 位作为模块和函数名的后缀：

```python
def _hash_key(parts):
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]
```

### 3. 文件生成与加载

```python
# 生成
module_path = write_codegen_module(stem, source)
# → 写入 .numba_cache/hook_codegen/{stem}.py

# 加载
module = load_codegen_module(stem, module_path)
# → 作为 natal._hook_codegen_{stem} 导入 sys.modules

# 注入全局变量
setattr(module, "_FIRST_HOOK", first_hook)

# 获取目标函数
tick_fn = getattr(module, fn_name)
```

**时序约束**：`setattr` 注入全局变量必须在首次调用 `_lifecycle_tick_*` 之前完成。这是因为 Numba 在编译/加载缓存时会检查全局变量的类型签名：如果首次调用时 `_FIRST_HOOK` 仍为 `None`（`NoneType`），而缓存中记录的是 `Dispatcher`（某种 callable 类型），类型不匹配会导致缓存失效。从数据流来看当前的代码保证了这一时序（注入发生在模块加载后、首次调用前），但这是一个需要维护的设计约束。

### 4. 完整数据流

```
用户注册 hook
      ↓
  compiler.hook() 装饰器
      ↓
  pop.register_compiled_hook(desc)
      ↓
  get_compiled_event_hooks()
      ↓
  CompiledEventHooks.from_compiled_hooks()
      ├─ compile_combined_hook() → 合并同事件 hook
      └─ compile_lifecycle_wrapper() → 生成 tick/run 函数
           ├─ _gen_lifecycle_source() → 构建源码
           ├─ write_codegen_module() → 写入磁盘
           ├─ load_codegen_module() → 导入模块
           └─ setattr(module, "_XXX_HOOK", hook) → 注入全局变量
      ↓
  种群调用 hooks.run_fn() 或 hooks.run_discrete_fn()
```

### 5. Numba 单例默认 hook

`_noop_hook` 作为一个单例的 `@njit(cache=True)` 函数，在没有任何 hook 注册时作为默认值。所有生命周期模块在没有 hook 的情况下都会引用同一个 `_noop_hook` 作为 `_FIRST_HOOK` / `_EARLY_HOOK` / `_LATE_HOOK`，因此 Numba 缓存仍然有效。

```python
@njit_switch(cache=True)
def _noop_hook(ind_count: np.ndarray, tick: int, deme_id: int = 0) -> int:
    return 0
```

## 为什么全局变量方式可缓存而参数方式不可

| 方面 | 函数参数方式 | 全局变量方式 |
|---|---|---|
| Numba 接收 | `first_hook` 作为参数，类型为具体 Dispatcher | `_FIRST_HOOK` 作为全局变量引用 |
| 特化数量 | 每组合 hook 创建新的特化 | 始终单一编译（函数名 + 源码决定） |
| 缓存键内容 | 函数字节码 + Dispatcher 类型标识信息 | 函数字节码 + 函数名 |
| 跨进程稳定性 | ❌ Dispatcher 类型标识信息每次重启不同 | ✅ 函数名和字节码跨进程不变 |
| 隔离性 | 隐含（参数隔离） | 需要通过不同模块名隔离 |

全局变量方式的核心在于：**Numba 在处理全局变量时，主要校验的是其类型签名兼容性而非对象身份**。而 `hash_key` 保证了不同 hook 组合生成不同函数名，相同 hook 组合生成相同函数名，因此每个组合只有一份编译结果，且跨进程可复用。

## 挂起问题

### 无 hook 情况

当没有任何 hook 注册时，`from_compiled_hooks` 仍然会编译 lifecycle wrapper（因为 `NUMBA_ENABLED` 为 True），三个全局变量均设为 `_noop_hook`。此时生成的函数名是固定的（`_noop_hook` 的 `stable_callable_identity` 唯一），因此每个种群类型（离散/结构化）只有一份缓存。

当 Numba 禁用（如测试环境），`from_compiled_hooks` 不编译 lifecycle wrapper，`run_fn` / `run_discrete_fn` 保持为 `None`。种群代码会 fallback 到参数版本的内核函数（`run_with_hooks` / `run_discrete_with_hooks`），这些函数通过 `njit_switch` 自动退化为 Python 执行。
