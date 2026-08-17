# #1 单一生命周期 tick 模块 — 设计方案

> 状态：**已实施**。本文保留实施完成前的设计共识，供追溯。
> 本文记录 architecture review（2026-08-13）候选 #1 的设计共识，供实施时遵循。
> 术语（模块 / 接口 / 深度 / 接缝 / 本地性 / 杠杆 / 删除测试）遵循 `codebase-design` 词汇表。

---

## 1. 背景与问题

### 1.1 现状

"一个 tick 按什么顺序做事"（first hook → 繁殖 → early hook → 存活 → late hook → 老化）这一概念在代码库中存在约 6 份并行实现：

1. 生成模板 `engine/templates/lifecycle_structured.tmpl.py`（Numba 开，结构化）
2. 生成模板 `engine/templates/lifecycle_discrete_v2.tmpl.py`（Numba 开，离散）
3. 生成模板 `engine/templates/lifecycle_wf.tmpl.py`（Numba 开，Wright-Fisher 融合 tick）
4. 模拟器回退 `engine/age_structured_simulator.py` 的 `run_with_hooks` / `run_tick_with_hooks` / `_event_with_hooks`
5. 种群层 Python 分派 `population/age_structured.py::_run_python_dispatch`、`population/discrete_generation.py::_run_python_dispatch`、`_run_wright_fisher`
6. 空间 Python 分派 `spatial/population.py::_run_python_dispatch_tick`

### 1.2 已经发生的分叉（正确性 bug）

- **hook 调用形状分歧**：模板调用 `hook(state, config, deme_id)`（`lifecycle_structured.tmpl.py:75-78`），而模拟器回退调用 `combined_hook(ind_count, tick)`（`age_structured_simulator.py:451`）——ndarray 被当作 state、int 被当作 config。回退路径（Numba 关闭 + 无 hook 时触发）与正常路径静默不兼容。
- **no-op hook 对齐不变量分叉**：`codegen.py` 的 HookProgram 打包器给每个 hook 追加 deme-selector 条目，另两份打包器（`_mixins/_hooks.py`、`spatial/population.py`）对 no-op hook 跳过——CSR 内核索引可能错位。
- 多份实现之间无法用测试保证一致（回退路径很少被触发，分叉不可见）。

### 1.3 目标

- **知识层统一**：tick 编排（阶段顺序、hook 调用形状、STOP 短路）只有一份定义。
- **机制强制**：改顺序 = 改一处 = 两条路径同步变，不靠测试"保证一致"。
- **Numba 关零 codegen**：纯 Python 路径直接解释执行，不写文件、不 importlib、不 setattr。
- **无模板文件**：`engine/templates/lifecycle_*.tmpl.py` 全部删除，由拼装器按源函数生成。

---

## 2. 架构总览：两套 codegen 的区分

| | Codegen 1：hook 拼装 | Codegen 2：lifecycle 编排 |
|---|---|---|
| 归属 | hooks 包（`hooks/compile/`、`hooks/entry/`） | engine 包（新建 `engine/lifecycle.py`） |
| 输入 | 事件的全部 hook 描述符（带 priority） | 阶段顺序 + hook 组合函数 |
| 输出 | `(state, config, deme_id) -> int` 组合函数 | tick 函数 + 多 tick 循环（RUN） |
| Numba 开 | `compile_combined_hook` / `compile_unified_event_hook` 生成 njit 源码模块 | 拼装器按源函数 getsource 拼出生成模块 |
| Numba 关 | 普通 Python 组合闭包（无 codegen） | `engine/lifecycle.py` 的 Python 执行器（零 codegen） |

接合点：Codegen 1 的输出（组合函数）恰好是 Codegen 2 的输入（`_run_event` 里 `hook(state, config, deme_id)` 这一行要的 hook）。本次 #1 基本不动 Codegen 1（只砍 1 参 hook）。

---

## 3. 详细设计

### 3.1 新文件：`engine/lifecycle.py`

包含全部"源函数"（普通 Python 函数，**不装饰**、njit 友好写法、静态定义在文件中以便 `inspect.getsource`）：

| 函数 | 职责 | 角色 |
|---|---|---|
| `run_structured_tick` | 结构化单 tick 编排（写死三行事件 + 三行阶段调用） | Numba 关：直接调；Numba 开：getsource 嵌入 → `_lifecycle_tick_{key}` |
| `run_discrete_tick` | 离散单 tick 编排 | 同上 |
| `run_wf_tick` | WF 单 tick（仅 first 事件 + 融合阶段） | 同上 |
| `run` | 多 tick 循环 + 记录回调 | **仅 Numba 关**：直接调，`record_fn` 回调注入 |
| `_run_event` | 事件块（CSR → hook → STOP 短路） | Numba 关：被 `run_*_tick` 调；Numba 开：getsource 嵌入 |
| `_run_loop` | 多 tick 循环 + 数组记录 | **仅 Numba 开**：getsource 嵌入 → `_lifecycle_run_{key}` |
| `_spatial_tick_shell` / `_spatial_run_shell` | 空间壳（prange + 迁移 + 记录） | 仅 Numba 开：getsource 嵌入 |

### 3.2 阶段函数统一签名

**决策**：阶段函数统一签名 `(state, config) -> state`。现有 7 个阶段函数签名全部修改（非公开 API，调用点都在 #1 范围内）：

| 函数 | 现在签名 | 改后 |
|---|---|---|
| `run_reproduction` / `run_survival` / `run_aging`（结构化） | `(ind, sperm, config) -> (ind, sperm)` | `(state, config) -> state` |
| `run_discrete_reproduction` / `run_discrete_survival` / `run_discrete_aging` | `(ind, config) -> ind` 等 | `(state, config) -> state` |
| `run_wf_loop`（22 参宽接口） | 22 个标量 | 经 `run_wf_tick(state, config)` 适配（宽接口问题留给 #7） |

- 阶段顺序**写死**在 `run_*_tick` 函数体里（不做阶段表、不做遍历、不做事件→hook 映射）。
- `run_wf_tick` 收三 hook 参数但忽略 early/late（签名统一，内部不用）。

### 3.3 `_run_event`（事件块）

```python
def _run_event(event_id, state, config, registry, _EVENT_HOOK, deme_id, _HAS_SPERM, _SPERM_SOURCE):
    """执行一个事件：CSR 声明式操作 → 自定义 hook → STOP 短路。"""
    ind = state.individual_count
    result = execute_csr_event_program_with_state(
        registry, event_id, ind, _SPERM_SOURCE(state), state.n_tick,
        bool(config.stochastic), _HAS_SPERM, bool(config.continuous_sampling), deme_id,
    )
    if result != RESULT_CONTINUE:
        return RESULT_STOP
    result = _EVENT_HOOK(state, config, deme_id)
    return RESULT_STOP if result != 0 else RESULT_CONTINUE
```

- `_EVENT_HOOK`：该事件的组合 hook（Codegen 1 产物）。
- `_HAS_SPERM` / `_SPERM_SOURCE`：模式差异哨兵。生成时替换为 `True`/`False` + `state.sperm_storage`/`None`；Python 路径传真值/真函数。
- **不使用 dummy 空数组**（避免 njit 分配开销）。

### 3.4 dummy 消除

**决策**：`execute_csr_event_program_with_state` 的 sperm 参数改为 `Optional[NDArray]`，`has_sperm_storage=False` 时传 `None`（内核该分支不访问 sperm）。消除 `_run_event`、离散 tick、空间离散壳、现有 `_run_python_dispatch` 中全部 `np.zeros((0,0,0))` dummy。

### 3.5 多 tick 循环

两条路径因**记录机制不同**各有一份（Numba 开 = njit 数组拼装；Numba 关 = Python 回调），合并留给 #5（记录器）：

**Numba 关**（`engine/lifecycle.py::run`）：

```python
def run(tick_fn, state, config, registry,
        first_hook, early_hook, late_hook, deme_id,
        n_steps, record_every, record_fn) -> tuple[object, bool]:
    """Numba 关闭时的多 tick 循环：每 tick 调 tick_fn，按 record_every 调 record_fn。"""
    for _ in range(n_steps):
        state, result = tick_fn(state, config, registry, first_hook, early_hook, late_hook, deme_id)
        if result != RESULT_CONTINUE:
            return state, True
        if record_every > 0 and (state.n_tick % record_every == 0):
            record_fn(state)
    return state, False
```

- `tick_fn` 显式传（run 纯 Python，不涉 njit 缓存）。
- `record_fn(state)` 由种群层适配（写回 `_state`/`_tick` 再调 `_record_current_snapshot`）。
- `record_every=0` 时不调 `record_fn`（空间容器禁用 deme 记录，不再手动置零 `deme.record_every`）。

**Numba 开**（`engine/lifecycle.py::_run_loop`）：现有 RUN 模板的循环 + 记录逻辑**原样搬入**（不改语义，记录层留给 #5），`_TICK_FN` 哨兵生成时替换成 `_lifecycle_tick_{key}`。

### 3.6 拼装器

`engine/lifecycle.py::assemble_lifecycle_module(mode, tick_fn_name, run_fn_name) -> str`：

- 一个装配入口，模式 ∈ {"structured", "discrete", "wf", "spatial_structured", "spatial_discrete"}。
- 模式差异收敛在内部 dict：源函数选择 + 替换表 + import 列表。
- 步骤：
  1. 取源函数源码（`inspect.getsource`）；
  2. 替换哨兵（`_EVENT_HOOK` → `_FIRST_HOOK` 等；`_HAS_SPERM` → `True`/`False`；`_SPERM_SOURCE` → `state.sperm_storage`/`None`；`_TICK_FN` → tick 函数名；`_DEME_TICK`/`_SPATIAL_TICK` → 空间 tick 名）；
  3. 签名行替换（`run_structured_tick(state, config, registry, first_hook, early_hook, late_hook, deme_id=-1)` → `def _lifecycle_tick_{key}(state, config, registry, deme_id=-1)`）；
  4. hook 参数名 → 全局名替换（`first_hook` → `_FIRST_HOOK` 等）；
  5. 拼头部样板（import、`@njit_switch(cache=True)` 装饰、函数签名）；
  6. 返回完整源码字符串。

**实现约定**（保证替换无歧义）：源函数体是"生成友好"的——hook 参数名/哨兵名只作为参数和调用出现，注释和字符串不用这些词，无 f-string。只做字符串级操作，不做 AST 解析。

### 3.7 `compile_lifecycle_wrapper` 改造

`engine/lifecycle_wrappers.py`：

- `_gen_lifecycle_source` / `_read_engine_template` 删除。
- `compile_lifecycle_wrapper(mode, first_hook, early_hook, late_hook)`：mode ∈ {"structured", "discrete", "wf"}，内部调 `assemble_lifecycle_module`；`compile_wf_lifecycle_wrapper` 合并进来。
- 缓存键**维持现状**（只含 hook 组合：`["lifecycle_" + mode] + [stable_callable_identity(fn) ...]`）。阶段表/源函数变化靠 `.numba_cache/` 内容对比（`_write_codegen_module` 发现源码不一致会重写）兜底。
- 写文件、`importlib` 加载、setattr 注入 hook 全局——全部保留。

### 3.8 空间

- 空间 = 固定壳函数（`_spatial_tick_shell` / `_spatial_run_shell`，prange + 迁移 + 记录）走同一拼装机制（getsource + 哨兵替换 `_DEME_TICK`），**无模板文件**。
- 空间壳分为 structured / discrete 两个版本（差异：sperm 有无），不在壳里塞 `_HAS_SPERM` 哨兵。
- 空间编排（prange/迁移/config bank/异构）本身**不动**（属于 #6）。
- Numba 关时空间 Python 路径：`for deme: run(tick_fn=..., record_every=0, ...)` + 迁移（现有逻辑），hook 用 deme 自己的 `CompiledEventHooks`。

### 3.9 种群 `run()` 两分支

- 删除 `should_use_python_dispatch()`（改为直接查 `NUMBA_ENABLED`）。
- 删除 `_run_python_dispatch` ×2、`_run_wright_fisher`、`run_with_hooks` / `run_tick_with_hooks` / `_event_with_hooks`。
- 两分支统一从 `wrappers.hooks` 拿 hook（`CompiledEventHooks` 为统一 hook 容器）：
  - Numba 开：`wrappers.run_fn(...)`（拼装生成的 RUN，数组记录）→ `_process_kernel_history`；
  - Numba 关：`engine.lifecycle.run(tick_fn=..., ..., record_fn=适配)`（回调记录，保持现状语义）。
- 收尾逻辑（`was_stopped` → finish；`finish` → `finish_simulation`）两分支共用。

### 3.10 1 参 hook 砍掉（破坏性 API 变更）

- hook 签名唯一 `(state, config, deme_id) -> int`（文档 `docs/zh/2_hooks.md:228` 本来只承诺三参）。
- 删除 `HookExecutor.execute_event` 的 `inspect.signature` 判别分支（`fallback.py:139-151`）。
- 删除 `decorator.py::_is_declarative_population_hook` 启发式。
- 全仓无 1 参 hook 真实使用者（demos 零实例）。
- 按项目门禁：需补负向合同测试（断言 1 参 hook 被拒绝而非被猜测），同步更新 `__init__.pyi`、docs。

---

## 4. 文件变更清单

### 新增

- `engine/lifecycle.py`（源函数 + 拼装器，~300 行）

### 删除

- `engine/templates/lifecycle_structured.tmpl.py`
- `engine/templates/lifecycle_discrete_v2.tmpl.py`
- `engine/templates/lifecycle_wf.tmpl.py`
- `engine/templates/spatial_lifecycle_structured.tmpl.py`
- `engine/templates/spatial_lifecycle_discrete.tmpl.py`
- （空间壳内容迁入 `engine/lifecycle.py`）

### 修改

- `engine/lifecycle_wrappers.py`：`compile_lifecycle_wrapper(mode, ...)` 改造、删 `_gen_lifecycle_source`、合并 WF
- `engine/age_structured_simulator.py`：删 `run_with_hooks` / `run_tick_with_hooks` / `_event_with_hooks`、4 个 export/import 传话（顺带）
- `engine/discrete_generation_simulator.py`：阶段函数改签名
- `engine/simulation/age_structured.py`：阶段函数改签名（结构化）
- `engine/simulation/discrete_generation.py`：`run_wf_loop` 保持（由 `run_wf_tick` 适配）
- `hooks/runtime/csr_kernel.py`：`execute_csr_event_program_with_state` sperm → `Optional[NDArray]`
- `hooks/runtime/fallback.py`：删 `inspect.signature` 判别
- `hooks/entry/decorator.py`：删 `_is_declarative_population_hook`
- `population/age_structured.py`：`run()` 两分支改造、删 `_run_python_dispatch`
- `population/discrete_generation.py`：同上 + 删 `_run_wright_fisher`
- `spatial/population.py`：`_run_python_dispatch_tick` 改为调 `engine.lifecycle.run`、删 record_every 置零体操
- `natal/__init__.pyi`、docs（1 参 hook 变更）

---

## 5. 测试策略

### 需要修改/删除的现有测试

- `test_lifecycle_wrappers.py`：`compile_lifecycle_wrapper` 签名变化（mode 参数）、模板删除的影响
- 引用模板文件或 `_gen_lifecycle_source` 的测试
- `test_hook_executor.py`：1 参 hook 相关用例（改为断言拒绝）
- 引用 `run_with_hooks` / `_run_python_dispatch` 的测试

### 新增测试

- **源函数单测**：`run_structured_tick` / `run_discrete_tick` / `run_wf_tick` 在 Numba 关时直接调用（hook 传 Python 闭包），断言阶段顺序与 STOP 短路
- **拼装器单测**：`assemble_lifecycle_module` 产物可加载、哨兵替换正确、生成模块行为与源函数一致（同一输入同一输出）
- **负向合同测试**：1 参 hook 被拒绝；无模板文件残留
- **双路径一致性测试**：Numba 开（生成模块）与 Numba 关（源函数）对同一场景输出一致（数值等价）

### 门禁

`pytest` + `pyright`（strict）+ `ruff check src demos`，全过才算完成；公开 API 变更后运行 `python scripts/generate_init_pyi.py`。

---

## 6. 实施顺序建议

1. **阶段函数签名统一**（`(state, config) -> state`）——独立可验证的第一步，同步所有调用点
2. **`execute_csr_event_program_with_state` sperm → Optional**（dummy 消除）
3. **1 参 hook 砍掉**（破坏性变更，尽早）
4. **`engine/lifecycle.py` 源函数 + 拼装器**（核心）
5. **`compile_lifecycle_wrapper(mode, ...)` 改造**（接入拼装）
6. **种群 `run()` 两分支改造**（删 Python 分派）
7. **空间壳迁移 + `_run_python_dispatch_tick` 改造**
8. **删除模板文件 + 清理**

---

## 7. 风险与后续

- **Numba 关记录行为不变**：`_record_current_snapshot` 保持现状（#1 不碰记录语义）。
- **多 tick 循环两份**（`run` vs `_run_loop`）：因记录机制差异（Python 回调 vs njit 数组），合并留给 #5（记录器）。
- **记录层（#5）前置条件**：本次不动的 `_process_kernel_history`、`_record_current_snapshot`、mask 公式 6 处复制，是 #5 的输入。
- **hook 调用形状分叉 bug**：本次彻底消除（唯一调用点 `_run_event` 的 `_EVENT_HOOK(state, config, deme_id)`）。
- **缓存**：生成模块命名/哈希机制不变，缓存键维持现状。
